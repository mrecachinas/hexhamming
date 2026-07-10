use crate::hex::hex_char_to_nibble;
#[cfg(target_arch = "aarch64")]
use crate::ALGO_NEON;
use crate::CURRENT_ALGO;
use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    hamming_distance_string_dispatch_with_max, LOOKUP,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};

use std::sync::atomic::Ordering;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};

// ---------------------------------------------------------------------------
// §5 helper: zero-copy slice from a PyBuffer
// ---------------------------------------------------------------------------

/// Minimum input length (bytes/chars) before it is worth releasing the GIL.
///
/// Releasing and reacquiring the GIL (and, for the string API, copying the
/// input into owned buffers so the closure satisfies `Ungil`) costs on the
/// order of tens to hundreds of nanoseconds. For small inputs — the common
/// case for this library — the distance computation itself is only a few ns,
/// so that overhead dominates. Below this threshold we compute directly on the
/// borrowed bytes while holding the GIL (sound: the borrow is valid for the
/// whole synchronous call). Above it, releasing the GIL lets other Python
/// threads make progress and the copy is amortized.
const GIL_RELEASE_THRESHOLD: usize = 4096;
const ARRAY_GIL_RELEASE_THRESHOLD: usize = 64 * 1024;

#[inline]
fn exact_bytes<'a>(obj: &'a Bound<'_, PyAny>) -> Option<&'a [u8]> {
    obj.cast::<PyBytes>().ok().map(PyBytesMethods::as_bytes)
}

/// Extract a contiguous `PyBuffer<u8>` from any buffer-protocol object.
fn extract_buffer(obj: &Bound<'_, PyAny>) -> PyResult<PyBuffer<u8>> {
    let buf: PyBuffer<u8> = PyBuffer::get(obj)
        .map_err(|_| PyValueError::new_err("error occurred while parsing arguments"))?;
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("input must be contiguous"));
    }
    Ok(buf)
}

/// SAFETY: caller must ensure `buf` is C-contiguous and remains alive for `'a`.
#[inline]
unsafe fn buffer_as_slice<'a>(buf: &'a PyBuffer<u8>) -> &'a [u8] {
    std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.item_count())
}

// ---------------------------------------------------------------------------
// §10 + §14: hamming_distance_string — direct typed params
// ---------------------------------------------------------------------------

/// Calculate the hamming distance of two hexadecimal strings.
///
/// Equivalent to `bin(int(a, 16) ^ int(b, 16)).count('1')` but uses SIMD
/// where available.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_string(py: Python<'_>, a: &str, b: &str) -> PyResult<u64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }
    if a.is_empty() {
        return Ok(0);
    }
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    // Small inputs: compute on the borrowed bytes while holding the GIL — no
    // allocation, no GIL round-trip. The borrows are valid for the whole call.
    if a_bytes.len() < GIL_RELEASE_THRESHOLD {
        return hamming_distance_string_dispatch(a_bytes, b_bytes).map_err(PyValueError::new_err);
    }
    // Large inputs: copy into owned buffers (so the closure is `Ungil`) and
    // release the GIL so other Python threads can run during the computation.
    let a_owned = a_bytes.to_vec();
    let b_owned = b_bytes.to_vec();
    py.detach(move || hamming_distance_string_dispatch(&a_owned, &b_owned))
        .map_err(PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// §5 + §10: hamming_distance_bytes — zero-copy PyBuffer
// ---------------------------------------------------------------------------

/// Calculate the hamming distance of two byte arrays.
///
/// Accepts any buffer-protocol object: `bytes`, `bytearray`, `memoryview`,
/// NumPy `uint8` arrays.
///
/// **WARNING**: mutating a `bytearray` during computation is undefined
/// behavior.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_bytes(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<u64> {
    if let (Some(a_slice), Some(b_slice)) = (exact_bytes(a), exact_bytes(b)) {
        if a_slice.len() != b_slice.len() {
            return Err(PyValueError::new_err("bytes are NOT the same length"));
        }
        if a_slice.is_empty() {
            return Ok(0);
        }
        if a_slice.len() < GIL_RELEASE_THRESHOLD {
            return Ok(hamming_distance_bytes_dispatch(a_slice, b_slice, -1));
        }
    }

    let buf_a = extract_buffer(a)?;
    let buf_b = extract_buffer(b)?;
    // SAFETY: buffers are C-contiguous; PyBuffer pins the Python object.
    let a_slice = unsafe { buffer_as_slice(&buf_a) };
    let b_slice = unsafe { buffer_as_slice(&buf_b) };

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("bytes are NOT the same length"));
    }
    if a_slice.is_empty() {
        return Ok(0);
    }

    // Small inputs: skip the GIL round-trip (already zero-copy via PyBuffer).
    let result = if a_slice.len() < GIL_RELEASE_THRESHOLD {
        hamming_distance_bytes_dispatch(a_slice, b_slice, -1)
    } else {
        py.detach(move || hamming_distance_bytes_dispatch(a_slice, b_slice, -1))
    };
    drop(buf_a);
    drop(buf_b);
    Ok(result)
}

// ---------------------------------------------------------------------------
// §1 + §10: check_hexstrings_within_dist — SIMD for len>=64
// ---------------------------------------------------------------------------

/// Check if two hex strings are within a specified Hamming distance.
///
/// For `len >= 64` uses the SIMD path (full distance then compare).
/// For shorter strings uses scalar with early termination.
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_hexstrings_within_dist(a: &str, b: &str, max_dist: i64) -> PyResult<bool> {
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >0"));
    }
    let max_dist_u64 = max_dist as u64;

    if a.len() != b.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }
    if a == b {
        return Ok(true);
    }
    // Max possible hamming distance per hex char is 4
    if max_dist_u64 >= (a.len() as u64) * 4 {
        return Ok(true);
    }

    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    // §1: SIMD path for long inputs — compute distance with early-exit
    if a_bytes.len() >= 64 {
        let dist = hamming_distance_string_dispatch_with_max(a_bytes, b_bytes, max_dist_u64)
            .map_err(PyValueError::new_err)?;
        return Ok(dist != u64::MAX);
    }

    // Scalar path with early termination for short inputs
    let len = a_bytes.len();
    let mut result: u64 = 0;
    let mut i = 0;

    // Process 4 chars at a time
    while i + 4 <= len {
        unsafe {
            let val1_0 = hex_char_to_nibble(*a_bytes.get_unchecked(i));
            let val2_0 = hex_char_to_nibble(*b_bytes.get_unchecked(i));
            let val1_1 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 1));
            let val2_1 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 1));
            let val1_2 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 2));
            let val2_2 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 2));
            let val1_3 = hex_char_to_nibble(*a_bytes.get_unchecked(i + 3));
            let val2_3 = hex_char_to_nibble(*b_bytes.get_unchecked(i + 3));

            let invalid =
                (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
            if invalid != 0 {
                return Err(PyValueError::new_err("hex string contains invalid char"));
            }

            result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
        }
        if result > max_dist_u64 {
            return Ok(false);
        }
        i += 4;
    }

    // Handle remaining chars
    while i < len {
        unsafe {
            let val1 = hex_char_to_nibble(*a_bytes.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b_bytes.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err(PyValueError::new_err("hex string contains invalid char"));
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        }
        if result > max_dist_u64 {
            return Ok(false);
        }
        i += 1;
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// §5 + §10: check_bytes_within_dist — zero-copy PyBuffer, typed max_dist
// ---------------------------------------------------------------------------

/// Check if two byte arrays are within a specified Hamming distance.
/// Returns `True` if distance <= max_dist, `False` otherwise.
///
/// Accepts any buffer-protocol object.
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_bytes_within_dist(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<bool> {
    if let (Some(a_slice), Some(b_slice)) = (exact_bytes(a), exact_bytes(b)) {
        if a_slice.is_empty() || b_slice.is_empty() {
            return Err(PyValueError::new_err("array size must be >0"));
        }
        if max_dist < 0 {
            return Err(PyValueError::new_err("`max_dist` must be >=0"));
        }
        if a_slice.len() != b_slice.len() {
            return Err(PyValueError::new_err("array sizes need to be the same"));
        }
        if a_slice.len() < GIL_RELEASE_THRESHOLD {
            return Ok(hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist) != u64::MAX);
        }
    }

    let buf_a = extract_buffer(a)?;
    let buf_b = extract_buffer(b)?;
    let a_slice = unsafe { buffer_as_slice(&buf_a) };
    let b_slice = unsafe { buffer_as_slice(&buf_b) };

    if a_slice.is_empty() || b_slice.is_empty() {
        return Err(PyValueError::new_err("array size must be >0"));
    }
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("array sizes need to be the same"));
    }

    let result = if a_slice.len() < GIL_RELEASE_THRESHOLD {
        hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist)
    } else {
        py.detach(move || hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist))
    };
    drop(buf_a);
    drop(buf_b);
    Ok(result != u64::MAX)
}

// ---------------------------------------------------------------------------
// §5 + §10: check_bytes_arrays_within_dist (legacy alias)
// ---------------------------------------------------------------------------

/// Legacy alias for `check_bytes_arrays_first_within_dist`.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<i64> {
    check_bytes_arrays_first_within_dist(py, array_of_elems, elem_to_compare, max_dist)
}

// ---------------------------------------------------------------------------
// §5 + §10: check_bytes_arrays_first_within_dist
// ---------------------------------------------------------------------------

/// Return the index of the first element within a specified Hamming distance,
/// or -1 if none found.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_first_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<i64> {
    if let (Some(big_slice), Some(small_slice)) =
        (exact_bytes(array_of_elems), exact_bytes(elem_to_compare))
    {
        if small_slice.is_empty() {
            return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
        }
        if max_dist < 0 {
            return Err(PyValueError::new_err("`max_dist` must be >=0"));
        }
        if big_slice.len() % small_slice.len() != 0 {
            return Err(PyValueError::new_err(
                "`array_of_elems` size must be multiplier of `elem_to_compare`",
            ));
        }
        if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
            return Ok(
                crate::bytes_array_first_within_dist(big_slice, small_slice, max_dist)
                    .ok()
                    .flatten()
                    .map(|i| i as i64)
                    .unwrap_or(-1),
            );
        }
    }

    let buf_big = extract_buffer(array_of_elems)?;
    let buf_small = extract_buffer(elem_to_compare)?;
    let big_slice = unsafe { buffer_as_slice(&buf_big) };
    let small_slice = unsafe { buffer_as_slice(&buf_small) };

    if small_slice.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_slice.len() % small_slice.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let calculate = || {
        // `first` is always serial because its early exit beats parallel setup
        // for early and common matches.
        crate::bytes_array_first_within_dist(big_slice, small_slice, max_dist)
            .ok()
            .flatten()
            .map(|i| i as i64)
            .unwrap_or(-1)
    };
    let result = if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        calculate()
    } else {
        py.detach(calculate)
    };
    drop(buf_big);
    drop(buf_small);
    Ok(result)
}

// ---------------------------------------------------------------------------
// §5 + §10: check_bytes_arrays_best_within_dist
// ---------------------------------------------------------------------------

/// Find the element with the smallest Hamming distance.
/// Returns `(best_distance, best_index)`, or `(-1, -1)` if none within
/// max_dist.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_best_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<(i64, i64)> {
    if let (Some(big_slice), Some(small_slice)) =
        (exact_bytes(array_of_elems), exact_bytes(elem_to_compare))
    {
        if small_slice.is_empty() {
            return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
        }
        if max_dist < 0 {
            return Err(PyValueError::new_err("`max_dist` must be >=0"));
        }
        if big_slice.len() % small_slice.len() != 0 {
            return Err(PyValueError::new_err(
                "`array_of_elems` size must be multiplier of `elem_to_compare`",
            ));
        }
        if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
            return Ok(
                crate::bytes_array_best_within_dist(big_slice, small_slice, max_dist)
                    .ok()
                    .flatten()
                    .map(|(distance, index)| (distance as i64, index as i64))
                    .unwrap_or((-1, -1)),
            );
        }
    }

    let buf_big = extract_buffer(array_of_elems)?;
    let buf_small = extract_buffer(elem_to_compare)?;
    let big_slice = unsafe { buffer_as_slice(&buf_big) };
    let small_slice = unsafe { buffer_as_slice(&buf_small) };

    if small_slice.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_slice.len() % small_slice.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let calculate = || {
        crate::bytes_array_best_within_dist(big_slice, small_slice, max_dist)
            .ok()
            .flatten()
            .map(|(d, i)| (d as i64, i as i64))
            .unwrap_or((-1, -1))
    };
    let result = if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        calculate()
    } else {
        py.detach(calculate)
    };
    drop(buf_big);
    drop(buf_small);
    Ok(result)
}

// ---------------------------------------------------------------------------
// §5 + §10 + §12: check_bytes_arrays_all_within_dist
// ---------------------------------------------------------------------------

/// Find all elements within a specified Hamming distance.
/// Returns list of `(distance, index)` tuples.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_all_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<Vec<(u64, u64)>> {
    if let (Some(big_slice), Some(small_slice)) =
        (exact_bytes(array_of_elems), exact_bytes(elem_to_compare))
    {
        if small_slice.is_empty() {
            return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
        }
        if max_dist < 0 {
            return Err(PyValueError::new_err("`max_dist` must be >=0"));
        }
        if big_slice.len() % small_slice.len() != 0 {
            return Err(PyValueError::new_err(
                "`array_of_elems` size must be multiplier of `elem_to_compare`",
            ));
        }
        if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
            return Ok(
                crate::bytes_array_all_within_dist(big_slice, small_slice, max_dist)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(distance, index)| (distance, index as u64))
                    .collect(),
            );
        }
    }

    let buf_big = extract_buffer(array_of_elems)?;
    let buf_small = extract_buffer(elem_to_compare)?;
    let big_slice = unsafe { buffer_as_slice(&buf_big) };
    let small_slice = unsafe { buffer_as_slice(&buf_small) };

    if small_slice.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_slice.len() % small_slice.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let calculate = || {
        crate::bytes_array_all_within_dist(big_slice, small_slice, max_dist)
            .unwrap_or_default()
            .into_iter()
            .map(|(d, i)| (d, i as u64))
            .collect::<Vec<(u64, u64)>>()
    };
    let results = if big_slice.len() < ARRAY_GIL_RELEASE_THRESHOLD {
        calculate()
    } else {
        py.detach(calculate)
    };
    drop(buf_big);
    drop(buf_small);
    Ok(results)
}

// ---------------------------------------------------------------------------
// §9 + §14: set_algo — delegate to api::set_algorithm
// ---------------------------------------------------------------------------

/// Change the SIMD algorithm used for calculations.
///
/// Returns empty string on success, or error message on failure.
///
/// **NOTE**: error-as-empty-string is legacy behaviour and will change to
/// raise `ValueError` in the next major version.
#[pyfunction]
fn set_algo(algo_name: &str) -> PyResult<String> {
    match crate::api::set_algorithm(algo_name) {
        Ok(()) => Ok(String::new()),
        Err(msg) => Ok(msg.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

/// Module for calculating hamming distance of two hexadecimal strings
#[pymodule]
fn hexhamming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "3.0.0")?;
    m.add_function(wrap_pyfunction!(hamming_distance_string, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(check_hexstrings_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_first_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_best_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_all_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(set_algo, m)?)?;

    // Auto-detect best algorithm on module load
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
            CURRENT_ALGO.store(ALGO_AVX512, Ordering::Relaxed);
        } else if is_x86_feature_detected!("avx2") {
            CURRENT_ALGO.store(ALGO_AVX2, Ordering::Relaxed);
        } else if is_x86_feature_detected!("sse4.1") {
            CURRENT_ALGO.store(ALGO_SSE41, Ordering::Relaxed);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
    }

    Ok(())
}
