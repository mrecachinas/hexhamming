use crate::hex::hex_char_to_nibble;
#[cfg(target_arch = "aarch64")]
use crate::ALGO_NEON;
use crate::{hamming_distance_bytes_dispatch, hamming_distance_string_dispatch, LOOKUP};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};
use crate::{ALGO_CLASSIC, ALGO_NATIVE, CURRENT_ALGO};

use std::sync::atomic::Ordering;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// §5 helper: zero-copy slice from a PyBuffer
// ---------------------------------------------------------------------------

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
    // §5: strings keep .to_vec() since &str is not buffer-protocol
    let a_owned = a.as_bytes().to_vec();
    let b_owned = b.as_bytes().to_vec();
    py.allow_threads(move || hamming_distance_string_dispatch(&a_owned, &b_owned))
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

    let result = py.allow_threads(move || hamming_distance_bytes_dispatch(a_slice, b_slice, -1));
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

    // §1: SIMD path for long inputs — compute full distance then compare
    if a_bytes.len() >= 64 {
        let dist =
            hamming_distance_string_dispatch(a_bytes, b_bytes).map_err(PyValueError::new_err)?;
        return Ok(dist <= max_dist_u64);
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

    let result =
        py.allow_threads(move || hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist));
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

    let result = py.allow_threads(move || {
        let elem_size = small_slice.len();
        let num_elements = big_slice.len() / elem_size;
        for i in 0..num_elements {
            let chunk = &big_slice[i * elem_size..(i + 1) * elem_size];
            let d = hamming_distance_bytes_dispatch(chunk, small_slice, max_dist);
            if d != u64::MAX {
                return i as i64;
            }
        }
        -1i64
    });
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

    let result = py.allow_threads(move || {
        let elem_size = small_slice.len();
        let num_elements = big_slice.len() / elem_size;
        let mut best_dist: Option<u64> = None;
        let mut best_index: i64 = -1;

        for i in 0..num_elements {
            let chunk = &big_slice[i * elem_size..(i + 1) * elem_size];
            let threshold = best_dist.map(|d| (d as i64) - 1).unwrap_or(max_dist);
            let d = hamming_distance_bytes_dispatch(chunk, small_slice, threshold);
            if d == u64::MAX {
                continue;
            }
            if best_dist.is_none() || d < best_dist.unwrap() {
                best_dist = Some(d);
                best_index = i as i64;
            }
        }
        (best_dist.map(|d| d as i64).unwrap_or(-1), best_index)
    });
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

    let results = py.allow_threads(move || {
        let elem_size = small_slice.len();
        let num_elements = big_slice.len() / elem_size;
        // §12: pre-allocate with bounded capacity
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(std::cmp::min(num_elements, 4096));

        for i in 0..num_elements {
            let chunk = &big_slice[i * elem_size..(i + 1) * elem_size];
            let d = hamming_distance_bytes_dispatch(chunk, small_slice, max_dist);
            if d != u64::MAX {
                out.push((d, i as u64));
            }
        }
        out
    });
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
