use crate::hex::hex_char_to_nibble;
use crate::CURRENT_ALGO;
use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    hamming_distance_string_dispatch_with_max, LOOKUP,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX2, ALGO_AVX512, ALGO_SSE41};

use std::ffi::CStr;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::sync::atomic::Ordering;

use pyo3::buffer::Element;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};

// ---------------------------------------------------------------------------
// §5 helper: zero-copy slice from the Python buffer protocol
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
const STRING_GIL_RELEASE_THRESHOLD: usize = 4096;
const BYTES_GIL_RELEASE_THRESHOLD: usize = 16 * 1024;
const ARRAY_GIL_RELEASE_THRESHOLD: usize = 64 * 1024;

#[inline]
fn exact_bytes<'a>(obj: &'a Bound<'_, PyAny>) -> Option<&'a [u8]> {
    obj.cast::<PyBytes>().ok().map(PyBytesMethods::as_bytes)
}

struct SimpleByteBuffer {
    raw: ffi::Py_buffer,
    acquired: bool,
    _pin: PhantomPinned,
}

impl SimpleByteBuffer {
    fn new() -> Self {
        Self {
            raw: ffi::Py_buffer::new(),
            acquired: false,
            _pin: PhantomPinned,
        }
    }

    fn acquire(mut self: Pin<&mut Self>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let this = unsafe { self.as_mut().get_unchecked_mut() };
        let result = unsafe {
            ffi::PyObject_GetBuffer(
                obj.as_ptr(),
                &mut this.raw,
                ffi::PyBUF_ND | ffi::PyBUF_FORMAT,
            )
        };
        if result != 0 {
            let _ = PyErr::fetch(obj.py());
            let message = if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } != 0 {
                "input must be contiguous"
            } else {
                "error occurred while parsing arguments"
            };
            return Err(PyValueError::new_err(message));
        }
        this.acquired = true;

        let compatible_format = if this.raw.format.is_null() {
            false
        } else {
            let format = unsafe { CStr::from_ptr(this.raw.format) };
            <u8 as Element>::is_compatible_format(format)
        };
        if this.raw.itemsize != 1 || !compatible_format {
            return Err(PyValueError::new_err(
                "error occurred while parsing arguments",
            ));
        }
        if this.raw.len < 0 || (this.raw.buf.is_null() && this.raw.len != 0) {
            return Err(PyValueError::new_err("invalid buffer view"));
        }
        Ok(())
    }

    /// SAFETY: the pinned guard keeps the exported buffer alive and stationary
    /// for the returned slice's lifetime.
    #[inline]
    unsafe fn as_slice(self: Pin<&Self>) -> &[u8] {
        let raw = &self.get_ref().raw;
        if raw.len == 0 {
            return &[];
        }
        std::slice::from_raw_parts(raw.buf as *const u8, raw.len as usize)
    }
}

impl Drop for SimpleByteBuffer {
    fn drop(&mut self) {
        if self.acquired {
            let _ = Python::try_attach(|_| unsafe {
                ffi::PyBuffer_Release(&mut self.raw);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Writable buffer guard — used by `*_into` batch APIs.
// ---------------------------------------------------------------------------

struct WritableByteBuffer {
    raw: ffi::Py_buffer,
    acquired: bool,
    _pin: PhantomPinned,
}

impl WritableByteBuffer {
    fn new() -> Self {
        Self {
            raw: ffi::Py_buffer::new(),
            acquired: false,
            _pin: PhantomPinned,
        }
    }

    fn acquire(mut self: Pin<&mut Self>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let this = unsafe { self.as_mut().get_unchecked_mut() };
        // C-contiguous + writable + format keeps this API strict: caller-side
        // non-contiguous or readonly buffers are rejected up front rather than
        // producing surprising partial writes.
        let result = unsafe {
            ffi::PyObject_GetBuffer(
                obj.as_ptr(),
                &mut this.raw,
                ffi::PyBUF_C_CONTIGUOUS | ffi::PyBUF_FORMAT | ffi::PyBUF_WRITABLE,
            )
        };
        if result != 0 {
            let _ = PyErr::fetch(obj.py());
            let message = if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } != 0 {
                "output must be a writable, C-contiguous buffer"
            } else {
                "output must support the buffer protocol"
            };
            return Err(PyValueError::new_err(message));
        }
        this.acquired = true;

        if this.raw.readonly != 0 {
            return Err(PyValueError::new_err("output must be writable"));
        }
        let compatible_format = if this.raw.format.is_null() {
            true
        } else {
            let format = unsafe { CStr::from_ptr(this.raw.format) };
            <u8 as Element>::is_compatible_format(format)
        };
        if this.raw.itemsize != 1 || !compatible_format {
            return Err(PyValueError::new_err(
                "output must be a byte buffer (itemsize 1)",
            ));
        }
        if this.raw.len < 0 || (this.raw.buf.is_null() && this.raw.len != 0) {
            return Err(PyValueError::new_err("invalid output buffer"));
        }
        Ok(())
    }

    #[inline]
    fn len(&self) -> usize {
        self.raw.len as usize
    }

    /// SAFETY: caller must not alias the returned pointer with any other
    /// borrow, and the guard must outlive all use of the pointer.
    #[inline]
    unsafe fn raw_mut_ptr(self: Pin<&mut Self>) -> *mut u8 {
        let this = self.get_unchecked_mut();
        if this.raw.len == 0 {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            this.raw.buf as *mut u8
        }
    }
}

impl Drop for WritableByteBuffer {
    fn drop(&mut self) {
        if self.acquired {
            let _ = Python::try_attach(|_| unsafe {
                ffi::PyBuffer_Release(&mut self.raw);
            });
        }
    }
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
    if a_bytes.len() < STRING_GIL_RELEASE_THRESHOLD {
        return hamming_distance_string_dispatch(a_bytes, b_bytes).map_err(PyValueError::new_err);
    }
    // Python strings are immutable and remain borrowed for the whole call, so
    // their byte slices can be processed while the interpreter is detached.
    py.detach(|| hamming_distance_string_dispatch(a_bytes, b_bytes))
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
        if a_slice.len() < BYTES_GIL_RELEASE_THRESHOLD {
            return Ok(hamming_distance_bytes_dispatch(a_slice, b_slice, -1));
        }
    }

    let mut buf_a = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_b = std::pin::pin!(SimpleByteBuffer::new());
    buf_a.as_mut().acquire(a)?;
    buf_b.as_mut().acquire(b)?;
    let a_slice = unsafe { buf_a.as_ref().as_slice() };
    let b_slice = unsafe { buf_b.as_ref().as_slice() };

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("bytes are NOT the same length"));
    }
    if a_slice.is_empty() {
        return Ok(0);
    }

    // Small inputs: skip the GIL round-trip (already zero-copy via PyBuffer).
    let result = if a_slice.len() < BYTES_GIL_RELEASE_THRESHOLD {
        hamming_distance_bytes_dispatch(a_slice, b_slice, -1)
    } else {
        py.detach(move || hamming_distance_bytes_dispatch(a_slice, b_slice, -1))
    };
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
        if a_slice.len() < BYTES_GIL_RELEASE_THRESHOLD {
            return Ok(hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist) != u64::MAX);
        }
    }

    let mut buf_a = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_b = std::pin::pin!(SimpleByteBuffer::new());
    buf_a.as_mut().acquire(a)?;
    buf_b.as_mut().acquire(b)?;
    let a_slice = unsafe { buf_a.as_ref().as_slice() };
    let b_slice = unsafe { buf_b.as_ref().as_slice() };

    if a_slice.is_empty() || b_slice.is_empty() {
        return Err(PyValueError::new_err("array size must be >0"));
    }
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err("array sizes need to be the same"));
    }

    let result = if a_slice.len() < BYTES_GIL_RELEASE_THRESHOLD {
        hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist)
    } else {
        py.detach(move || hamming_distance_bytes_dispatch(a_slice, b_slice, max_dist))
    };
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

    let mut buf_big = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_small = std::pin::pin!(SimpleByteBuffer::new());
    buf_big.as_mut().acquire(array_of_elems)?;
    buf_small.as_mut().acquire(elem_to_compare)?;
    let big_slice = unsafe { buf_big.as_ref().as_slice() };
    let small_slice = unsafe { buf_small.as_ref().as_slice() };

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

    let mut buf_big = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_small = std::pin::pin!(SimpleByteBuffer::new());
    buf_big.as_mut().acquire(array_of_elems)?;
    buf_small.as_mut().acquire(elem_to_compare)?;
    let big_slice = unsafe { buf_big.as_ref().as_slice() };
    let small_slice = unsafe { buf_small.as_ref().as_slice() };

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

    let mut buf_big = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_small = std::pin::pin!(SimpleByteBuffer::new());
    buf_big.as_mut().acquire(array_of_elems)?;
    buf_small.as_mut().acquire(elem_to_compare)?;
    let big_slice = unsafe { buf_big.as_ref().as_slice() };
    let small_slice = unsafe { buf_small.as_ref().as_slice() };

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
    Ok(results)
}

// ---------------------------------------------------------------------------
// Batch APIs — amortize the Python↔Rust boundary across many distance calls.
// ---------------------------------------------------------------------------

/// Acquire read-only buffers for `a` and `b`, invoking the closure with their
/// slices. Both buffers live for the whole call (dropped on return).
#[inline]
fn with_two_readonly_buffers<F, R>(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&[u8], &[u8], bool) -> PyResult<R>,
{
    if let (Some(a_slice), Some(b_slice)) = (exact_bytes(a), exact_bytes(b)) {
        return f(a_slice, b_slice, true);
    }
    let mut buf_a = std::pin::pin!(SimpleByteBuffer::new());
    let mut buf_b = std::pin::pin!(SimpleByteBuffer::new());
    buf_a.as_mut().acquire(a)?;
    buf_b.as_mut().acquire(b)?;
    let a_slice = unsafe { buf_a.as_ref().as_slice() };
    let b_slice = unsafe { buf_b.as_ref().as_slice() };
    // General buffer-protocol inputs may be writable. Keep the GIL attached
    // while reading them so another Python thread cannot mutate the storage.
    f(a_slice, b_slice, false)
}

#[inline]
fn buffer_ranges_overlap(a_ptr: *const u8, a_len: usize, b_ptr: *const u8, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a_start = a_ptr as usize;
    let b_start = b_ptr as usize;
    let a_end = a_start.checked_add(a_len).unwrap_or(usize::MAX);
    let b_end = b_start.checked_add(b_len).unwrap_or(usize::MAX);
    a_start < b_end && b_start < a_end
}

/// Compute Hamming distances between corresponding fixed-width records in `a`
/// and `b`. Returns a list of `int` distances, one per record.
///
/// `a` and `b` must be equal-length buffer-protocol objects whose length is a
/// multiple of `element_size`.
#[pyfunction]
#[pyo3(signature = (a, b, element_size))]
fn hamming_distances_bytes(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    element_size: usize,
) -> PyResult<Vec<u64>> {
    with_two_readonly_buffers(a, b, |a_slice, b_slice, can_detach| {
        // Small batches stay attached; large batches detach the GIL. The raw
        // slices remain valid because the Py_buffer guards outlive `f`.
        let compute = || {
            crate::bytes_pairwise_distances(a_slice, b_slice, element_size)
                .map_err(PyValueError::new_err)
        };
        if can_detach && a_slice.len() >= ARRAY_GIL_RELEASE_THRESHOLD {
            py.detach(compute)
        } else {
            compute()
        }
    })
}

/// Compute Hamming distances and return them as `bytes` of little-endian u64
/// values (8 bytes per distance).
#[pyfunction]
#[pyo3(signature = (a, b, element_size))]
fn hamming_distances_bytes_packed<'py>(
    py: Python<'py>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    element_size: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let distances = with_two_readonly_buffers(a, b, |a_slice, b_slice, can_detach| {
        let compute = || {
            crate::bytes_pairwise_distances(a_slice, b_slice, element_size)
                .map_err(PyValueError::new_err)
        };
        if can_detach && a_slice.len() >= ARRAY_GIL_RELEASE_THRESHOLD {
            py.detach(compute)
        } else {
            compute()
        }
    })?;
    let byte_len = distances
        .len()
        .checked_mul(8)
        .ok_or_else(|| PyValueError::new_err("distance buffer size overflows platform usize"))?;
    PyBytes::new_with(py, byte_len, |dst| {
        // SAFETY: allocated exactly distances.len()*8 bytes; write_unaligned
        // supports arbitrary alignment of dst.
        let ptr = dst.as_mut_ptr();
        for (i, d) in distances.iter().enumerate() {
            unsafe {
                std::ptr::write_unaligned(ptr.add(i * 8) as *mut u64, d.to_le());
            }
        }
        Ok(())
    })
}

/// Compute Hamming distances and write them as little-endian u64 values into
/// `output`. Returns the number of distances written (== `len(a)//element_size`).
///
/// `output` must be a writable, C-contiguous byte buffer of exactly
/// `count * 8` bytes. Non-contiguous, read-only, or wrong-size outputs are
/// rejected with `ValueError`.
#[pyfunction]
#[pyo3(signature = (a, b, element_size, output))]
fn hamming_distances_bytes_into(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    element_size: usize,
    output: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    // Acquire input buffers first (may be bytes fast-path or general buffer),
    // then the writable output buffer. Any Bound<'_, PyAny> references stay
    // out of the detach closure — only raw slices and pointers cross the GIL
    // boundary, which the pinned guards keep alive for the whole call.
    let mut buf_out = std::pin::pin!(WritableByteBuffer::new());
    buf_out.as_mut().acquire(output)?;
    let out_len = buf_out.len();
    // SAFETY: pointer captured while GIL is still held; the pinned guard keeps
    // the buffer exported for the entire call including any detached region.
    let out_ptr = unsafe { buf_out.as_mut().raw_mut_ptr() };
    // Passing raw pointers into `py.detach` requires the closure to satisfy
    // `Ungil`. `*mut u8` is `!Sync`, so shipping the value as a `usize` and
    // rematerializing it inside the closure keeps the closure `Ungil`-clean.
    let out_ptr_addr = out_ptr as usize;

    with_two_readonly_buffers(a, b, |a_slice, b_slice, can_detach| {
        let compute = || -> PyResult<usize> {
            if buffer_ranges_overlap(
                a_slice.as_ptr(),
                a_slice.len(),
                out_ptr_addr as *const u8,
                out_len,
            ) || buffer_ranges_overlap(
                b_slice.as_ptr(),
                b_slice.len(),
                out_ptr_addr as *const u8,
                out_len,
            ) {
                return Err(PyValueError::new_err(
                    "output buffer must not overlap input buffers",
                ));
            }
            // SAFETY: exclusive access to the writable buffer for the closure
            // duration; the pinned guard keeps the memory alive.
            let out_slice =
                unsafe { std::slice::from_raw_parts_mut(out_ptr_addr as *mut u8, out_len) };
            crate::bytes_pairwise_distances_into(a_slice, b_slice, element_size, out_slice)
                .map_err(PyValueError::new_err)
        };
        if can_detach && a_slice.len() >= ARRAY_GIL_RELEASE_THRESHOLD {
            py.detach(compute)
        } else {
            compute()
        }
    })
}

// ---------------------------------------------------------------------------
// Multi-query catalog scans
// ---------------------------------------------------------------------------

#[inline]
fn resolve_and_dispatch_multi<F, R>(
    py: Python<'_>,
    catalog: &Bound<'_, PyAny>,
    queries: &Bound<'_, PyAny>,
    query_width: usize,
    max_dist: i64,
    compute: F,
) -> PyResult<R>
where
    F: FnOnce(&[u8], &[u8], usize, i64) -> Result<R, &'static str> + Send,
    R: Send,
{
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    with_two_readonly_buffers(
        catalog,
        queries,
        |catalog_slice, queries_slice, can_detach| {
            // Detach when total work (catalog * queries scan) is non-trivial.
            // catalog.len() is a proxy since queries usually << catalog.
            let compute_call = || compute(catalog_slice, queries_slice, query_width, max_dist);
            let total_work = catalog_slice.len().saturating_mul(queries_slice.len());
            let result = if can_detach && total_work >= 4 * ARRAY_GIL_RELEASE_THRESHOLD {
                py.detach(compute_call)
            } else {
                compute_call()
            };
            result.map_err(PyValueError::new_err)
        },
    )
}

/// Run [`check_bytes_arrays_first_within_dist`] against `catalog` for every
/// fixed-width slice of `queries`. Returns a list of `int` indices (or `-1`
/// when no record matches), one per query.
#[pyfunction]
#[pyo3(signature = (catalog, queries, query_width, max_dist))]
fn check_bytes_arrays_first_many_within_dist(
    py: Python<'_>,
    catalog: &Bound<'_, PyAny>,
    queries: &Bound<'_, PyAny>,
    query_width: usize,
    max_dist: i64,
) -> PyResult<Vec<i64>> {
    resolve_and_dispatch_multi(py, catalog, queries, query_width, max_dist, |c, q, w, m| {
        let results = crate::bytes_array_first_many_within_dist(c, q, w, m)?;
        Ok(results
            .into_iter()
            .map(|r| r.map(|i| i as i64).unwrap_or(-1))
            .collect())
    })
}

/// Run [`check_bytes_arrays_best_within_dist`] against `catalog` for every
/// fixed-width slice of `queries`. Returns a list of `(distance, index)`
/// tuples, using `(-1, -1)` for queries with no match.
#[pyfunction]
#[pyo3(signature = (catalog, queries, query_width, max_dist))]
fn check_bytes_arrays_best_many_within_dist(
    py: Python<'_>,
    catalog: &Bound<'_, PyAny>,
    queries: &Bound<'_, PyAny>,
    query_width: usize,
    max_dist: i64,
) -> PyResult<Vec<(i64, i64)>> {
    resolve_and_dispatch_multi(py, catalog, queries, query_width, max_dist, |c, q, w, m| {
        let results = crate::bytes_array_best_many_within_dist(c, q, w, m)?;
        Ok(results
            .into_iter()
            .map(|r| r.map(|(d, i)| (d as i64, i as i64)).unwrap_or((-1, -1)))
            .collect())
    })
}

/// Run [`check_bytes_arrays_all_within_dist`] against `catalog` for every
/// fixed-width slice of `queries`. Returns a list of lists of
/// `(distance, index)` tuples, one inner list per query.
#[pyfunction]
#[pyo3(signature = (catalog, queries, query_width, max_dist))]
fn check_bytes_arrays_all_many_within_dist(
    py: Python<'_>,
    catalog: &Bound<'_, PyAny>,
    queries: &Bound<'_, PyAny>,
    query_width: usize,
    max_dist: i64,
) -> PyResult<Vec<Vec<(u64, u64)>>> {
    resolve_and_dispatch_multi(py, catalog, queries, query_width, max_dist, |c, q, w, m| {
        let results = crate::bytes_array_all_many_within_dist(c, q, w, m)?;
        Ok(results
            .into_iter()
            .map(|matches| {
                matches
                    .into_iter()
                    .map(|(d, i)| (d, i as u64))
                    .collect::<Vec<(u64, u64)>>()
            })
            .collect())
    })
}

// ---------------------------------------------------------------------------
// Packed/into transport for dense all-results
// ---------------------------------------------------------------------------

/// Dense-transport variant of `check_bytes_arrays_all_within_dist`.
///
/// Returns a `(distance_bytes, index_bytes)` tuple. `distance_bytes` is a
/// contiguous little-endian u16 buffer (2 bytes per match); `index_bytes` is a
/// contiguous little-endian u32 buffer (4 bytes per match). Matches are
/// ordered by ascending catalog index. Fails if the element width would allow
/// distances exceeding `u16::MAX` bits, or if the catalog exceeds
/// `u32::MAX` records.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_all_within_dist_packed<'py>(
    py: Python<'py>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    let (dists, idxs) =
        with_two_readonly_buffers(array_of_elems, elem_to_compare, |big, small, can_detach| {
            let compute = || {
                crate::bytes_array_all_within_dist_packed(big, small, max_dist)
                    .map_err(PyValueError::new_err)
            };
            if can_detach && big.len() >= ARRAY_GIL_RELEASE_THRESHOLD {
                py.detach(compute)
            } else {
                compute()
            }
        })?;
    let d_byte_len = dists
        .len()
        .checked_mul(2)
        .ok_or_else(|| PyValueError::new_err("distance buffer size overflows platform usize"))?;
    let i_byte_len = idxs
        .len()
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("index buffer size overflows platform usize"))?;
    let d_bytes = PyBytes::new_with(py, d_byte_len, |dst| {
        let ptr = dst.as_mut_ptr();
        for (i, v) in dists.iter().enumerate() {
            unsafe {
                std::ptr::write_unaligned(ptr.add(i * 2) as *mut u16, v.to_le());
            }
        }
        Ok(())
    })?;
    let i_bytes = PyBytes::new_with(py, i_byte_len, |dst| {
        let ptr = dst.as_mut_ptr();
        for (i, v) in idxs.iter().enumerate() {
            unsafe {
                std::ptr::write_unaligned(ptr.add(i * 4) as *mut u32, v.to_le());
            }
        }
        Ok(())
    })?;
    Ok((d_bytes, i_bytes))
}

/// Write `check_bytes_arrays_all_within_dist` results into caller-provided
/// writable buffers. `out_distances` receives little-endian u16 distances (2
/// bytes per match); `out_indices` receives little-endian u32 indices (4
/// bytes per match). Returns the number of matches written.
///
/// Both output buffers must be sized for the worst case (`num_records * 2`
/// and `num_records * 4` bytes respectively); if the resolved match count
/// exceeds the capacity, `ValueError` is raised.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist, out_distances, out_indices))]
fn check_bytes_arrays_all_within_dist_into(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: i64,
    out_distances: &Bound<'_, PyAny>,
    out_indices: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    let mut buf_d = std::pin::pin!(WritableByteBuffer::new());
    let mut buf_i = std::pin::pin!(WritableByteBuffer::new());
    buf_d.as_mut().acquire(out_distances)?;
    buf_i.as_mut().acquire(out_indices)?;
    let d_len = buf_d.len();
    let i_len = buf_i.len();
    // SAFETY: pointers captured under GIL; pinned guards outlive detached use.
    let d_ptr_addr = unsafe { buf_d.as_mut().raw_mut_ptr() } as usize;
    let i_ptr_addr = unsafe { buf_i.as_mut().raw_mut_ptr() } as usize;
    if buffer_ranges_overlap(
        d_ptr_addr as *const u8,
        d_len,
        i_ptr_addr as *const u8,
        i_len,
    ) {
        return Err(PyValueError::new_err(
            "output buffers must not overlap each other",
        ));
    }

    with_two_readonly_buffers(array_of_elems, elem_to_compare, |big, small, can_detach| {
        let compute = || -> PyResult<usize> {
            let d_ptr = d_ptr_addr as *mut u8;
            let i_ptr = i_ptr_addr as *mut u8;
            for (ptr, len) in [(d_ptr as *const u8, d_len), (i_ptr as *const u8, i_len)] {
                if buffer_ranges_overlap(big.as_ptr(), big.len(), ptr, len)
                    || buffer_ranges_overlap(small.as_ptr(), small.len(), ptr, len)
                {
                    return Err(PyValueError::new_err(
                        "output buffers must not overlap input buffers",
                    ));
                }
            }
            let width = small.len();
            if width == 0 {
                return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
            }
            if big.len() % width != 0 {
                return Err(PyValueError::new_err(
                    "`array_of_elems` size must be multiplier of `elem_to_compare`",
                ));
            }
            let num_records = big.len() / width;
            let max_bits = (width as u64).saturating_mul(8);
            if max_bits > u16::MAX as u64 {
                return Err(PyValueError::new_err(
                    "element width too large for u16 packed distances",
                ));
            }
            if num_records > u32::MAX as usize {
                return Err(PyValueError::new_err(
                    "catalog record count exceeds u32::MAX",
                ));
            }
            // Worst-case capacity check up front so partial writes are never
            // observable.
            let required_distances = num_records.checked_mul(2).ok_or_else(|| {
                PyValueError::new_err("distance buffer size overflows platform usize")
            })?;
            let required_indices = num_records.checked_mul(4).ok_or_else(|| {
                PyValueError::new_err("index buffer size overflows platform usize")
            })?;
            if d_len < required_distances {
                return Err(PyValueError::new_err(
                    "out_distances must have capacity for num_records * 2 bytes",
                ));
            }
            if i_len < required_indices {
                return Err(PyValueError::new_err(
                    "out_indices must have capacity for num_records * 4 bytes",
                ));
            }
            let matches = crate::bytes_array_all_within_dist(big, small, max_dist)
                .map_err(PyValueError::new_err)?;
            for (k, (d, idx)) in matches.iter().enumerate() {
                unsafe {
                    std::ptr::write_unaligned(d_ptr.add(k * 2) as *mut u16, (*d as u16).to_le());
                    std::ptr::write_unaligned(i_ptr.add(k * 4) as *mut u32, (*idx as u32).to_le());
                }
            }
            Ok(matches.len())
        };
        if can_detach && big.len() >= ARRAY_GIL_RELEASE_THRESHOLD {
            py.detach(compute)
        } else {
            compute()
        }
    })
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
    m.add_function(wrap_pyfunction!(hamming_distances_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distances_bytes_packed, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distances_bytes_into, m)?)?;
    m.add_function(wrap_pyfunction!(
        check_bytes_arrays_first_many_within_dist,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        check_bytes_arrays_best_many_within_dist,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        check_bytes_arrays_all_many_within_dist,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        check_bytes_arrays_all_within_dist_packed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        check_bytes_arrays_all_within_dist_into,
        m
    )?)?;
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
        // LLVM's auto-vectorized native byte loop is faster than the hand-NEON
        // implementation on Apple Silicon. Hex dispatch still selects NEON.
        CURRENT_ALGO.store(crate::ALGO_NATIVE, Ordering::Relaxed);
    }

    Ok(())
}
