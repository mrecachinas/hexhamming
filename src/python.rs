use crate::hex::hex_char_to_nibble;
use crate::{
    hamming_distance_bytes_dispatch, hamming_distance_string_dispatch,
    ALGO_CLASSIC, ALGO_NATIVE,
    CURRENT_ALGO, LOOKUP,
};
#[cfg(target_arch = "x86_64")]
use crate::{ALGO_AVX512, ALGO_AVX2, ALGO_SSE41};
#[cfg(target_arch = "aarch64")]
use crate::ALGO_NEON;

use std::sync::atomic::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Calculate the hamming distance of two hexadecimal strings
///
/// This is equivalent to `bin(int(a, 16) ^ int(b, 16)).count('1')`
/// but optimized using SIMD instructions where available.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_string(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
    let a_str: &str = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_str: &str = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_str.len() != b_str.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }

    if a_str.is_empty() {
        return Ok(0);
    }

    let a_owned = a_str.as_bytes().to_vec();
    let b_owned = b_str.as_bytes().to_vec();
    py.allow_threads(move || {
        hamming_distance_string_dispatch(&a_owned, &b_owned)
    }).map_err(PyValueError::new_err)
}

/// Calculate the hamming distance of two byte arrays
#[pyfunction]
#[pyo3(signature = (a, b))]
fn hamming_distance_bytes(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
    let a_bytes: &[u8] = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_bytes: &[u8] = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_bytes.len() != b_bytes.len() {
        return Err(PyValueError::new_err("bytes are NOT the same length"));
    }

    if a_bytes.is_empty() {
        return Ok(0);
    }

    let a_owned = a_bytes.to_vec();
    let b_owned = b_bytes.to_vec();
    Ok(py.allow_threads(move || {
        hamming_distance_bytes_dispatch(&a_owned, &b_owned, -1)
    }))
}

/// Check if two hex strings are within a specified Hamming distance
/// Optimized with early termination and branchless parsing
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_hexstrings_within_dist(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // Extract strings with proper error handling
    let a_str: &str = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_str: &str = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    // Extract max_dist - need to handle negative numbers
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >0"));
    }

    let max_dist_u64 = max_dist_val as u64;

    if a_str.len() != b_str.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }

    if a_str == b_str {
        return Ok(true);
    }

    // Max possible hamming distance per hex char is 4, so if max_dist >= 4*len, always true
    if max_dist_u64 >= (a_str.len() as u64) * 4 {
        return Ok(true);
    }

    let a_bytes = a_str.as_bytes();
    let b_bytes = b_str.as_bytes();
    let len = a_bytes.len();

    let mut result: u64 = 0;
    let mut i = 0;

    // Process 4 chars at a time for better throughput
    // SAFETY: bounds checked by loop condition
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
            
            // Validate all 8 values
            let invalid = (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
            if invalid != 0 {
                return Err(PyValueError::new_err("hex string contains invalid char"));
            }
            
            result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
        }
        
        // Early termination check
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

/// Check if two byte arrays are within a specified Hamming distance
/// Returns True if distance <= max_dist, False otherwise
#[pyfunction]
#[pyo3(signature = (a, b, max_dist))]
fn check_bytes_within_dist(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let a_bytes: &[u8] = a.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let b_bytes: &[u8] = b.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if a_bytes.is_empty() || b_bytes.is_empty() {
        return Err(PyValueError::new_err("array size must be >0"));
    }
    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if a_bytes.len() != b_bytes.len() {
        return Err(PyValueError::new_err("array sizes need to be the same"));
    }

    let a_owned = a_bytes.to_vec();
    let b_owned = b_bytes.to_vec();
    let result = py.allow_threads(move || {
        hamming_distance_bytes_dispatch(&a_owned, &b_owned, max_dist_val)
    });
    Ok(result == 1)
}

/// Check if any element of byte array is within a specified Hamming Distance
/// and return its index or -1 otherwise.
/// (Legacy name, equivalent to check_bytes_arrays_first_within_dist)
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    check_bytes_arrays_first_within_dist(py, array_of_elems, elem_to_compare, max_dist)
}

/// Check if any element of byte array is within a specified Hamming Distance
/// and return the index of the first match, or -1 otherwise.
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_first_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let big_array: &[u8] = array_of_elems.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let small_array: &[u8] = elem_to_compare.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if small_array.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_array.len() % small_array.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let result = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            if hamming_distance_bytes_dispatch(chunk, &small_owned, max_dist_val) == 1 {
                return i as i64;
            }
        }
        -1i64
    });
    Ok(result)
}

/// Find the element in byte array with the smallest Hamming distance
/// Returns (best_distance, best_index), or (-1, -1) if none found within max_dist
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_best_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<(i64, i64)> {
    let big_array: &[u8] = array_of_elems.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let small_array: &[u8] = elem_to_compare.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if small_array.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_array.len() % small_array.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let result = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        let mut best_dist: i64 = -1;
        let mut best_index: i64 = -1;

        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            // Use current best as threshold for early termination, or max_dist if no match yet
            let threshold = if best_dist >= 0 { best_dist - 1 } else { max_dist_val };
            if hamming_distance_bytes_dispatch(chunk, &small_owned, threshold) == 0 {
                continue;
            }
            let dist = hamming_distance_bytes_dispatch(chunk, &small_owned, -1) as i64;
            if best_dist < 0 || dist < best_dist {
                best_dist = dist;
                best_index = i as i64;
            }
        }
        (best_dist, best_index)
    });
    Ok(result)
}

/// Find all elements in byte array within a specified Hamming distance
/// Returns list of (distance, index) tuples
#[pyfunction]
#[pyo3(signature = (array_of_elems, elem_to_compare, max_dist))]
fn check_bytes_arrays_all_within_dist(
    py: Python<'_>,
    array_of_elems: &Bound<'_, PyAny>,
    elem_to_compare: &Bound<'_, PyAny>,
    max_dist: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u64, u64)>> {
    let big_array: &[u8] = array_of_elems.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let small_array: &[u8] = elem_to_compare.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;
    let max_dist_val: i64 = max_dist.extract().map_err(|_| {
        PyValueError::new_err("error occurred while parsing arguments")
    })?;

    if small_array.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    if max_dist_val < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    if big_array.len() % small_array.len() != 0 {
        return Err(PyValueError::new_err(
            "`array_of_elems` size must be multiplier of `elem_to_compare`",
        ));
    }

    let big_owned = big_array.to_vec();
    let small_owned = small_array.to_vec();
    let results = py.allow_threads(move || {
        let elem_size = small_owned.len();
        let num_elements = big_owned.len() / elem_size;
        let mut out: Vec<(u64, u64)> = Vec::new();

        for i in 0..num_elements {
            let chunk = &big_owned[i * elem_size..(i + 1) * elem_size];
            if hamming_distance_bytes_dispatch(chunk, &small_owned, max_dist_val) == 0 {
                continue;
            }
            let dist = hamming_distance_bytes_dispatch(chunk, &small_owned, -1);
            out.push((dist, i as u64));
        }
        out
    });
    Ok(results)
}

/// Change algorithm used for calculations
/// Returns empty string if successful, or error message otherwise
#[pyfunction]
fn set_algo(algo_name: &str) -> PyResult<String> {
    match algo_name.to_lowercase().as_str() {
        "avx512" | "avx-512" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                    CURRENT_ALGO.store(ALGO_AVX512, Ordering::Relaxed);
                    return Ok(String::new());
                }
                return Ok("CPU doesn't support AVX-512 BITALG".to_string());
            }
            #[cfg(not(target_arch = "x86_64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

        "extra" | "avx" | "avx2" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    CURRENT_ALGO.store(ALGO_AVX2, Ordering::Relaxed);
                    return Ok(String::new());
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                return Ok(String::new());
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Ok("CPU doesn't support this feature".to_string());
            }
            #[cfg(target_arch = "x86_64")]
            Ok("CPU doesn't support this feature".to_string())
        }

        "native" | "popcount" => {
            CURRENT_ALGO.store(ALGO_NATIVE, Ordering::Relaxed);
            Ok(String::new())
        }

        "sse41" | "sse" => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.1") {
                    CURRENT_ALGO.store(ALGO_SSE41, Ordering::Relaxed);
                    return Ok(String::new());
                }
                Ok("CPU doesn't support this feature".to_string())
            }
            #[cfg(not(target_arch = "x86_64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

        "neon" => {
            #[cfg(target_arch = "aarch64")]
            {
                CURRENT_ALGO.store(ALGO_NEON, Ordering::Relaxed);
                Ok(String::new())
            }
            #[cfg(not(target_arch = "aarch64"))]
            Ok("Library was built without this algorithm.".to_string())
        }

        "classic" => {
            CURRENT_ALGO.store(ALGO_CLASSIC, Ordering::Relaxed);
            Ok(String::new())
        }

        _ => Ok("Library was built without this algorithm.".to_string()),
    }
}

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
