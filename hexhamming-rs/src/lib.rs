use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod hamming;
mod simd;

use hamming::{hamming_distance_string_impl, hamming_distance_bytes_impl, check_hexstrings_within_dist_impl, check_bytes_arrays_within_dist_impl};

/// Calculate the hamming distance of two hexadecimal strings
#[pyfunction]
fn hamming_distance_string(a: &str, b: &str) -> PyResult<u64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }
    
    match hamming_distance_string_impl(a, b) {
        Ok(dist) => Ok(dist),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Calculate the hamming distance of two byte arrays
#[pyfunction]
fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> PyResult<u64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("bytes are NOT the same length"));
    }
    
    Ok(hamming_distance_bytes_impl(a, b))
}

/// Check if the hex strings are within a specified Hamming Distance
#[pyfunction]
fn check_hexstrings_within_dist(a: &str, b: &str, max_dist: u64) -> PyResult<bool> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("strings are NOT the same length"));
    }
    
    match check_hexstrings_within_dist_impl(a, b, max_dist) {
        Ok(result) => Ok(result),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Check if any element of byte array are within a specified Hamming Distance
#[pyfunction]
fn check_bytes_arrays_within_dist(array_of_elems: &[u8], elem_to_compare: &[u8], max_dist: i64) -> PyResult<i32> {
    if elem_to_compare.is_empty() {
        return Err(PyValueError::new_err("`elem_to_compare` size must be >0"));
    }
    
    if max_dist < 0 {
        return Err(PyValueError::new_err("`max_dist` must be >=0"));
    }
    
    if array_of_elems.len() % elem_to_compare.len() != 0 {
        return Err(PyValueError::new_err("`array_of_elems` size must be multiplier of `elem_to_compare`"));
    }
    
    Ok(check_bytes_arrays_within_dist_impl(array_of_elems, elem_to_compare, max_dist as u64))
}

/// Set algorithm to use (for compatibility with C++ version)
#[pyfunction]
fn set_algo(algorithm: &str) -> PyResult<String> {
    // In the Rust version, we auto-select the best algorithm at runtime
    // This function exists for API compatibility but doesn't actually change anything
    match algorithm {
        "extra" | "native" | "sse41" | "classic" => Ok(String::new()),
        _ => Ok("Library was built without this algorithm.".to_string()),
    }
}

/// A Python module implemented in Rust for fast hexadecimal hamming distance calculation
#[pymodule]
fn hexhamming_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_distance_string, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(check_hexstrings_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(check_bytes_arrays_within_dist, m)?)?;
    m.add_function(wrap_pyfunction!(set_algo, m)?)?;
    m.add("__version__", "2.2.3")?;
    Ok(())
}