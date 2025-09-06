use crate::simd;

/// Lookup table for popcount of 4-bit values (nibbles)
/// LOOKUP[i] = number of 1 bits in i (for i in 0..16)
const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Fast hex character to nibble conversion - optimized and unsafe for performance
#[inline]
unsafe fn hex_char_to_nibble_unchecked(c: u8) -> u8 {
    // This mirrors the C++ implementation exactly:
    // val = (c > '9') ? (c &~ 0x20) - 55: (c - '0');
    if c > b'9' {
        (c & !0x20) - 55  // Convert A-F/a-f to 10-15, case insensitive
    } else {
        c - b'0'  // Convert 0-9 to 0-9
    }
}

/// Safe wrapper that validates the result
#[inline]
fn hex_char_to_nibble_fast(c: u8) -> Option<u8> {
    unsafe {
        let result = hex_char_to_nibble_unchecked(c);
        if result <= 15 {
            Some(result)
        } else {
            None
        }
    }
}

/// Simple and fast implementation matching C++ performance
fn hamming_distance_string_classic(a: &str, b: &str) -> Result<u64, &'static str> {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let len = a_bytes.len();
    let mut result = 0u64;
    
    // Simple loop like C++ version - sometimes simpler is faster
    for i in 0..len {
        // Do unsafe conversion first, then validate
        unsafe {
            let val1 = hex_char_to_nibble_unchecked(a_bytes[i]);
            let val2 = hex_char_to_nibble_unchecked(b_bytes[i]);
            
            // Check bounds (like C++ version)
            if val1 > 15 || val2 > 15 {
                return Err("hex string contains invalid char");
            }
            
            result += LOOKUP[(val1 ^ val2) as usize] as u64;
        }
    }
    
    Ok(result)
}

/// Classic popcount implementation for u64
#[allow(dead_code)]
fn popcount_classic(mut x: u64) -> u64 {
    // Brian Kernighan's algorithm optimized version
    const M1: u64 = 0x5555555555555555;
    const M2: u64 = 0x3333333333333333;
    const M4: u64 = 0x0F0F0F0F0F0F0F0F;
    const H01: u64 = 0x0101010101010101;
    
    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    (x * H01) >> 56
}

/// Classic implementation for byte arrays
#[allow(dead_code)]
fn hamming_distance_bytes_classic(a: &[u8], b: &[u8]) -> u64 {
    let mut result = 0u64;
    let len = a.len();
    let mut i = 0;
    
    // Process in 8-byte chunks when possible
    while i + 8 <= len {
        let a_chunk = u64::from_le_bytes([
            a[i], a[i+1], a[i+2], a[i+3],
            a[i+4], a[i+5], a[i+6], a[i+7]
        ]);
        let b_chunk = u64::from_le_bytes([
            b[i], b[i+1], b[i+2], b[i+3],
            b[i+4], b[i+5], b[i+6], b[i+7]
        ]);
        result += popcount_classic(a_chunk ^ b_chunk);
        i += 8;
    }
    
    // Process remaining bytes
    while i < len {
        result += popcount_classic((a[i] ^ b[i]) as u64);
        i += 1;
    }
    
    result
}

/// Native implementation using hardware popcount
#[cfg(target_arch = "x86_64")]
fn hamming_distance_bytes_native(a: &[u8], b: &[u8]) -> u64 {
    let mut result = 0u64;
    let len = a.len();
    let mut i = 0;
    
    // Process in 8-byte chunks when possible
    while i + 8 <= len {
        let a_chunk = u64::from_le_bytes([
            a[i], a[i+1], a[i+2], a[i+3],
            a[i+4], a[i+5], a[i+6], a[i+7]
        ]);
        let b_chunk = u64::from_le_bytes([
            b[i], b[i+1], b[i+2], b[i+3],
            b[i+4], b[i+5], b[i+6], b[i+7]
        ]);
        // Use hardware popcount when available
        result += (a_chunk ^ b_chunk).count_ones() as u64;
        i += 8;
    }
    
    // Process remaining bytes
    while i < len {
        result += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    
    result
}

/// Auto-selecting implementation that chooses the best algorithm
pub fn hamming_distance_string_impl(a: &str, b: &str) -> Result<u64, &'static str> {
    // Always use the optimized classic implementation 
    // The current SIMD implementation needs more work to be faster than this optimized version
    hamming_distance_string_classic(a, b)
}

/// Auto-selecting implementation for byte arrays
pub fn hamming_distance_bytes_impl(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // Try to use SIMD first for large arrays
        if a.len() >= 32 && simd::has_avx2() {
            return simd::hamming_distance_bytes_avx2(a, b);
        } else if a.len() >= 16 && simd::has_sse41() {
            return simd::hamming_distance_bytes_sse41(a, b);
        } else {
            return hamming_distance_bytes_native(a, b);
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        hamming_distance_bytes_classic(a, b)
    }
}

/// Check if hex strings are within distance (early termination)
pub fn check_hexstrings_within_dist_impl(a: &str, b: &str, max_dist: u64) -> Result<bool, &'static str> {
    if a == b {
        return Ok(true);
    }
    
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut result = 0u64;
    
    for i in 0..a_bytes.len() {
        let val1 = hex_char_to_nibble_fast(a_bytes[i])
            .ok_or("hex string contains invalid char")?;
        let val2 = hex_char_to_nibble_fast(b_bytes[i])
            .ok_or("hex string contains invalid char")?;
        
        result += LOOKUP[(val1 ^ val2) as usize] as u64;
        
        // Early termination if we exceed max_dist
        if result > max_dist {
            return Ok(false);
        }
    }
    
    Ok(true)
}

/// Check if any element in byte array is within distance
pub fn check_bytes_arrays_within_dist_impl(array_of_elems: &[u8], elem_to_compare: &[u8], max_dist: u64) -> i32 {
    let elem_size = elem_to_compare.len();
    let num_elements = array_of_elems.len() / elem_size;
    
    for i in 0..num_elements {
        let start_idx = i * elem_size;
        let end_idx = start_idx + elem_size;
        let elem = &array_of_elems[start_idx..end_idx];
        
        let distance = hamming_distance_bytes_impl(elem, elem_to_compare);
        if distance <= max_dist {
            return i as i32;
        }
    }
    
    -1
}