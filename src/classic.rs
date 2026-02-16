use crate::hex::hex_char_to_nibble;
use crate::LOOKUP;

/// Classic popcount implementation using bit manipulation (Wilkes-Wheeler-Gill)
#[inline(always)]
pub(crate) fn popcnt64_classic(mut x: u64) -> u64 {
    const M1: u64 = 0x5555555555555555;
    const M2: u64 = 0x3333333333333333;
    const M4: u64 = 0x0F0F0F0F0F0F0F0F;
    const H01: u64 = 0x0101010101010101;
    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    (x.wrapping_mul(H01)) >> 56
}

/// Calculate hamming distance between two hex strings using classic algorithm
/// Optimized with branchless lookup and bounds check elimination
#[inline(always)]
pub(crate) fn hamming_distance_string_classic(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let len = a.len();
    let mut result: u64 = 0;
    let mut i = 0;
    
    // Process 4 hex chars at a time to reduce loop overhead
    while i + 4 <= len {
        // SAFETY: i + 3 < len verified by loop condition
        unsafe {
            let val1_0 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2_0 = hex_char_to_nibble(*b.get_unchecked(i));
            let val1_1 = hex_char_to_nibble(*a.get_unchecked(i + 1));
            let val2_1 = hex_char_to_nibble(*b.get_unchecked(i + 1));
            let val1_2 = hex_char_to_nibble(*a.get_unchecked(i + 2));
            let val2_2 = hex_char_to_nibble(*b.get_unchecked(i + 2));
            let val1_3 = hex_char_to_nibble(*a.get_unchecked(i + 3));
            let val2_3 = hex_char_to_nibble(*b.get_unchecked(i + 3));
            
            // Check all 8 values for validity (0xFF indicates invalid)
            // Use bitwise OR to combine checks - any 0xFF will result in high bit set
            let invalid = (val1_0 | val2_0 | val1_1 | val2_1 | val1_2 | val2_2 | val1_3 | val2_3) & 0xF0;
            if invalid != 0 {
                return Err("hex string contains invalid char");
            }
            
            result += *LOOKUP.get_unchecked((val1_0 ^ val2_0) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_1 ^ val2_1) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_2 ^ val2_2) as usize) as u64
                   + *LOOKUP.get_unchecked((val1_3 ^ val2_3) as usize) as u64;
        }
        i += 4;
    }
    
    // Handle remaining characters
    while i < len {
        // SAFETY: i < len verified by loop condition
        unsafe {
            let val1 = hex_char_to_nibble(*a.get_unchecked(i));
            let val2 = hex_char_to_nibble(*b.get_unchecked(i));
            if (val1 | val2) & 0xF0 != 0 {
                return Err("hex string contains invalid char");
            }
            result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        }
        i += 1;
    }
    
    Ok(result)
}

/// Calculate hamming distance between two byte arrays using classic algorithm
/// Optimized with loop unrolling and bounds check elimination
#[inline(always)]
pub(crate) fn hamming_distance_bytes_classic(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();

    if max_dist < 0 {
        // Full distance calculation - heavily optimized
        let mut difference: u64 = 0;
        let mut i = 0;
        
        // Process 32 bytes at a time (4 x 8-byte chunks)
        while i + 32 <= length {
            // SAFETY: i + 31 < length verified by loop condition
            unsafe {
                let a0 = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b0 = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                let a1 = u64::from_ne_bytes(*(a.as_ptr().add(i + 8) as *const [u8; 8]));
                let b1 = u64::from_ne_bytes(*(b.as_ptr().add(i + 8) as *const [u8; 8]));
                let a2 = u64::from_ne_bytes(*(a.as_ptr().add(i + 16) as *const [u8; 8]));
                let b2 = u64::from_ne_bytes(*(b.as_ptr().add(i + 16) as *const [u8; 8]));
                let a3 = u64::from_ne_bytes(*(a.as_ptr().add(i + 24) as *const [u8; 8]));
                let b3 = u64::from_ne_bytes(*(b.as_ptr().add(i + 24) as *const [u8; 8]));
                
                difference += popcnt64_classic(a0 ^ b0)
                           + popcnt64_classic(a1 ^ b1)
                           + popcnt64_classic(a2 ^ b2)
                           + popcnt64_classic(a3 ^ b3);
            }
            i += 32;
        }
        
        // Process remaining 8-byte chunks
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_classic(a_chunk ^ b_chunk);
            }
            i += 8;
        }
        
        // Process remaining bytes
        while i < length {
            unsafe {
                difference += popcnt64_classic((*a.get_unchecked(i) ^ *b.get_unchecked(i)) as u64);
            }
            i += 1;
        }
        difference
    } else {
        // Early termination if exceeds max_dist
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;
        let mut i = 0;
        
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_classic(a_chunk ^ b_chunk);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            unsafe {
                difference += popcnt64_classic((*a.get_unchecked(i) ^ *b.get_unchecked(i)) as u64);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}
