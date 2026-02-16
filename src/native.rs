/// Native popcount implementation using CPU instruction
#[inline(always)]
pub(crate) fn popcnt64_native(x: u64) -> u64 {
    x.count_ones() as u64
}

/// Calculate hamming distance between two byte arrays using native popcount
/// Optimized with aggressive loop unrolling and bounds check elimination
#[inline(always)]
pub(crate) fn hamming_distance_bytes_native(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();

    if max_dist < 0 {
        let mut difference: u64 = 0;
        let mut i = 0;
        
        // Process 32 bytes at a time (4 x 8-byte chunks) to saturate execution units
        while i + 32 <= length {
            unsafe {
                let a0 = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b0 = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                let a1 = u64::from_ne_bytes(*(a.as_ptr().add(i + 8) as *const [u8; 8]));
                let b1 = u64::from_ne_bytes(*(b.as_ptr().add(i + 8) as *const [u8; 8]));
                let a2 = u64::from_ne_bytes(*(a.as_ptr().add(i + 16) as *const [u8; 8]));
                let b2 = u64::from_ne_bytes(*(b.as_ptr().add(i + 16) as *const [u8; 8]));
                let a3 = u64::from_ne_bytes(*(a.as_ptr().add(i + 24) as *const [u8; 8]));
                let b3 = u64::from_ne_bytes(*(b.as_ptr().add(i + 24) as *const [u8; 8]));
                
                difference += popcnt64_native(a0 ^ b0)
                           + popcnt64_native(a1 ^ b1)
                           + popcnt64_native(a2 ^ b2)
                           + popcnt64_native(a3 ^ b3);
            }
            i += 32;
        }
        
        // Process remaining 8-byte chunks
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_native(a_chunk ^ b_chunk);
            }
            i += 8;
        }
        
        // Process remaining bytes
        while i < length {
            unsafe {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            }
            i += 1;
        }
        difference
    } else {
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;
        let mut i = 0;
        
        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_native(a_chunk ^ b_chunk);
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 8;
        }
        while i < length {
            unsafe {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            }
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}
