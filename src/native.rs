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

        // Process 32 bytes at a time — check threshold once per batch
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
            if difference > max_dist_u64 {
                return u64::MAX;
            }
            i += 32;
        }

        while i + 8 <= length {
            unsafe {
                let a_chunk = u64::from_ne_bytes(*(a.as_ptr().add(i) as *const [u8; 8]));
                let b_chunk = u64::from_ne_bytes(*(b.as_ptr().add(i) as *const [u8; 8]));
                difference += popcnt64_native(a_chunk ^ b_chunk);
            }
            i += 8;
        }
        while i < length {
            unsafe {
                difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            }
            i += 1;
        }
        if difference > max_dist_u64 {
            u64::MAX
        } else {
            difference
        }
    }
}

#[inline(always)]
pub(crate) fn hamming_distance_bytes_native_16(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    debug_assert_eq!(a.len(), 16);
    debug_assert_eq!(b.len(), 16);

    let difference = unsafe {
        let a0 = u64::from_ne_bytes(*(a.as_ptr() as *const [u8; 8]));
        let b0 = u64::from_ne_bytes(*(b.as_ptr() as *const [u8; 8]));
        let a1 = u64::from_ne_bytes(*(a.as_ptr().add(8) as *const [u8; 8]));
        let b1 = u64::from_ne_bytes(*(b.as_ptr().add(8) as *const [u8; 8]));
        popcnt64_native(a0 ^ b0) + popcnt64_native(a1 ^ b1)
    };
    let max_dist_u64 = max_dist as u64;
    if max_dist >= 0 && difference > max_dist_u64 {
        u64::MAX
    } else {
        difference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcnt64_matches_count_ones() {
        // Verify native matches std count_ones for various values
        for x in [
            0u64,
            1,
            0xFF,
            0xDEADBEEF,
            0xFFFFFFFFFFFFFFFF,
            0x123456789ABCDEF0,
        ] {
            assert_eq!(popcnt64_native(x), x.count_ones() as u64);
        }
    }

    #[test]
    fn bytes_native_full_distance() {
        assert_eq!(hamming_distance_bytes_native(b"\xff", b"\x00", -1), 8);
        assert_eq!(
            hamming_distance_bytes_native(b"\x00\x00", b"\x00\x00", -1),
            0
        );
        // 64 bytes to exercise 32-byte unrolled loop
        let a = vec![0xFFu8; 64];
        let b = vec![0x00u8; 64];
        assert_eq!(hamming_distance_bytes_native(&a, &b, -1), 512);
    }

    #[test]
    fn bytes_native_with_max_dist() {
        // Within threshold → returns actual distance
        assert_eq!(hamming_distance_bytes_native(b"\xff", b"\xfe", 2), 1);
        // Exceeds threshold → returns u64::MAX
        assert_eq!(hamming_distance_bytes_native(b"\xff", b"\x00", 2), u64::MAX);
    }

    #[test]
    fn bytes_native_agrees_with_classic() {
        use crate::classic::hamming_distance_bytes_classic;
        // Test various sizes to cover all loop paths
        for size in [1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127] {
            let a: Vec<u8> = (0..size).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_add(1)).collect();
            assert_eq!(
                hamming_distance_bytes_native(&a, &b, -1),
                hamming_distance_bytes_classic(&a, &b, -1),
                "mismatch at size {size}"
            );
        }
    }

    #[test]
    fn bytes_native_16_agrees_with_generic() {
        let a: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..16).map(|i| (i as u8).wrapping_mul(17)).collect();
        for max_dist in [-1, 0, 1, 63, 64] {
            assert_eq!(
                hamming_distance_bytes_native_16(&a, &b, max_dist),
                hamming_distance_bytes_native(&a, &b, max_dist),
                "mismatch at max_dist {max_dist}"
            );
        }
    }
}
