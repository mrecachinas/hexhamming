use crate::{
    bytes_array_all_within_dist, bytes_array_best_within_dist, bytes_array_first_within_dist,
    bytes_hamming_distance, bytes_within_dist, hex_hamming_distance,
};

#[test]
fn test_basic_hamming() {
    assert_eq!(hex_hamming_distance("deadbeef", "00000000").unwrap(), 24);
    assert_eq!(hex_hamming_distance("ffff", "0000").unwrap(), 16);
    assert_eq!(hex_hamming_distance("0000", "0000").unwrap(), 0);
    assert_eq!(hex_hamming_distance("f", "0").unwrap(), 4);
}

#[test]
fn test_mixed_case() {
    assert_eq!(hex_hamming_distance("DEADBEEF", "deadbeef").unwrap(), 0);
    assert_eq!(hex_hamming_distance("AbCdEf", "abcdef").unwrap(), 0);
    assert_eq!(hex_hamming_distance("aAbBcC", "AABBCC").unwrap(), 0);
}

#[test]
fn test_long_strings_32plus() {
    // 32 chars — exercises the SSE pack/32-char loop
    let a32 = "f".repeat(32);
    let b32 = "0".repeat(32);
    assert_eq!(hex_hamming_distance(&a32, &b32).unwrap(), 128);

    // 64 chars — exercises the AVX2 64-char loop
    let a64 = "f".repeat(64);
    let b64 = "0".repeat(64);
    assert_eq!(hex_hamming_distance(&a64, &b64).unwrap(), 256);

    // 128 chars — multiple AVX2 iterations
    let a128 = "f".repeat(128);
    let b128 = "0".repeat(128);
    assert_eq!(hex_hamming_distance(&a128, &b128).unwrap(), 512);

    // 254 chars — AVX2 loop + SSE tail + scalar tail
    let a254 = "f".repeat(254);
    let b254 = "0".repeat(254);
    assert_eq!(hex_hamming_distance(&a254, &b254).unwrap(), 1016);
}

#[test]
fn test_very_long_strings_no_overflow() {
    // Regression: the AVX-512 string path accumulated per-byte popcounts into
    // u8 lanes without flushing, overflowing for strings longer than ~4032
    // chars. Exercise lengths well past that boundary through the default
    // dispatch (AVX-512/AVX2 on x86, batched NEON on aarch64).
    for &n in &[4032usize, 4096, 5000, 8192, 10_000] {
        let a = "f".repeat(n);
        let b = "0".repeat(n);
        // Every hex char differs in all 4 bits → 4 * n.
        assert_eq!(
            hex_hamming_distance(&a, &b).unwrap(),
            4 * n as u64,
            "wrong distance for {n}-char all-f vs all-0"
        );
        // Half the distance with a partial pattern, length not a multiple of 64.
        let c = "f0".repeat(n / 2);
        let d = "00".repeat(n / 2);
        assert_eq!(
            hex_hamming_distance(&c, &d).unwrap(),
            4 * (n as u64 / 2),
            "wrong distance for {n}-char f0 vs 00"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_string_long_no_overflow() {
    use crate::set_algorithm;
    // Only meaningful where AVX-512 BITALG is available; otherwise set_algorithm
    // returns Err and we skip.
    if set_algorithm("avx512").is_err() {
        return;
    }
    for &n in &[4096usize, 8192, 12_000] {
        let a = "f".repeat(n);
        let b = "0".repeat(n);
        assert_eq!(hex_hamming_distance(&a, &b).unwrap(), 4 * n as u64);
    }
    set_algorithm("native").ok();
}

#[test]
fn test_long_mixed_content() {
    // Mixed hex chars to exercise all parse paths across SIMD lanes
    let a = "0123456789abcdef".repeat(8); // 128 chars
    let b = "fedcba9876543210".repeat(8);
    let result = hex_hamming_distance(&a, &b).unwrap();
    // Each pair: 0^f=f(4), 1^e=f(4), 2^d=f(4), 3^c=f(4),
    //            4^b=f(4), 5^a=f(4), 6^9=f(4), 7^8=f(4),
    //            8^7=f(4), 9^6=f(4), a^5=f(4), b^4=f(4),
    //            c^3=f(4), d^2=f(4), e^1=f(4), f^0=f(4) = 64 per 16 chars
    assert_eq!(result, 64 * 8);

    // Mixed case in long string
    let a_mixed = "AaBbCcDdEeFf0011".repeat(4); // 64 chars
    let b_mixed = "aAbBcCdDeEfF0011".repeat(4);
    assert_eq!(hex_hamming_distance(&a_mixed, &b_mixed).unwrap(), 0);
}

#[test]
fn test_invalid_chars() {
    assert!(hex_hamming_distance("zz", "00").is_err());
    assert!(hex_hamming_distance("gg", "00").is_err());
    assert!(hex_hamming_distance(
        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
        "00000000000000000000000000000000ff"
    )
    .is_err());
    assert!(hex_hamming_distance(
        "``````````````````````````````````",
        "00000000000000000000000000000000ff"
    )
    .is_err());
}

#[test]
fn test_length_mismatch() {
    assert!(hex_hamming_distance("ff", "f").is_err());
}

#[test]
fn test_empty() {
    assert_eq!(hex_hamming_distance("", "").unwrap(), 0);
}

#[test]
fn test_bytes_basic() {
    assert_eq!(bytes_hamming_distance(b"\xff", b"\x00").unwrap(), 8);
    assert_eq!(
        bytes_hamming_distance(b"\xde\xad\xbe\xef", b"\x00\x00\x00\x00").unwrap(),
        24
    );
}

// ---------------------------------------------------------------------------
// Wave-1 regression tests: dispatch contract, various lengths, within-dist API
// ---------------------------------------------------------------------------

/// Helper: compute expected byte Hamming distance between two byte slices
/// using a simple scalar method (for oracle comparison).
fn expected_byte_distance(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

#[test]
fn test_dispatch_returns_actual_distance_various_lengths() {
    // Exercise dispatch at many boundary lengths that tickle SIMD tails.
    let lengths: &[usize] = &[1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 256, 1024];
    for &len in lengths {
        let a = vec![0xFFu8; len];
        let b = vec![0x00u8; len];
        let expected = len as u64 * 8;
        let got = bytes_hamming_distance(&a, &b).unwrap();
        assert_eq!(
            got, expected,
            "bytes distance mismatch for len={}: got {} expected {}",
            len, got, expected
        );

        // With max_dist = -1 (unlimited) — should still return actual distance
        let got_unlimited = crate::hamming_distance_bytes_dispatch(&a, &b, -1);
        assert_eq!(
            got_unlimited, expected,
            "dispatch(-1) mismatch for len={}",
            len
        );

        // With max_dist = expected + 10 — within threshold, should return actual
        let got_within = crate::hamming_distance_bytes_dispatch(&a, &b, (expected + 10) as i64);
        assert_eq!(
            got_within, expected,
            "dispatch(within) mismatch for len={}",
            len
        );
    }
}

#[test]
fn test_dispatch_returns_sentinel_when_exceeded() {
    let lengths: &[usize] = &[1, 8, 16, 32, 64, 128, 256, 1024];
    for &len in lengths {
        let a = vec![0xFFu8; len];
        let b = vec![0x00u8; len];
        // max_dist = 0 should always be exceeded for non-identical inputs
        let got = crate::hamming_distance_bytes_dispatch(&a, &b, 0);
        assert_eq!(
            got,
            u64::MAX,
            "expected u64::MAX for len={} with max_dist=0, got {}",
            len,
            got
        );

        // max_dist = 1 should also be exceeded (actual distance = len*8)
        let got1 = crate::hamming_distance_bytes_dispatch(&a, &b, 1);
        assert_eq!(
            got1,
            u64::MAX,
            "expected u64::MAX for len={} with max_dist=1, got {}",
            len,
            got1
        );
    }
}

#[test]
fn test_dispatch_partial_diff_various_lengths() {
    // Only one byte differs → distance = popcount(0xFF) = 8
    let lengths: &[usize] = &[1, 8, 16, 32, 64, 128, 256, 1024];
    for &len in lengths {
        let a = vec![0x00u8; len];
        let mut b = vec![0x00u8; len];
        b[0] = 0xFF;
        let expected = 8u64;
        let got = bytes_hamming_distance(&a, &b).unwrap();
        assert_eq!(
            got, expected,
            "partial diff mismatch for len={}: got {} expected {}",
            len, got, expected
        );
    }
}

#[test]
fn test_dispatch_agrees_with_oracle() {
    // Pseudo-random bytes to exercise diverse bit patterns
    let lengths: &[usize] = &[7, 15, 31, 33, 63, 127, 255, 512];
    for &len in lengths {
        let a: Vec<u8> = (0..len).map(|i| (i * 37 + 13) as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i * 53 + 97) as u8).collect();
        let expected = expected_byte_distance(&a, &b);
        let got = bytes_hamming_distance(&a, &b).unwrap();
        assert_eq!(
            got, expected,
            "oracle mismatch for len={}: got {} expected {}",
            len, got, expected
        );
    }
}

#[test]
fn test_bytes_within_dist_api() {
    let a = b"\xde\xad\xbe\xef";
    let b = b"\x00\x00\x00\x00";
    // Actual distance = 24
    assert!(bytes_within_dist(a, b, 24).unwrap());
    assert!(bytes_within_dist(a, b, 30).unwrap());
    assert!(!bytes_within_dist(a, b, 2).unwrap());
    assert!(!bytes_within_dist(a, b, 0).unwrap());
}

#[test]
fn test_bytes_array_first_within_dist_api() {
    // 3 elements of 4 bytes each
    let big = [
        0x00u8, 0x00, 0x00, 0x00, // elem 0: all zeros
        0xDE, 0xAD, 0xBE, 0xEF, // elem 1: 24 bits from zero
        0xFF, 0xFF, 0xFF, 0xFF, // elem 2: 32 bits from zero
    ];
    let needle = [0x00u8, 0x00, 0x00, 0x00];
    // max_dist=5 → only elem 0 qualifies (dist=0)
    assert_eq!(
        bytes_array_first_within_dist(&big, &needle, 5).unwrap(),
        Some(0)
    );
    // max_dist=30 → elem 0 still first (dist=0 < 30)
    assert_eq!(
        bytes_array_first_within_dist(&big, &needle, 30).unwrap(),
        Some(0)
    );
    // Check with needle that only matches elem 2
    let needle2 = [0xFFu8, 0xFF, 0xFF, 0xFF];
    assert_eq!(
        bytes_array_first_within_dist(&big, &needle2, 5).unwrap(),
        Some(2)
    );
}

#[test]
fn test_bytes_array_best_within_dist_api() {
    // 3 elements: distances from needle=[0,0,0,0] are 0, 24, 32
    let big = [
        0xFFu8, 0xFF, 0xFF, 0xFF, // elem 0: dist 32
        0x01, 0x00, 0x00, 0x00, // elem 1: dist 1
        0x00, 0x00, 0x00, 0x00, // elem 2: dist 0
    ];
    let needle = [0x00u8, 0x00, 0x00, 0x00];
    let result = bytes_array_best_within_dist(&big, &needle, 40).unwrap();
    assert_eq!(result, Some((0, 2))); // elem 2 is best (dist=0)

    // With tight threshold: max_dist=2 → elem 1 (dist=1) and elem 2 (dist=0) qualify, best is elem 2
    let result2 = bytes_array_best_within_dist(&big, &needle, 2).unwrap();
    assert_eq!(result2, Some((0, 2)));

    // With max_dist=0 → only exact match (elem 2)
    let result3 = bytes_array_best_within_dist(&big, &needle, 0).unwrap();
    assert_eq!(result3, Some((0, 2)));
}

#[test]
fn test_bytes_array_all_within_dist_api() {
    let big = [
        0x00u8, 0x00, 0x00, 0x00, // elem 0: dist 0
        0x01, 0x00, 0x00, 0x00, // elem 1: dist 1
        0xFF, 0xFF, 0xFF, 0xFF, // elem 2: dist 32
    ];
    let needle = [0x00u8, 0x00, 0x00, 0x00];
    let results = bytes_array_all_within_dist(&big, &needle, 5).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], (0, 0)); // elem 0, dist 0
    assert_eq!(results[1], (1, 1)); // elem 1, dist 1

    // All within max_dist=40
    let results_all = bytes_array_all_within_dist(&big, &needle, 40).unwrap();
    assert_eq!(results_all.len(), 3);
    assert_eq!(results_all[2], (32, 2)); // elem 2, dist 32
}

#[test]
fn test_hex_string_various_lengths() {
    // Exercise hex string path at lengths that hit different SIMD tiers
    let lengths: &[usize] = &[1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 256, 1024];
    for &len in lengths {
        let a = "f".repeat(len);
        let b = "0".repeat(len);
        // Each hex char: f ^ 0 = 0xF → 4 bits
        let expected = len as u64 * 4;
        let got = hex_hamming_distance(&a, &b).unwrap();
        assert_eq!(
            got, expected,
            "hex distance mismatch for len={}: got {} expected {}",
            len, got, expected
        );
    }
}

#[test]
fn test_hex_string_mixed_pattern_various_lengths() {
    // Use a repeating pattern of "a5" so xor("a5", "00") = 0xA5 → popcount = 4
    let lengths: &[usize] = &[2, 16, 32, 64, 128, 254, 256, 512, 1024];
    for &len in lengths {
        // Repeat "a5" to fill length (len must be even for this pattern)
        let len_even = len & !1; // round down to even
        let a = "a5".repeat(len_even / 2);
        let b = "00".repeat(len_even / 2);
        // a^0 = 0xA = 1010 → 2 bits, 5^0 = 0x5 = 0101 → 2 bits → 4 bits per 2 chars
        let expected = (len_even / 2) as u64 * 4;
        let got = hex_hamming_distance(&a, &b).unwrap();
        assert_eq!(
            got, expected,
            "hex mixed pattern mismatch for len={}: got {} expected {}",
            len_even, got, expected
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for hamming_distance_string_dispatch_with_max
// ---------------------------------------------------------------------------

#[test]
fn test_string_dispatch_with_max_agrees_with_full() {
    // _with_max(max=u64::MAX) should return same result as full dispatch
    let lengths: &[usize] = &[64, 96, 128, 256, 1024];
    for &len in lengths {
        let a = "f".repeat(len);
        let b = "0".repeat(len);
        let full = hex_hamming_distance(&a, &b).unwrap();
        let with_max =
            crate::hamming_distance_string_dispatch_with_max(a.as_bytes(), b.as_bytes(), u64::MAX)
                .unwrap();
        assert_eq!(
            full, with_max,
            "dispatch_with_max mismatch at len={}: full={} with_max={}",
            len, full, with_max
        );
    }
}

#[test]
fn test_string_dispatch_with_max_returns_sentinel() {
    // Returns u64::MAX when actual distance > max_dist
    let lengths: &[usize] = &[64, 96, 128, 256, 1024];
    for &len in lengths {
        let a = "f".repeat(len);
        let b = "0".repeat(len);
        let result =
            crate::hamming_distance_string_dispatch_with_max(a.as_bytes(), b.as_bytes(), 1)
                .unwrap();
        assert_eq!(
            result,
            u64::MAX,
            "expected sentinel for len={}, got {}",
            len,
            result
        );
    }
}

#[test]
fn test_string_dispatch_with_max_returns_actual() {
    // Returns actual distance when <= max_dist
    let lengths: &[usize] = &[64, 96, 128, 256, 1024];
    for &len in lengths {
        let a = "f".repeat(len);
        let b = "0".repeat(len);
        let expected = len as u64 * 4;
        let result = crate::hamming_distance_string_dispatch_with_max(
            a.as_bytes(),
            b.as_bytes(),
            expected + 100,
        )
        .unwrap();
        assert_eq!(
            result, expected,
            "mismatch at len={}: expected {} got {}",
            len, expected, result
        );
    }
}

#[test]
fn test_string_dispatch_with_max_invalid_chars() {
    let a = "z".repeat(64);
    let b = "0".repeat(64);
    assert!(
        crate::hamming_distance_string_dispatch_with_max(a.as_bytes(), b.as_bytes(), 100).is_err()
    );
}

#[test]
fn test_string_dispatch_with_max_mixed_pattern() {
    // Mixed content with known distances
    let lengths: &[usize] = &[64, 128, 256, 1024];
    for &len in lengths {
        let a = "0123456789abcdef".repeat(len / 16);
        let b = "fedcba9876543210".repeat(len / 16);
        let full = hex_hamming_distance(&a, &b).unwrap();

        // Within threshold — returns actual
        let result =
            crate::hamming_distance_string_dispatch_with_max(a.as_bytes(), b.as_bytes(), full + 10)
                .unwrap();
        assert_eq!(result, full, "within-threshold mismatch at len={}", len);

        // Exactly at threshold
        let result_exact =
            crate::hamming_distance_string_dispatch_with_max(a.as_bytes(), b.as_bytes(), full)
                .unwrap();
        assert_eq!(
            result_exact, full,
            "exact-threshold mismatch at len={}",
            len
        );

        // Below threshold — returns sentinel
        if full > 0 {
            let result_below = crate::hamming_distance_string_dispatch_with_max(
                a.as_bytes(),
                b.as_bytes(),
                full - 1,
            )
            .unwrap();
            assert_eq!(
                result_below,
                u64::MAX,
                "below-threshold should be sentinel at len={}",
                len
            );
        }
    }
}

fn array_oracle(
    big: &[u8],
    small: &[u8],
    max_dist: i64,
) -> (Option<usize>, Option<(u64, usize)>, Vec<(u64, usize)>) {
    let width = small.len();
    let mut first = None;
    let mut best = None;
    let mut all = Vec::new();
    for (index, record) in big.chunks_exact(width).enumerate() {
        let distance = expected_byte_distance(record, small);
        if max_dist >= 0 && distance > max_dist as u64 {
            continue;
        }
        if first.is_none() {
            first = Some(index);
        }
        if best
            .map(|(best_distance, best_index)| {
                distance < best_distance || (distance == best_distance && index < best_index)
            })
            .unwrap_or(true)
        {
            best = Some((distance, index));
        }
        all.push((distance, index));
    }
    (first, best, all)
}

#[test]
fn test_fixed_width_array_scanners_match_randomized_oracle() {
    for algorithm in ["native", "classic"] {
        crate::set_algorithm(algorithm).unwrap();
        for &width in &[1usize, 3, 7, 15, 16, 17, 31, 32, 33] {
            let count = 37;
            let mut state = 0xA5A5_1234_5678_9ABCu64 ^ width as u64;
            let mut next_byte = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            };
            let small: Vec<u8> = (0..width).map(|_| next_byte()).collect();
            let mut big: Vec<u8> = (0..count * width).map(|_| next_byte()).collect();

            // Duplicate exact matches test lowest-index tie behavior and all
            // result ordering. A four-bit near match supplies d-1/d/d+1
            // threshold cases without relying on random distances.
            for &index in &[2usize, 19, 36] {
                big[index * width..(index + 1) * width].copy_from_slice(&small);
            }
            let near_index = 12;
            big[near_index * width..(near_index + 1) * width].copy_from_slice(&small);
            big[near_index * width] ^= 0b1111;

            for max_dist in [0, 3, 4, 5, 8, -1] {
                let expected = array_oracle(&big, &small, max_dist);
                assert_eq!(
                    bytes_array_first_within_dist(&big, &small, max_dist).unwrap(),
                    expected.0,
                    "first mismatch algorithm={algorithm} width={width} max_dist={max_dist}"
                );
                assert_eq!(
                    bytes_array_best_within_dist(&big, &small, max_dist).unwrap(),
                    expected.1,
                    "best mismatch algorithm={algorithm} width={width} max_dist={max_dist}"
                );
                assert_eq!(
                    bytes_array_all_within_dist(&big, &small, max_dist).unwrap(),
                    expected.2,
                    "all mismatch algorithm={algorithm} width={width} max_dist={max_dist}"
                );
            }
        }
    }
    crate::set_algorithm("native").unwrap();
}

#[test]
fn test_parallel_fixed_width_scanners_preserve_boundaries_and_order() {
    crate::set_algorithm("native").unwrap();
    let width = 16;
    let count = (16 * 1024 * 1024) / width + 7;
    let small = vec![0u8; width];
    let mut big = vec![0xFFu8; count * width];
    let quarter = count / 4;
    let exact_indices = [quarter - 1, quarter, quarter + 1, count - 1];
    for &index in &exact_indices {
        big[index * width..(index + 1) * width].copy_from_slice(&small);
    }

    assert_eq!(
        bytes_array_first_within_dist(&big, &small, 0).unwrap(),
        Some(exact_indices[0])
    );
    assert_eq!(
        bytes_array_best_within_dist(&big, &small, 0).unwrap(),
        Some((0, exact_indices[0]))
    );
    let all = bytes_array_all_within_dist(&big, &small, 0).unwrap();
    assert_eq!(
        all,
        exact_indices
            .into_iter()
            .map(|index| (0, index))
            .collect::<Vec<_>>()
    );
}
