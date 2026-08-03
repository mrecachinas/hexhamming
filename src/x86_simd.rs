// x86_64 SIMD implementations
// Note: Unlike ARM64, x86 SSE/AVX2 implementations use VPSHUFB-based popcount which
// processes 16/32 bytes in parallel with a lookup table. This is typically faster than
// scalar count_ones() because it avoids horizontal reduction overhead. The ARM64 native
// approach works better there because the CNT instruction handles accumulation efficiently.

use crate::classic::hamming_distance_string_classic;
use crate::hex::hex_char_to_nibble;
use crate::native::hamming_distance_bytes_native;
use crate::{LOOKUP, SCALAR_THRESHOLD};

#[allow(unused_imports)]
use std::arch::x86_64::*;

/// SSE4.1 popcount for 128-bit value using hardware popcnt
#[target_feature(enable = "sse4.1", enable = "popcnt")]
pub unsafe fn popcnt128_sse(n: __m128i) -> u64 {
    let lo = _mm_cvtsi128_si64(n) as u64;
    let hi = _mm_extract_epi64(n, 1) as u64;
    lo.count_ones() as u64 + hi.count_ones() as u64
}

/// SSE4.1 VPSHUFB-based popcount (accumulates into byte lanes)
/// Returns vector with per-byte popcounts - use _mm_sad_epu8 to sum
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn popcnt128_shuffle(v: __m128i, mask: __m128i, table: __m128i) -> __m128i {
    let lo = _mm_and_si128(v, mask);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
    _mm_add_epi8(_mm_shuffle_epi8(table, lo), _mm_shuffle_epi8(table, hi))
}

/// SSE4.1 implementation for byte arrays - heavily optimized
#[target_feature(enable = "sse4.1", enable = "popcnt")]
pub unsafe fn hamming_distance_bytes_sse(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut i = 0;

    // For small inputs, use scalar (SIMD setup overhead not worth it)
    if length < SCALAR_THRESHOLD {
        return hamming_distance_bytes_native(a, b, max_dist);
    }

    // VPSHUFB lookup table for 4-bit popcount
    let mask = _mm_set1_epi8(0x0F);
    let table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

    if max_dist < 0 {
        let mut total = _mm_setzero_si128();

        // Process 256 bytes at a time (16 x 16 bytes) before horizontal sum
        // This maximizes throughput by keeping counts in u8 lanes (max 255 per lane)
        // 16 iterations x 16 bytes x max 8 bits = max 2048 bits, but per-lane max is 16*8=128 < 255
        while i + 256 <= length {
            let mut acc = _mm_setzero_si128();
            for _ in 0..16 {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let xor = _mm_xor_si128(a16, b16);
                acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                i += 16;
            }
            total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));
        }

        // Process 64 bytes at a time (4 x 16 bytes)
        while i + 64 <= length {
            let mut acc = _mm_setzero_si128();
            for _ in 0..4 {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let xor = _mm_xor_si128(a16, b16);
                acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                i += 16;
            }
            total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));
        }

        // Process remaining 16-byte chunks
        let mut acc = _mm_setzero_si128();
        while i + 16 <= length {
            let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let xor = _mm_xor_si128(a16, b16);
            acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
            i += 16;
        }
        total = _mm_add_epi64(total, _mm_sad_epu8(acc, _mm_setzero_si128()));

        // Extract final sum
        let mut difference = (_mm_extract_epi64(total, 0) + _mm_extract_epi64(total, 1)) as u64;

        // Process remaining bytes with native popcnt
        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            i += 1;
        }
        difference
    } else {
        // Early termination path — batched VPSHUFB accumulator approach
        // Accumulate into u8 lanes for 16 iterations (256 B), then one SAD + check
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        while i + 256 <= length {
            let mut acc = _mm_setzero_si128();
            for _ in 0..16 {
                let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let xor = _mm_xor_si128(a16, b16);
                acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
                i += 16;
            }
            let sad = _mm_sad_epu8(acc, _mm_setzero_si128());
            difference += (_mm_extract_epi64(sad, 0) + _mm_extract_epi64(sad, 1)) as u64;
            if difference > max_dist_u64 {
                return u64::MAX;
            }
        }

        // Remaining 16-byte chunks
        let mut acc = _mm_setzero_si128();
        let mut acc_count = 0u32;
        while i + 16 <= length {
            let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let xor = _mm_xor_si128(a16, b16);
            acc = _mm_add_epi8(acc, popcnt128_shuffle(xor, mask, table));
            acc_count += 1;
            i += 16;
        }
        if acc_count > 0 {
            let sad = _mm_sad_epu8(acc, _mm_setzero_si128());
            difference += (_mm_extract_epi64(sad, 0) + _mm_extract_epi64(sad, 1)) as u64;
        }

        // Scalar tail
        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            i += 1;
        }
        if difference > max_dist_u64 {
            u64::MAX
        } else {
            difference
        }
    }
}

/// AVX2 VPSHUFB-based popcount - accumulates into byte lanes
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn popcnt256_shuffle(v: __m256i, mask: __m256i, table: __m256i) -> __m256i {
    let lo = _mm256_and_si256(v, mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
    _mm256_add_epi8(
        _mm256_shuffle_epi8(table, lo),
        _mm256_shuffle_epi8(table, hi),
    )
}

/// AVX2 implementation for byte arrays - heavily optimized with batched horizontal sums
#[target_feature(enable = "avx2")]
pub unsafe fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut i = 0;

    // For small inputs, fall back to SSE or scalar
    if length < 64 {
        if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("popcnt") {
            return hamming_distance_bytes_sse(a, b, max_dist);
        }
        return hamming_distance_bytes_native(a, b, max_dist);
    }

    // VPSHUFB lookup table for 4-bit popcount
    let mask = _mm256_set1_epi8(0x0F);
    let table = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );

    if max_dist < 0 {
        let mut total = _mm256_setzero_si256();

        // Process 512 bytes at a time (16 x 32 bytes) before horizontal sum
        // Per-lane accumulation: 16 iters x 8 bits max = 128 < 255, safe for u8
        while i + 512 <= length {
            let mut acc = _mm256_setzero_si256();
            for _ in 0..16 {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                i += 32;
            }
            total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
        }

        // Process 128 bytes at a time (4 x 32 bytes)
        while i + 128 <= length {
            let mut acc = _mm256_setzero_si256();
            for _ in 0..4 {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                i += 32;
            }
            total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));
        }

        // Process remaining 32-byte chunks
        let mut acc = _mm256_setzero_si256();
        while i + 32 <= length {
            let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let xor = _mm256_xor_si256(a32, b32);
            acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
            i += 32;
        }
        total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, _mm256_setzero_si256()));

        // Extract final sum from 4 u64 lanes
        let mut difference = (_mm256_extract_epi64(total, 0)
            + _mm256_extract_epi64(total, 1)
            + _mm256_extract_epi64(total, 2)
            + _mm256_extract_epi64(total, 3)) as u64;

        // Process remaining bytes
        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            i += 1;
        }
        difference
    } else {
        // Early termination path — batched 16×32 B (512 B) per SAD+check
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        while i + 512 <= length {
            let mut acc = _mm256_setzero_si256();
            for _ in 0..16 {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                i += 32;
            }
            let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
            // Efficient horizontal sum: extract 128-bit halves, add, then reduce
            let lo128 = _mm256_castsi256_si128(sad);
            let hi128 = _mm256_extracti128_si256(sad, 1);
            let sum128 = _mm_add_epi64(lo128, hi128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            difference += _mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)) as u64;
            if difference > max_dist_u64 {
                return u64::MAX;
            }
        }

        // Process remaining 128-byte batches
        while i + 128 <= length {
            let mut acc = _mm256_setzero_si256();
            for _ in 0..4 {
                let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                let xor = _mm256_xor_si256(a32, b32);
                acc = _mm256_add_epi8(acc, popcnt256_shuffle(xor, mask, table));
                i += 32;
            }
            let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
            let lo128 = _mm256_castsi256_si128(sad);
            let hi128 = _mm256_extracti128_si256(sad, 1);
            let sum128 = _mm_add_epi64(lo128, hi128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            difference += _mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)) as u64;
            if difference > max_dist_u64 {
                return u64::MAX;
            }
        }

        // Process remaining 32-byte chunks
        while i + 32 <= length {
            let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let xor = _mm256_xor_si256(a32, b32);
            let cnt = popcnt256_shuffle(xor, mask, table);
            let sad = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
            let lo128 = _mm256_castsi256_si128(sad);
            let hi128 = _mm256_extracti128_si256(sad, 1);
            let sum128 = _mm_add_epi64(lo128, hi128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            difference += _mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)) as u64;
            i += 32;
        }

        // Scalar tail
        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            i += 1;
        }
        if difference > max_dist_u64 {
            u64::MAX
        } else {
            difference
        }
    }
}

/// Branchless vectorized hex ASCII → nibble conversion for AVX2.
/// Same subtract-and-correct strategy as SSE/NEON, but on 32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hex_parse_avx2(
    chars: __m256i,
    case_mask: __m256i,
    ascii_0: __m256i,
    seven: __m256i,
    nine: __m256i,
    ten: __m256i,
) -> __m256i {
    let digit_val = _mm256_sub_epi8(chars, ascii_0);
    let letter_val = _mm256_sub_epi8(_mm256_and_si256(chars, case_mask), ascii_0);
    let is_letter = _mm256_cmpgt_epi8(digit_val, nine);
    let adjusted = _mm256_sub_epi8(letter_val, seven);
    let result = _mm256_blendv_epi8(digit_val, adjusted, is_letter);
    let bad_letter = _mm256_and_si256(is_letter, _mm256_cmpgt_epi8(ten, adjusted));
    _mm256_or_si256(result, bad_letter)
}

/// AVX2 nibble VPSHUFB-LUT implementation for hex strings.
/// Parses 64 hex chars (2×32) → nibbles, XORs, uses VPSHUFB LUT for
/// nibble-level popcount (0-4 per lane), then batched SAD accumulation.
#[target_feature(enable = "avx2", enable = "popcnt")]
pub unsafe fn hamming_distance_string_avx2(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    // Fall back to SSE for inputs < 64 chars
    if length < 64 {
        return hamming_distance_string_sse(a, b);
    }

    let zero = _mm256_setzero_si256();
    let fifteen = _mm256_set1_epi8(15);
    let case_mask = _mm256_set1_epi8(!0x20i8); // 0xDF
    let ascii_0 = _mm256_set1_epi8(b'0' as i8);
    let seven = _mm256_set1_epi8(7);
    let nine = _mm256_set1_epi8(9);
    let ten = _mm256_set1_epi8(10);

    // Nibble popcount LUT: popcnt[i] = number of 1-bits in i, for i in 0..15
    let popcnt_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );

    let mut i = 0;
    let mut total = _mm256_setzero_si256();

    // Process 64 hex chars × 4 iterations (256 chars) per batched SAD+accumulate.
    // Each nibble XOR produces max 4 set bits, and we accumulate popcount values
    // 0-4 per byte. After 8 iterations (8 loads of 32 nibbles per accumulator add),
    // max per-lane is 8*4 = 32 < 255. We batch 4 iterations of the 64-char loop
    // = 8 accumulator additions per u8 lane = max 32 per lane.
    while i + 256 <= length {
        let mut acc = _mm256_setzero_si256();
        for _ in 0..4 {
            let a_lo = hex_parse_avx2(
                _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );
            let b_lo = hex_parse_avx2(
                _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );
            let a_hi = hex_parse_avx2(
                _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );
            let b_hi = hex_parse_avx2(
                _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );

            // §8: consolidated validation — cmpgt(or(a,b), 15) plus negative check
            let or_lo = _mm256_or_si256(a_lo, b_lo);
            let or_hi = _mm256_or_si256(a_hi, b_hi);
            let invalid = _mm256_or_si256(
                _mm256_cmpgt_epi8(or_lo, fifteen),
                _mm256_cmpgt_epi8(or_hi, fifteen),
            );
            let negative = _mm256_or_si256(
                _mm256_cmpgt_epi8(zero, or_lo),
                _mm256_cmpgt_epi8(zero, or_hi),
            );
            let bad = _mm256_or_si256(invalid, negative);
            if _mm256_testz_si256(bad, bad) == 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles → VPSHUFB nibble-popcount LUT (values 0-15 → 0-4)
            let xor_lo = _mm256_xor_si256(a_lo, b_lo);
            let xor_hi = _mm256_xor_si256(a_hi, b_hi);
            let cnt_lo = _mm256_shuffle_epi8(popcnt_lut, xor_lo);
            let cnt_hi = _mm256_shuffle_epi8(popcnt_lut, xor_hi);
            acc = _mm256_add_epi8(acc, _mm256_add_epi8(cnt_lo, cnt_hi));

            i += 64;
        }
        total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, zero));
    }

    // Process remaining 64-char iterations individually
    let mut acc = _mm256_setzero_si256();
    while i + 64 <= length {
        let a_lo = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_lo = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let a_hi = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hi = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        let or_lo = _mm256_or_si256(a_lo, b_lo);
        let or_hi = _mm256_or_si256(a_hi, b_hi);
        let invalid = _mm256_or_si256(
            _mm256_cmpgt_epi8(or_lo, fifteen),
            _mm256_cmpgt_epi8(or_hi, fifteen),
        );
        let negative = _mm256_or_si256(
            _mm256_cmpgt_epi8(zero, or_lo),
            _mm256_cmpgt_epi8(zero, or_hi),
        );
        let bad = _mm256_or_si256(invalid, negative);
        if _mm256_testz_si256(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor_lo = _mm256_xor_si256(a_lo, b_lo);
        let xor_hi = _mm256_xor_si256(a_hi, b_hi);
        let cnt_lo = _mm256_shuffle_epi8(popcnt_lut, xor_lo);
        let cnt_hi = _mm256_shuffle_epi8(popcnt_lut, xor_hi);
        acc = _mm256_add_epi8(acc, _mm256_add_epi8(cnt_lo, cnt_hi));

        i += 64;
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, zero));

    // Extract final sum
    let mut difference = (_mm256_extract_epi64(total, 0)
        + _mm256_extract_epi64(total, 1)
        + _mm256_extract_epi64(total, 2)
        + _mm256_extract_epi64(total, 3)) as u64;

    // Fall through to SSE for remaining < 64 chars
    if i < length {
        let remaining = hamming_distance_string_sse(&a[i..], &b[i..])?;
        difference += remaining;
    }

    Ok(difference)
}

/// Branchless vectorized hex ASCII → nibble conversion for AVX-512BW.
/// 64 lanes, same subtract-and-correct strategy.
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn hex_parse_avx512(
    chars: __m512i,
    case_mask: __m512i,
    ascii_0: __m512i,
    seven: __m512i,
    nine: __m512i,
    ten: __m512i,
) -> __m512i {
    let digit_val = _mm512_sub_epi8(chars, ascii_0);
    let letter_val = _mm512_sub_epi8(_mm512_and_si512(chars, case_mask), ascii_0);
    let is_letter = _mm512_cmpgt_epi8_mask(digit_val, nine);
    let adjusted = _mm512_sub_epi8(letter_val, seven);
    let result = _mm512_mask_blend_epi8(is_letter, digit_val, adjusted);
    // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
    let bad_letter = is_letter & _mm512_cmpgt_epi8_mask(ten, adjusted);
    let ones = _mm512_set1_epi8(-1); // 0xFF
    _mm512_mask_blend_epi8(bad_letter, result, ones)
}

/// AVX-512 BITALG implementation for hex strings.
/// Parses 64 hex chars per load, XORs nibbles directly, uses VPOPCNTB
/// for native per-byte popcount — no pack step needed.
#[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
pub unsafe fn hamming_distance_string_avx512(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    // With masked loads, we can handle any length efficiently
    if length < 16 {
        return hamming_distance_string_classic(a, b);
    }

    let fifteen = _mm512_set1_epi8(15);
    let case_mask = _mm512_set1_epi8(!0x20i8); // 0xDF
    let ascii_0 = _mm512_set1_epi8(b'0' as i8);
    let seven = _mm512_set1_epi8(7);
    let nine = _mm512_set1_epi8(9);
    let ten = _mm512_set1_epi8(10);
    let zero = _mm512_setzero_si512();

    let mut i = 0;
    // Wide (epi64) running total. The per-byte popcount accumulator `acc` is
    // flushed into this via SAD before any lane can overflow.
    let mut total = _mm512_setzero_si512();

    // Each 64-char iteration adds at most 4 set bits per byte lane to `acc`.
    // A u8 lane overflows after 64 iterations (64*4 = 256), so flush `acc` into
    // the wide `total` at least every 63 iterations. We use BATCH=32 for a
    // comfortable margin. (The previous code accumulated into `total` as epi8
    // with no flush, silently overflowing for strings longer than ~4032 chars.)
    const BATCH: usize = 32;
    while i + 64 <= length {
        let mut acc = zero;
        let mut n = 0;
        while n < BATCH && i + 64 <= length {
            let a_nib = hex_parse_avx512(
                _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );
            let b_nib = hex_parse_avx512(
                _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
            );

            // §8: consolidated validation — cmpgt(or(a,b), 15) plus negative check
            let or_nib = _mm512_or_si512(a_nib, b_nib);
            let invalid =
                _mm512_cmpgt_epi8_mask(or_nib, fifteen) | _mm512_cmpgt_epi8_mask(zero, or_nib);
            if invalid != 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles and VPOPCNTB — counts set bits per byte
            let xor = _mm512_xor_si512(a_nib, b_nib);
            acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));

            i += 64;
            n += 1;
        }
        // Flush the byte accumulator into the wide total.
        total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));
    }

    let mut difference = _mm512_reduce_add_epi64(total) as u64;

    // Handle remaining chars with masked AVX-512 load (no fallthrough)
    let remaining = length - i;
    if remaining > 0 {
        let mask = if remaining >= 64 {
            !0u64
        } else {
            (1u64 << remaining) - 1
        };
        let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
        let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);

        let a_nib = hex_parse_avx512(a_tail, case_mask, ascii_0, seven, nine, ten);
        let b_nib = hex_parse_avx512(b_tail, case_mask, ascii_0, seven, nine, ten);

        // §8: consolidated validation — only active lanes
        let or_nib = _mm512_or_si512(a_nib, b_nib);
        let invalid =
            (_mm512_cmpgt_epi8_mask(or_nib, fifteen) | _mm512_cmpgt_epi8_mask(zero, or_nib)) & mask;
        if invalid != 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm512_xor_si512(a_nib, b_nib);
        let cnt = _mm512_popcnt_epi8(xor);
        let sad = _mm512_sad_epu8(cnt, zero);
        difference += _mm512_reduce_add_epi64(sad) as u64;
    }

    Ok(difference)
}

/// AVX-512 BITALG implementation for byte arrays.
/// XOR + VPOPCNTB for native per-byte popcount.
#[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
pub unsafe fn hamming_distance_bytes_avx512(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut i = 0;

    let zero = _mm512_setzero_si512();

    if max_dist < 0 {
        let mut total = _mm512_setzero_si512();

        // Process 1024 bytes at a time (16 × 64) before horizontal sum
        // Per-lane max: 16 × 8 = 128 < 255, safe for u8
        while i + 1024 <= length {
            let mut acc = _mm512_setzero_si512();
            for _ in 0..16 {
                let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
                let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
                let xor = _mm512_xor_si512(a64, b64);
                acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
                i += 64;
            }
            total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));
        }

        // Process remaining 64-byte chunks
        let mut acc = _mm512_setzero_si512();
        while i + 64 <= length {
            let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let xor = _mm512_xor_si512(a64, b64);
            acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
            i += 64;
        }
        total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));

        let mut difference = _mm512_reduce_add_epi64(total) as u64;

        // Masked tail — no scalar fallback
        let remaining = length - i;
        if remaining > 0 {
            let mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
            let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
            let xor = _mm512_xor_si512(a_tail, b_tail);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
        }
        difference
    } else {
        // Early termination path — accumulate 16 iters (1024 B) before SAD + check
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        while i + 1024 <= length {
            let mut acc = _mm512_setzero_si512();
            for _ in 0..16 {
                let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
                let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
                let xor = _mm512_xor_si512(a64, b64);
                acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
                i += 64;
            }
            let sad = _mm512_sad_epu8(acc, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
            if difference > max_dist_u64 {
                return u64::MAX;
            }
        }

        // Remaining 64-byte chunks
        let mut acc = _mm512_setzero_si512();
        while i + 64 <= length {
            let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let xor = _mm512_xor_si512(a64, b64);
            acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(xor));
            i += 64;
        }
        let sad = _mm512_sad_epu8(acc, zero);
        difference += _mm512_reduce_add_epi64(sad) as u64;

        // Masked tail for early termination path
        let remaining = length - i;
        if remaining > 0 {
            let mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
            let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
            let xor = _mm512_xor_si512(a_tail, b_tail);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
        }
        if difference > max_dist_u64 {
            u64::MAX
        } else {
            difference
        }
    }
}

/// Branchless vectorized hex ASCII → nibble conversion for SSE4.1.
/// Same subtract-and-correct strategy as the NEON version:
///   1. digit_val = c - '0': digits → 0-9
///   2. letter_val = (c & 0xDF) - '0' - 7: letters → 10-15
///   3. Select letter path where digit_val > 9
///   4. Force invalid where letter result < 10 (catches '@', '`')
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn hex_parse_sse(
    chars: __m128i,
    case_mask: __m128i,
    ascii_0: __m128i,
    seven: __m128i,
    nine: __m128i,
    ten: __m128i,
) -> __m128i {
    let digit_val = _mm_sub_epi8(chars, ascii_0);
    let letter_val = _mm_sub_epi8(_mm_and_si128(chars, case_mask), ascii_0);
    let is_letter = _mm_cmpgt_epi8(digit_val, nine);
    let adjusted = _mm_sub_epi8(letter_val, seven);
    let result = _mm_blendv_epi8(digit_val, adjusted, is_letter);
    // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
    let bad_letter = _mm_and_si128(is_letter, _mm_cmplt_epi8(adjusted, ten));
    _mm_or_si128(result, bad_letter)
}

/// SSE4.1 pack-to-bytes implementation for hex strings.
/// Parses 32 hex chars (2×16) → nibbles, XORs, packs pairs into bytes,
/// then uses hardware popcnt on u64 extracts.
#[target_feature(enable = "sse4.1", enable = "popcnt")]
pub unsafe fn hamming_distance_string_sse(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 32 {
        return hamming_distance_string_classic(a, b);
    }

    let zero = _mm_setzero_si128();
    let fifteen = _mm_set1_epi8(15);
    let case_mask = _mm_set1_epi8(!0x20i8); // 0xDF
    let ascii_0 = _mm_set1_epi8(b'0' as i8);
    let seven = _mm_set1_epi8(7);
    let nine = _mm_set1_epi8(9);
    let ten = _mm_set1_epi8(10);

    let mut i = 0;
    let mut difference: u64 = 0;

    // Process 32 hex chars at a time: parse→XOR→pack→popcnt
    while i + 32 <= length {
        let a_lo = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_lo = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let a_hi = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hi = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        // §8: consolidated validation — cmpgt(or(a,b), 15) plus negative check
        let or_lo = _mm_or_si128(a_lo, b_lo);
        let or_hi = _mm_or_si128(a_hi, b_hi);
        let invalid = _mm_or_si128(
            _mm_cmpgt_epi8(or_lo, fifteen),
            _mm_cmpgt_epi8(or_hi, fifteen),
        );
        let negative = _mm_or_si128(_mm_cmplt_epi8(or_lo, zero), _mm_cmplt_epi8(or_hi, zero));
        let bad = _mm_or_si128(invalid, negative);
        if _mm_testz_si128(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        // XOR nibbles
        let xor_lo = _mm_xor_si128(a_lo, b_lo);
        let xor_hi = _mm_xor_si128(a_hi, b_hi);

        // Pack nibble pairs into bytes: even nibbles << 4 | odd nibbles
        // Deinterleave even/odd using shuffle masks
        let shuf_even = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuf_odd = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

        // From xor_lo (16 nibbles) → 8 bytes in low half
        let even_lo = _mm_shuffle_epi8(xor_lo, shuf_even);
        let odd_lo = _mm_shuffle_epi8(xor_lo, shuf_odd);
        // From xor_hi (16 nibbles) → 8 bytes in low half
        let even_hi = _mm_shuffle_epi8(xor_hi, shuf_even);
        let odd_hi = _mm_shuffle_epi8(xor_hi, shuf_odd);

        // Combine: [even_lo_8 | even_hi_8] and [odd_lo_8 | odd_hi_8]
        // Use _mm_unpacklo_epi64 to merge the two 8-byte halves
        let even = _mm_unpacklo_epi64(even_lo, even_hi);
        let odd = _mm_unpacklo_epi64(odd_lo, odd_hi);

        // Pack: (even << 4) | odd
        // _mm_slli_epi16 shifts 16-bit lanes, so bits leak across byte
        // boundaries. Mask to keep only the high nibble per byte.
        let hi_nib_mask = _mm_set1_epi8(0xF0u8 as i8);
        let packed = _mm_or_si128(_mm_and_si128(_mm_slli_epi16(even, 4), hi_nib_mask), odd);

        // Hardware popcnt on the 16 packed bytes (extract as two u64s)
        let lo64 = _mm_cvtsi128_si64(packed) as u64;
        let hi64 = _mm_extract_epi64(packed, 1) as u64;
        difference += lo64.count_ones() as u64 + hi64.count_ones() as u64;

        i += 32;
    }

    // Process remaining 16-byte chunks with shuffle-based popcount
    let popcnt_mask = _mm_set1_epi8(0x0F);
    let popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    let mut acc = _mm_setzero_si128();
    while i + 16 <= length {
        let a_hex = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hex = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        // §8: consolidated validation
        let or_hex = _mm_or_si128(a_hex, b_hex);
        let bad = _mm_cmpgt_epi8(or_hex, fifteen);
        if _mm_testz_si128(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm_xor_si128(a_hex, b_hex);
        acc = _mm_add_epi8(
            acc,
            _mm_shuffle_epi8(popcnt_table, _mm_and_si128(xor, popcnt_mask)),
        );

        i += 16;
    }
    let sad = _mm_sad_epu8(acc, zero);
    difference += (_mm_extract_epi64(sad, 0) + _mm_extract_epi64(sad, 1)) as u64;

    // Scalar tail
    while i < length {
        let val1 = hex_char_to_nibble(*a.get_unchecked(i));
        let val2 = hex_char_to_nibble(*b.get_unchecked(i));
        if (val1 | val2) & 0xF0 != 0 {
            return Err("hex string contains invalid char");
        }
        difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        i += 1;
    }

    Ok(difference)
}

/// SSE4.1 hex string distance with early-exit at max_dist.
/// Returns Ok(u64::MAX) when distance exceeds max_dist.
#[target_feature(enable = "sse4.1", enable = "popcnt")]
pub unsafe fn hamming_distance_string_sse_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 32 {
        return hamming_distance_string_classic_with_max(a, b, max_dist);
    }

    let zero = _mm_setzero_si128();
    let fifteen = _mm_set1_epi8(15);
    let case_mask = _mm_set1_epi8(!0x20i8);
    let ascii_0 = _mm_set1_epi8(b'0' as i8);
    let seven = _mm_set1_epi8(7);
    let nine = _mm_set1_epi8(9);
    let ten = _mm_set1_epi8(10);

    let mut i = 0;
    let mut difference: u64 = 0;

    // Process 32 hex chars at a time with threshold check
    while i + 32 <= length {
        let a_lo = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_lo = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let a_hi = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hi = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        let or_lo = _mm_or_si128(a_lo, b_lo);
        let or_hi = _mm_or_si128(a_hi, b_hi);
        let invalid = _mm_or_si128(
            _mm_cmpgt_epi8(or_lo, fifteen),
            _mm_cmpgt_epi8(or_hi, fifteen),
        );
        let negative = _mm_or_si128(_mm_cmplt_epi8(or_lo, zero), _mm_cmplt_epi8(or_hi, zero));
        let bad = _mm_or_si128(invalid, negative);
        if _mm_testz_si128(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor_lo = _mm_xor_si128(a_lo, b_lo);
        let xor_hi = _mm_xor_si128(a_hi, b_hi);

        let shuf_even = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuf_odd = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

        let even_lo = _mm_shuffle_epi8(xor_lo, shuf_even);
        let odd_lo = _mm_shuffle_epi8(xor_lo, shuf_odd);
        let even_hi = _mm_shuffle_epi8(xor_hi, shuf_even);
        let odd_hi = _mm_shuffle_epi8(xor_hi, shuf_odd);

        let even = _mm_unpacklo_epi64(even_lo, even_hi);
        let odd = _mm_unpacklo_epi64(odd_lo, odd_hi);

        let hi_nib_mask = _mm_set1_epi8(0xF0u8 as i8);
        let packed = _mm_or_si128(_mm_and_si128(_mm_slli_epi16(even, 4), hi_nib_mask), odd);

        let lo64 = _mm_cvtsi128_si64(packed) as u64;
        let hi64 = _mm_extract_epi64(packed, 1) as u64;
        difference += lo64.count_ones() as u64 + hi64.count_ones() as u64;

        if difference > max_dist {
            return Ok(u64::MAX);
        }

        i += 32;
    }

    // 16-byte tail with shuffle popcount
    let popcnt_mask = _mm_set1_epi8(0x0F);
    let popcnt_table = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    while i + 16 <= length {
        let a_hex = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hex = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        let or_hex = _mm_or_si128(a_hex, b_hex);
        let bad = _mm_cmpgt_epi8(or_hex, fifteen);
        if _mm_testz_si128(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm_xor_si128(a_hex, b_hex);
        let cnt = _mm_shuffle_epi8(popcnt_table, _mm_and_si128(xor, popcnt_mask));
        let sad = _mm_sad_epu8(cnt, zero);
        difference += (_mm_extract_epi64(sad, 0) + _mm_extract_epi64(sad, 1)) as u64;

        i += 16;
    }

    // Scalar tail
    while i < length {
        let val1 = hex_char_to_nibble(*a.get_unchecked(i));
        let val2 = hex_char_to_nibble(*b.get_unchecked(i));
        if (val1 | val2) & 0xF0 != 0 {
            return Err("hex string contains invalid char");
        }
        difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        i += 1;
    }

    if difference > max_dist {
        Ok(u64::MAX)
    } else {
        Ok(difference)
    }
}

/// AVX2 hex string distance with early-exit at max_dist.
/// Returns Ok(u64::MAX) when distance exceeds max_dist.
#[target_feature(enable = "avx2", enable = "popcnt")]
pub unsafe fn hamming_distance_string_avx2_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 64 {
        return hamming_distance_string_sse_with_max(a, b, max_dist);
    }

    let zero = _mm256_setzero_si256();
    let fifteen = _mm256_set1_epi8(15);
    let case_mask = _mm256_set1_epi8(!0x20i8);
    let ascii_0 = _mm256_set1_epi8(b'0' as i8);
    let seven = _mm256_set1_epi8(7);
    let nine = _mm256_set1_epi8(9);
    let ten = _mm256_set1_epi8(10);

    let popcnt_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );

    let mut i = 0;
    let mut difference: u64 = 0;

    // Process 64 hex chars at a time with threshold check
    while i + 64 <= length {
        let a_lo = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_lo = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let a_hi = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_hi = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        let or_lo = _mm256_or_si256(a_lo, b_lo);
        let or_hi = _mm256_or_si256(a_hi, b_hi);
        let invalid = _mm256_or_si256(
            _mm256_cmpgt_epi8(or_lo, fifteen),
            _mm256_cmpgt_epi8(or_hi, fifteen),
        );
        let negative = _mm256_or_si256(
            _mm256_cmpgt_epi8(zero, or_lo),
            _mm256_cmpgt_epi8(zero, or_hi),
        );
        let bad = _mm256_or_si256(invalid, negative);
        if _mm256_testz_si256(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor_lo = _mm256_xor_si256(a_lo, b_lo);
        let xor_hi = _mm256_xor_si256(a_hi, b_hi);
        let cnt_lo = _mm256_shuffle_epi8(popcnt_lut, xor_lo);
        let cnt_hi = _mm256_shuffle_epi8(popcnt_lut, xor_hi);
        let acc = _mm256_add_epi8(cnt_lo, cnt_hi);
        let sad = _mm256_sad_epu8(acc, zero);
        let lo128 = _mm256_castsi256_si128(sad);
        let hi128 = _mm256_extracti128_si256(sad, 1);
        let sum128 = _mm_add_epi64(lo128, hi128);
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        difference += _mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)) as u64;

        if difference > max_dist {
            return Ok(u64::MAX);
        }

        i += 64;
    }

    // Fall through to SSE with_max for remaining < 64 chars
    if i < length {
        let remaining_max = max_dist.saturating_sub(difference);
        let remaining = hamming_distance_string_sse_with_max(&a[i..], &b[i..], remaining_max)?;
        if remaining == u64::MAX {
            return Ok(u64::MAX);
        }
        difference += remaining;
    }

    if difference > max_dist {
        Ok(u64::MAX)
    } else {
        Ok(difference)
    }
}

/// AVX-512 hex string distance with early-exit at max_dist.
/// Returns Ok(u64::MAX) when distance exceeds max_dist.
#[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
pub unsafe fn hamming_distance_string_avx512_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 16 {
        return hamming_distance_string_classic_with_max(a, b, max_dist);
    }

    let fifteen = _mm512_set1_epi8(15);
    let case_mask = _mm512_set1_epi8(!0x20i8);
    let ascii_0 = _mm512_set1_epi8(b'0' as i8);
    let seven = _mm512_set1_epi8(7);
    let nine = _mm512_set1_epi8(9);
    let ten = _mm512_set1_epi8(10);
    let zero = _mm512_setzero_si512();

    let mut i = 0;
    let mut difference: u64 = 0;

    // Process 64 hex chars at a time with threshold check
    while i + 64 <= length {
        let a_nib = hex_parse_avx512(
            _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_nib = hex_parse_avx512(
            _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );

        let or_nib = _mm512_or_si512(a_nib, b_nib);
        let invalid =
            _mm512_cmpgt_epi8_mask(or_nib, fifteen) | _mm512_cmpgt_epi8_mask(zero, or_nib);
        if invalid != 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm512_xor_si512(a_nib, b_nib);
        let cnt = _mm512_popcnt_epi8(xor);
        let sad = _mm512_sad_epu8(cnt, zero);
        difference += _mm512_reduce_add_epi64(sad) as u64;

        if difference > max_dist {
            return Ok(u64::MAX);
        }

        i += 64;
    }

    // Masked tail
    let remaining = length - i;
    if remaining > 0 {
        let mask = if remaining >= 64 {
            !0u64
        } else {
            (1u64 << remaining) - 1
        };
        let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
        let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);

        let a_nib = hex_parse_avx512(a_tail, case_mask, ascii_0, seven, nine, ten);
        let b_nib = hex_parse_avx512(b_tail, case_mask, ascii_0, seven, nine, ten);

        let or_nib = _mm512_or_si512(a_nib, b_nib);
        let invalid =
            (_mm512_cmpgt_epi8_mask(or_nib, fifteen) | _mm512_cmpgt_epi8_mask(zero, or_nib)) & mask;
        if invalid != 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm512_xor_si512(a_nib, b_nib);
        let cnt = _mm512_popcnt_epi8(xor);
        let sad = _mm512_sad_epu8(cnt, zero);
        difference += _mm512_reduce_add_epi64(sad) as u64;
    }

    if difference > max_dist {
        Ok(u64::MAX)
    } else {
        Ok(difference)
    }
}

// -----------------------------------------------------------------------------
// AVX-512 BITALG cross-record scanners for fixed-width catalogs (widths 16, 32).
//
// Modeled on the NEON `array_first_neon` / `array_best_neon` / `array_all_neon`
// helpers in `neon_simd.rs`. The key insight: a 512-bit ZMM register can hold
// four 16-byte records or two 32-byte records. By broadcasting the query into
// the same register, one XOR + VPOPCNTB + VPSADBW pass yields per-record
// Hamming distances for a whole batch. This trades the NEON pattern of four
// independent 128-bit ops (four `hamming_distance_neon_fixed` calls) for a
// single wider vector op per batch of four records.
//
// Semantic invariants preserved from the NEON scanners:
//   * `first` returns the lowest matching index and short-circuits on match.
//   * `best` returns (distance, index) with the lowest distance, lowest index
//     on ties, and short-circuits when it observes an exact match (d == 0).
//   * `all` returns matches in ascending index order.
//   * `max_dist < 0` disables the threshold check.
//   * The tail (records not divisible by four) uses the same fixed-width
//     kernel — bitwise identical to the batch result for a single record.
// -----------------------------------------------------------------------------

/// Compute Hamming distance between a single 16-byte record and query using
/// scalar POPCNT. Cheap enough for the tail path; the batch path is where the
/// AVX-512 win comes from.
///
/// # Safety
/// `record` and `query` must each be valid for 16 readable bytes.
#[inline(always)]
unsafe fn hamming_distance_avx512_fixed16(record: *const u8, query: *const u8) -> u64 {
    let ra = core::ptr::read_unaligned(record as *const u64);
    let rb = core::ptr::read_unaligned(record.add(8) as *const u64);
    let qa = core::ptr::read_unaligned(query as *const u64);
    let qb = core::ptr::read_unaligned(query.add(8) as *const u64);
    (ra ^ qa).count_ones() as u64 + (rb ^ qb).count_ones() as u64
}

/// Compute Hamming distance between a single 32-byte record and query using
/// scalar POPCNT.
///
/// # Safety
/// `record` and `query` must each be valid for 32 readable bytes.
#[inline(always)]
unsafe fn hamming_distance_avx512_fixed32(record: *const u8, query: *const u8) -> u64 {
    let r0 = core::ptr::read_unaligned(record as *const u64);
    let r1 = core::ptr::read_unaligned(record.add(8) as *const u64);
    let r2 = core::ptr::read_unaligned(record.add(16) as *const u64);
    let r3 = core::ptr::read_unaligned(record.add(24) as *const u64);
    let q0 = core::ptr::read_unaligned(query as *const u64);
    let q1 = core::ptr::read_unaligned(query.add(8) as *const u64);
    let q2 = core::ptr::read_unaligned(query.add(16) as *const u64);
    let q3 = core::ptr::read_unaligned(query.add(24) as *const u64);
    (r0 ^ q0).count_ones() as u64
        + (r1 ^ q1).count_ones() as u64
        + (r2 ^ q2).count_ones() as u64
        + (r3 ^ q3).count_ones() as u64
}

/// Compute four Hamming distances (four 16-byte records vs one 16-byte query)
/// in a single AVX-512 pass.
///
/// Pipeline:
///   1. Broadcast the 16-byte query to all four 128-bit lanes of a ZMM.
///   2. Load 64 bytes = four contiguous records into a ZMM.
///   3. XOR + `_mm512_popcnt_epi8` for per-byte popcount.
///   4. `_mm512_sad_epu8` sums each 8-byte lane; a 16-byte record spans two
///      adjacent 8-byte SAD lanes, so pair-sum them scalar-side.
///
/// # Safety
/// `records` must be valid for 64 readable bytes and `query` must be valid for
/// 16 readable bytes. Caller must ensure the CPU supports the target features
/// (`avx512f`, `avx512bw`, `avx512bitalg`) — the dispatcher does this via
/// `is_x86_feature_detected!`.
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
unsafe fn hamming_distance_avx512_fixed4_w16(records: *const u8, query: *const u8) -> [u64; 4] {
    let q128 = _mm_loadu_si128(query as *const __m128i);
    let q_bcast = _mm512_broadcast_i32x4(q128);
    let r = _mm512_loadu_si512(records as *const __m512i);
    let xor = _mm512_xor_si512(r, q_bcast);
    let pop = _mm512_popcnt_epi8(xor);
    // Each of the 8 qwords in `sad` holds the sum of eight per-byte popcounts.
    // A 16-byte record spans two adjacent qwords, so pair-sum {0,1},{2,3},…
    let sad = _mm512_sad_epu8(pop, _mm512_setzero_si512());
    let mut buf = [0u64; 8];
    _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, sad);
    [
        buf[0] + buf[1],
        buf[2] + buf[3],
        buf[4] + buf[5],
        buf[6] + buf[7],
    ]
}

/// Compute four Hamming distances (four 32-byte records vs one 32-byte query)
/// in two AVX-512 passes.
///
/// Two 512-bit loads cover 128 bytes = four 32-byte records. The 32-byte query
/// is broadcast to both halves of a ZMM via `_mm512_broadcast_i64x4`, then each
/// half-vector XOR + VPOPCNTB + VPSADBW reduces to two per-record distances.
///
/// # Safety
/// `records` must be valid for 128 readable bytes and `query` must be valid for
/// 32 readable bytes. Caller must ensure the CPU supports the target features.
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
unsafe fn hamming_distance_avx512_fixed4_w32(records: *const u8, query: *const u8) -> [u64; 4] {
    let q256 = _mm256_loadu_si256(query as *const __m256i);
    let q_bcast = _mm512_broadcast_i64x4(q256);
    let zero = _mm512_setzero_si512();

    let r01 = _mm512_loadu_si512(records as *const __m512i);
    let r23 = _mm512_loadu_si512(records.add(64) as *const __m512i);

    let sad01 = _mm512_sad_epu8(_mm512_popcnt_epi8(_mm512_xor_si512(r01, q_bcast)), zero);
    let sad23 = _mm512_sad_epu8(_mm512_popcnt_epi8(_mm512_xor_si512(r23, q_bcast)), zero);

    // Each 32-byte record spans four adjacent qwords in the SAD result.
    let mut buf = [0u64; 8];
    _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, sad01);
    let d0 = buf[0] + buf[1] + buf[2] + buf[3];
    let d1 = buf[4] + buf[5] + buf[6] + buf[7];
    _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, sad23);
    let d2 = buf[0] + buf[1] + buf[2] + buf[3];
    let d3 = buf[4] + buf[5] + buf[6] + buf[7];
    [d0, d1, d2, d3]
}

#[inline(always)]
fn within_fixed_threshold(distance: u64, max_dist: i64) -> bool {
    max_dist < 0 || distance <= max_dist as u64
}

// The per-width kernel wrappers below let the generic scanners stay free of
// const generics and `if WIDTH == …` branches, matching the specialization
// shape of the NEON scanners while giving LLVM straight-line code for each
// width. Function pointers are captured statically in the ArrayScanner table.

#[inline(always)]
unsafe fn scan_batch4_w16(records: *const u8, query: *const u8) -> [u64; 4] {
    hamming_distance_avx512_fixed4_w16(records, query)
}
#[inline(always)]
unsafe fn scan_batch4_w32(records: *const u8, query: *const u8) -> [u64; 4] {
    hamming_distance_avx512_fixed4_w32(records, query)
}

/// Generic scanner: find the first record index whose distance to `query` is
/// within `max_dist`. Batches of four records per AVX-512 pass; scalar tail.
///
/// # Safety
/// `big_array.len()` must be a multiple of `WIDTH`. Caller guarantees the CPU
/// supports the target features.
#[inline]
unsafe fn array_first_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    batch4: unsafe fn(*const u8, *const u8) -> [u64; 4],
    single: unsafe fn(*const u8, *const u8) -> u64,
) -> Option<usize> {
    let count = big_array.len() / WIDTH;
    let big_ptr = big_array.as_ptr();
    let query_ptr = small_array.as_ptr();
    if count == 0 {
        return None;
    }

    let first_distance = single(big_ptr, query_ptr);
    if within_fixed_threshold(first_distance, max_dist) {
        return Some(0);
    }
    let mut index = 1;

    while index + 4 <= count {
        let distances = batch4(big_ptr.add(index * WIDTH), query_ptr);
        for (lane, &distance) in distances.iter().enumerate() {
            if within_fixed_threshold(distance, max_dist) {
                return Some(index + lane);
            }
        }
        index += 4;
    }
    while index < count {
        let distance = single(big_ptr.add(index * WIDTH), query_ptr);
        if within_fixed_threshold(distance, max_dist) {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Generic scanner: find (distance, index) of the record closest to `query`
/// within `max_dist`, breaking ties by lowest index. Short-circuits on exact
/// match (distance == 0).
///
/// # Safety
/// Same as `array_first_avx512`.
#[inline]
unsafe fn array_best_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    batch4: unsafe fn(*const u8, *const u8) -> [u64; 4],
    single: unsafe fn(*const u8, *const u8) -> u64,
) -> Option<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let big_ptr = big_array.as_ptr();
    let query_ptr = small_array.as_ptr();
    if count == 0 {
        return None;
    }

    let first_distance = single(big_ptr, query_ptr);
    let mut best = within_fixed_threshold(first_distance, max_dist).then_some((first_distance, 0));
    if first_distance == 0 {
        return best;
    }
    let mut index = 1;

    while index + 4 <= count {
        let distances = batch4(big_ptr.add(index * WIDTH), query_ptr);
        for (lane, &distance) in distances.iter().enumerate() {
            let candidate_index = index + lane;
            let eligible = match best {
                Some((best_distance, _)) => distance < best_distance,
                None => within_fixed_threshold(distance, max_dist),
            };
            if !eligible {
                continue;
            }
            if best.is_none() || distance < best.unwrap().0 {
                best = Some((distance, candidate_index));
                if distance == 0 {
                    return best;
                }
            }
        }
        index += 4;
    }

    while index < count {
        let distance = single(big_ptr.add(index * WIDTH), query_ptr);
        let eligible = match best {
            Some((best_distance, _)) => distance < best_distance,
            None => within_fixed_threshold(distance, max_dist),
        };
        if eligible {
            best = Some((distance, index));
            if distance == 0 {
                return best;
            }
        }
        index += 1;
    }
    best
}

/// Generic scanner: collect all (distance, index) pairs in ascending-index
/// order with `distance <= max_dist` (or all if `max_dist < 0`).
///
/// # Safety
/// Same as `array_first_avx512`.
#[inline]
unsafe fn array_all_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
    batch4: unsafe fn(*const u8, *const u8) -> [u64; 4],
    single: unsafe fn(*const u8, *const u8) -> u64,
) -> Vec<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let mut matches = Vec::new();
    let mut index = 0;
    let big_ptr = big_array.as_ptr();
    let query_ptr = small_array.as_ptr();

    while index + 4 <= count {
        let distances = batch4(big_ptr.add(index * WIDTH), query_ptr);
        for (lane, &distance) in distances.iter().enumerate() {
            if within_fixed_threshold(distance, max_dist) {
                matches.push((distance, index + lane));
            }
        }
        index += 4;
    }

    while index < count {
        let distance = single(big_ptr.add(index * WIDTH), query_ptr);
        if within_fixed_threshold(distance, max_dist) {
            matches.push((distance, index));
        }
        index += 1;
    }
    matches
}

// -----------------------------------------------------------------------------
// Public (crate-visible) scanner entry points. `select_array_scanner_for_width`
// captures these as function pointers, so the runtime feature check has already
// happened at the point they are invoked.
// -----------------------------------------------------------------------------

/// # Safety
/// Caller must ensure `avx512f + avx512bw + avx512bitalg` are available and
/// that `big_array.len() % 16 == 0`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_first_avx512_16(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    array_first_avx512::<16>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w16,
        hamming_distance_avx512_fixed16,
    )
}

/// # Safety
/// Same as `array_first_avx512_16`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_best_avx512_16(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    array_best_avx512::<16>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w16,
        hamming_distance_avx512_fixed16,
    )
}

/// # Safety
/// Same as `array_first_avx512_16`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_all_avx512_16(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    array_all_avx512::<16>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w16,
        hamming_distance_avx512_fixed16,
    )
}

/// # Safety
/// Caller must ensure `avx512f + avx512bw + avx512bitalg` are available and
/// that `big_array.len() % 32 == 0`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_first_avx512_32(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    array_first_avx512::<32>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w32,
        hamming_distance_avx512_fixed32,
    )
}

/// # Safety
/// Same as `array_first_avx512_32`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_best_avx512_32(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    array_best_avx512::<32>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w32,
        hamming_distance_avx512_fixed32,
    )
}

/// # Safety
/// Same as `array_first_avx512_32`.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512bitalg")]
pub(crate) unsafe fn array_all_avx512_32(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    array_all_avx512::<32>(
        big_array,
        small_array,
        max_dist,
        scan_batch4_w32,
        hamming_distance_avx512_fixed32,
    )
}

// -----------------------------------------------------------------------------
// Feature-checked trampolines used by the `ArrayScanner` function-pointer table
// in `api.rs`. Function pointers cannot carry `#[target_feature]`, so these
// safe wrappers re-check the feature at every call (cheap after the first
// invocation because `is_x86_feature_detected!` caches the result).
// -----------------------------------------------------------------------------

#[inline]
pub(crate) fn array_first_avx512_16_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    unsafe { array_first_avx512_16(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_best_avx512_16_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    unsafe { array_best_avx512_16(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_all_avx512_16_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    unsafe { array_all_avx512_16(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_first_avx512_32_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    unsafe { array_first_avx512_32(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_best_avx512_32_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    unsafe { array_best_avx512_32(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_all_avx512_32_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    unsafe { array_all_avx512_32(big_array, small_array, max_dist) }
}

/// Scalar fallback for hex string distance with max_dist.
#[inline]
unsafe fn hamming_distance_string_classic_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();
    let mut difference: u64 = 0;
    let mut i = 0;
    while i < length {
        let val1 = hex_char_to_nibble(*a.get_unchecked(i));
        let val2 = hex_char_to_nibble(*b.get_unchecked(i));
        if (val1 | val2) & 0xF0 != 0 {
            return Err("hex string contains invalid char");
        }
        difference += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        if difference > max_dist {
            return Ok(u64::MAX);
        }
        i += 1;
    }
    Ok(difference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512_masked_byte_inputs_match_scalar_results() {
        if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512bitalg") {
            return;
        }

        for length in [1usize, 7, 8, 15, 16, 31, 32, 48, 63] {
            let a = vec![0xFF; length];
            let b = vec![0x00; length];
            let expected = (length * 8) as u64;

            unsafe {
                assert_eq!(hamming_distance_bytes_avx512(&a, &b, -1), expected);
                assert_eq!(
                    hamming_distance_bytes_avx512(&a, &b, expected as i64),
                    expected
                );
                assert_eq!(
                    hamming_distance_bytes_avx512(&a, &b, expected as i64 - 1),
                    u64::MAX
                );
            }
        }
    }

    fn avx512_scanner_hw_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512bitalg")
    }

    fn scalar_byte_distance(a: &[u8], b: &[u8]) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones() as u64)
            .sum()
    }

    // Deterministic PRNG so the batched-vs-scalar comparisons stay
    // reproducible when this test runs on real AVX-512 hardware.
    struct SplitMix {
        state: u64,
    }
    impl SplitMix {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next(&mut self) -> u8 {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 56) as u8
        }
        fn vec(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn oracle_first_best_all(
        big: &[u8],
        small: &[u8],
        max_dist: i64,
    ) -> (Option<usize>, Option<(u64, usize)>, Vec<(u64, usize)>) {
        let mut first = None;
        let mut best: Option<(u64, usize)> = None;
        let mut all = Vec::new();
        for (i, record) in big.chunks_exact(small.len()).enumerate() {
            let d = scalar_byte_distance(record, small);
            if max_dist >= 0 && d > max_dist as u64 {
                continue;
            }
            if first.is_none() {
                first = Some(i);
            }
            best = match best {
                Some((bd, bi)) if bd < d || (bd == d && bi < i) => Some((bd, bi)),
                _ => Some((d, i)),
            };
            all.push((d, i));
        }
        (first, best, all)
    }

    // Batch-of-four kernel produces the same per-record distances as a scalar
    // popcount. Exercises the pair-sum reduction of adjacent SAD qwords.
    #[test]
    fn avx512_fixed4_w16_matches_scalar() {
        if !avx512_scanner_hw_available() {
            return;
        }
        let mut rng = SplitMix::new(0xC0FFEE_D15EA5E);
        for _ in 0..8 {
            let records = rng.vec(64);
            let query = rng.vec(16);
            let expected = [
                scalar_byte_distance(&records[0..16], &query),
                scalar_byte_distance(&records[16..32], &query),
                scalar_byte_distance(&records[32..48], &query),
                scalar_byte_distance(&records[48..64], &query),
            ];
            let actual =
                unsafe { hamming_distance_avx512_fixed4_w16(records.as_ptr(), query.as_ptr()) };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn avx512_fixed4_w32_matches_scalar() {
        if !avx512_scanner_hw_available() {
            return;
        }
        let mut rng = SplitMix::new(0xDEADBEEF_FEEDFACE);
        for _ in 0..8 {
            let records = rng.vec(128);
            let query = rng.vec(32);
            let expected = [
                scalar_byte_distance(&records[0..32], &query),
                scalar_byte_distance(&records[32..64], &query),
                scalar_byte_distance(&records[64..96], &query),
                scalar_byte_distance(&records[96..128], &query),
            ];
            let actual =
                unsafe { hamming_distance_avx512_fixed4_w32(records.as_ptr(), query.as_ptr()) };
            assert_eq!(actual, expected);
        }
    }

    // Full-scanner semantic parity for widths 16 and 32 across a matrix of
    // catalog sizes (including sizes not divisible by four to exercise the
    // scalar tail) and thresholds (first/best/all ordering, ties, exact match
    // short-circuit, and the `max_dist < 0` catch-all path).
    fn assert_scanners_match_oracle(width: usize, count: usize, seed: u64) {
        let mut rng = SplitMix::new(seed);
        let small = rng.vec(width);
        let mut big = rng.vec(count * width);

        // Seed multiple exact-match records to test lowest-index tie behavior,
        // and one near-match record to give us threshold cases.
        let match_indices = if count >= 8 {
            vec![1usize, count / 2, count - 1]
        } else {
            vec![0usize.min(count.saturating_sub(1))]
        };
        for &idx in &match_indices {
            let record = &mut big[idx * width..(idx + 1) * width];
            record.copy_from_slice(&small);
        }
        if count >= 4 {
            let near = count / 3;
            big[near * width..(near + 1) * width].copy_from_slice(&small);
            big[near * width] ^= 0xF0;
        }

        for &max_dist in &[-1i64, 0, 3, 4, 5, 8, 128] {
            let (efirst, ebest, eall) = oracle_first_best_all(&big, &small, max_dist);

            let (afirst, abest, aall) = if width == 16 {
                (
                    unsafe { array_first_avx512_16(&big, &small, max_dist) },
                    unsafe { array_best_avx512_16(&big, &small, max_dist) },
                    unsafe { array_all_avx512_16(&big, &small, max_dist) },
                )
            } else {
                (
                    unsafe { array_first_avx512_32(&big, &small, max_dist) },
                    unsafe { array_best_avx512_32(&big, &small, max_dist) },
                    unsafe { array_all_avx512_32(&big, &small, max_dist) },
                )
            };
            assert_eq!(
                afirst, efirst,
                "first mismatch width={width} count={count} max_dist={max_dist}"
            );
            assert_eq!(
                abest, ebest,
                "best mismatch width={width} count={count} max_dist={max_dist}"
            );
            assert_eq!(
                aall, eall,
                "all mismatch width={width} count={count} max_dist={max_dist}"
            );
        }
    }

    #[test]
    fn avx512_scanners_w16_random_oracle_various_counts() {
        if !avx512_scanner_hw_available() {
            return;
        }
        // Counts include values just below/at/above a batch of 4 to exercise
        // the batch/tail boundary, plus a larger catalog.
        for &count in &[1usize, 3, 4, 5, 7, 8, 15, 33, 64, 512] {
            assert_scanners_match_oracle(16, count, 0x1234_5678 ^ count as u64);
        }
    }

    #[test]
    fn avx512_scanners_w32_random_oracle_various_counts() {
        if !avx512_scanner_hw_available() {
            return;
        }
        for &count in &[1usize, 3, 4, 5, 7, 8, 15, 33, 64, 512] {
            assert_scanners_match_oracle(32, count, 0xCAFEBABE ^ count as u64);
        }
    }

    // `best` must return the lowest index among ties and short-circuit on
    // an exact match (distance == 0). Include several exact matches to
    // guarantee both conditions.
    #[test]
    fn avx512_scanner_best_preserves_lowest_index_tie() {
        if !avx512_scanner_hw_available() {
            return;
        }
        let width = 16usize;
        let small = vec![0x5Au8; width];
        let count = 20usize;
        let mut big = vec![0xA5u8; count * width];
        for &i in &[3usize, 3, 7, 13] {
            big[i * width..(i + 1) * width].copy_from_slice(&small);
        }
        // Two records with distance == 1 by flipping one bit.
        for &i in &[5usize, 15] {
            big[i * width..(i + 1) * width].copy_from_slice(&small);
            big[i * width] ^= 0x01;
        }
        assert_eq!(unsafe { array_first_avx512_16(&big, &small, 0) }, Some(3));
        assert_eq!(
            unsafe { array_best_avx512_16(&big, &small, -1) },
            Some((0, 3))
        );
        let all = unsafe { array_all_avx512_16(&big, &small, 1) };
        assert_eq!(all, vec![(0, 3), (1, 5), (0, 7), (0, 13), (1, 15)]);
    }
}
