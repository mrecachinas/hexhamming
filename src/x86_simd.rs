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
        // Early termination path - check every 16 bytes
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        while i + 16 <= length {
            let a16 = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let b16 = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let xor = _mm_xor_si128(a16, b16);
            difference += popcnt128_sse(xor);
            if difference > max_dist_u64 {
                return 0;
            }
            i += 16;
        }

        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
    }
}

/// AVX2 VPSHUFB-based popcount - accumulates into byte lanes
#[target_feature(enable = "avx2")]
unsafe fn popcnt256_shuffle(v: __m256i, mask: __m256i, table: __m256i) -> __m256i {
    let lo = _mm256_and_si256(v, mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
    _mm256_add_epi8(_mm256_shuffle_epi8(table, lo), _mm256_shuffle_epi8(table, hi))
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
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
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
        // Early termination path
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        // Use batched counting but check periodically (every 128 bytes)
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
            difference += (_mm256_extract_epi64(sad, 0)
                + _mm256_extract_epi64(sad, 1)
                + _mm256_extract_epi64(sad, 2)
                + _mm256_extract_epi64(sad, 3)) as u64;
            if difference > max_dist_u64 {
                return 0;
            }
        }

        // Process remaining 32-byte chunks
        while i + 32 <= length {
            let a32 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let b32 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let xor = _mm256_xor_si256(a32, b32);
            let cnt = popcnt256_shuffle(xor, mask, table);
            let sad = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
            difference += (_mm256_extract_epi64(sad, 0)
                + _mm256_extract_epi64(sad, 1)
                + _mm256_extract_epi64(sad, 2)
                + _mm256_extract_epi64(sad, 3)) as u64;
            if difference > max_dist_u64 {
                return 0;
            }
            i += 32;
        }

        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            if difference > max_dist_u64 {
                return 0;
            }
            i += 1;
        }
        1
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

/// AVX2 pack-to-bytes implementation for hex strings.
/// Parses 64 hex chars (2×32) → nibbles, XORs, packs pairs into bytes,
/// then uses hardware popcnt on u64 extracts.
#[target_feature(enable = "avx2", enable = "popcnt")]
pub unsafe fn hamming_distance_string_avx2(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    // Fall back to SSE for inputs < 64 chars
    if length < 64 {
        return hamming_distance_string_sse(a, b);
    }

    let zero = _mm256_setzero_si256();
    let fifteen = _mm256_set1_epi8(15);
    let case_mask = _mm256_set1_epi8(!0x20i8);     // 0xDF
    let ascii_0 = _mm256_set1_epi8(b'0' as i8);
    let seven = _mm256_set1_epi8(7);
    let nine = _mm256_set1_epi8(9);
    let ten = _mm256_set1_epi8(10);

    let mut i = 0;
    let mut difference: u64 = 0;

    // Process 64 hex chars at a time: 2×32 chars → parse → XOR → pack → popcnt
    while i + 64 <= length {
        let a_lo = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_lo = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let a_hi = hex_parse_avx2(
            _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_hi = hex_parse_avx2(
            _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i),
            case_mask, ascii_0, seven, nine, ten,
        );

        // Validate: any lane > 15 or < 0 (signed) means invalid
        let invalid = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi8(a_lo, fifteen),
                _mm256_cmpgt_epi8(b_lo, fifteen),
            ),
            _mm256_or_si256(
                _mm256_cmpgt_epi8(a_hi, fifteen),
                _mm256_cmpgt_epi8(b_hi, fifteen),
            ),
        );
        let negative = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi8(zero, a_lo),
                _mm256_cmpgt_epi8(zero, b_lo),
            ),
            _mm256_or_si256(
                _mm256_cmpgt_epi8(zero, a_hi),
                _mm256_cmpgt_epi8(zero, b_hi),
            ),
        );
        let bad = _mm256_or_si256(invalid, negative);
        if _mm256_testz_si256(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        // XOR nibbles
        let xor_lo = _mm256_xor_si256(a_lo, b_lo);
        let xor_hi = _mm256_xor_si256(a_hi, b_hi);

        // Pack nibble pairs into bytes using VPSHUFB to deinterleave
        let shuf_even = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let shuf_odd = _mm256_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
        );

        // AVX2 VPSHUFB operates within 128-bit lanes, so each 256-bit register
        // gives us 8 even/odd bytes in low half of each 128-bit lane.
        // We need to collect them: use _mm256_unpacklo_epi64 to merge within lanes,
        // then _mm256_permute4x64_epi64 to consolidate across lanes.
        let even_lo = _mm256_shuffle_epi8(xor_lo, shuf_even);
        let odd_lo  = _mm256_shuffle_epi8(xor_lo, shuf_odd);
        let even_hi = _mm256_shuffle_epi8(xor_hi, shuf_even);
        let odd_hi  = _mm256_shuffle_epi8(xor_hi, shuf_odd);

        // Merge: low 64 bits of each lane contain our data
        // unpacklo_epi64 merges lo+hi within each 128-bit lane
        let even_merged = _mm256_unpacklo_epi64(even_lo, even_hi);
        let odd_merged  = _mm256_unpacklo_epi64(odd_lo, odd_hi);

        // Consolidate: permute to put lane0_lo64, lane0_hi64, lane1_lo64, lane1_hi64
        // into contiguous order. After unpacklo, layout is:
        //   lane0: [even_lo_lane0_8B | even_hi_lane0_8B]
        //   lane1: [even_lo_lane1_8B | even_hi_lane1_8B]
        // We want: [even_lo_lane0 | even_hi_lane0 | even_lo_lane1 | even_hi_lane1]
        // which is already the natural order: q0, q1, q2, q3 → permute 0,2,1,3
        let even = _mm256_permute4x64_epi64(even_merged, 0b11_01_10_00); // 0,2,1,3
        let odd  = _mm256_permute4x64_epi64(odd_merged, 0b11_01_10_00);

        // Pack: (even << 4) | odd, with mask to prevent cross-byte leakage
        let hi_nib_mask = _mm256_set1_epi8(0xF0u8 as i8);
        let packed = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(even, 4), hi_nib_mask),
            odd,
        );

        // Hardware popcnt on 32 packed bytes (extract as four u64s)
        let v128_lo = _mm256_castsi256_si128(packed);
        let v128_hi = _mm256_extracti128_si256(packed, 1);
        difference += (_mm_cvtsi128_si64(v128_lo) as u64).count_ones() as u64
            + (_mm_extract_epi64(v128_lo, 1) as u64).count_ones() as u64
            + (_mm_cvtsi128_si64(v128_hi) as u64).count_ones() as u64
            + (_mm_extract_epi64(v128_hi, 1) as u64).count_ones() as u64;

        i += 64;
    }

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
    let ones = _mm512_set1_epi8(-1);  // 0xFF
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
    let case_mask = _mm512_set1_epi8(!0x20i8);     // 0xDF
    let ascii_0 = _mm512_set1_epi8(b'0' as i8);
    let seven = _mm512_set1_epi8(7);
    let nine = _mm512_set1_epi8(9);
    let ten = _mm512_set1_epi8(10);
    let zero = _mm512_setzero_si512();

    let mut i = 0;
    let mut total = _mm512_setzero_si512();

    // Process 64 hex chars at a time
    // Each nibble XOR produces at most 4 set bits, so per-byte popcount
    // accumulator maxes at 4 per lane. Safe to accumulate 63 iterations
    // (63 * 4 = 252 < 255) before horizontal sum. For <256 chars we
    // never exceed 4 iterations, so no overflow concern.
    while i + 64 <= length {
        let a_nib = hex_parse_avx512(
            _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_nib = hex_parse_avx512(
            _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i),
            case_mask, ascii_0, seven, nine, ten,
        );

        // Validate: any nibble > 15 or < 0 (signed) means invalid
        let invalid = _mm512_cmpgt_epi8_mask(a_nib, fifteen)
            | _mm512_cmpgt_epi8_mask(b_nib, fifteen)
            | _mm512_cmpgt_epi8_mask(zero, a_nib)
            | _mm512_cmpgt_epi8_mask(zero, b_nib);
        if invalid != 0 {
            return Err("hex string contains invalid char");
        }

        // XOR nibbles and VPOPCNTB — counts set bits per byte
        let xor = _mm512_xor_si512(a_nib, b_nib);
        let cnt = _mm512_popcnt_epi8(xor);
        total = _mm512_add_epi8(total, cnt);

        i += 64;
    }

    // Horizontal sum via SAD against zero → u64 lanes
    let sad = _mm512_sad_epu8(total, zero);
    let mut difference = _mm512_reduce_add_epi64(sad) as u64;

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

        // Validate only the active lanes
        let invalid = (_mm512_cmpgt_epi8_mask(a_nib, fifteen)
            | _mm512_cmpgt_epi8_mask(b_nib, fifteen)
            | _mm512_cmpgt_epi8_mask(zero, a_nib)
            | _mm512_cmpgt_epi8_mask(zero, b_nib))
            & mask;
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

    if length < 64 {
        return hamming_distance_bytes_avx2(a, b, max_dist);
    }

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
            let mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
            let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
            let xor = _mm512_xor_si512(a_tail, b_tail);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
        }
        difference
    } else {
        // Early termination path
        let max_dist_u64 = max_dist as u64;
        let mut difference: u64 = 0;

        while i + 64 <= length {
            let a64 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let b64 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let xor = _mm512_xor_si512(a64, b64);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
            if difference > max_dist_u64 {
                return 0;
            }
            i += 64;
        }

        // Masked tail for early termination path
        let remaining = length - i;
        if remaining > 0 {
            let mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let a_tail = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
            let b_tail = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
            let xor = _mm512_xor_si512(a_tail, b_tail);
            let cnt = _mm512_popcnt_epi8(xor);
            let sad = _mm512_sad_epu8(cnt, zero);
            difference += _mm512_reduce_add_epi64(sad) as u64;
            if difference > max_dist_u64 {
                return 0;
            }
        }
        1
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
    let case_mask = _mm_set1_epi8(!0x20i8);     // 0xDF
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
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_lo = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let a_hi = hex_parse_sse(
            _mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i),
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_hi = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i),
            case_mask, ascii_0, seven, nine, ten,
        );

        // Validate all 4 vectors: any lane > 15 or < 0 (signed) means invalid
        let invalid = _mm_or_si128(
            _mm_or_si128(
                _mm_cmpgt_epi8(a_lo, fifteen),
                _mm_cmpgt_epi8(b_lo, fifteen),
            ),
            _mm_or_si128(
                _mm_cmpgt_epi8(a_hi, fifteen),
                _mm_cmpgt_epi8(b_hi, fifteen),
            ),
        );
        let negative = _mm_or_si128(
            _mm_or_si128(
                _mm_cmplt_epi8(a_lo, zero),
                _mm_cmplt_epi8(b_lo, zero),
            ),
            _mm_or_si128(
                _mm_cmplt_epi8(a_hi, zero),
                _mm_cmplt_epi8(b_hi, zero),
            ),
        );
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
        let shuf_odd  = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

        // From xor_lo (16 nibbles) → 8 bytes in low half
        let even_lo = _mm_shuffle_epi8(xor_lo, shuf_even);
        let odd_lo  = _mm_shuffle_epi8(xor_lo, shuf_odd);
        // From xor_hi (16 nibbles) → 8 bytes in low half
        let even_hi = _mm_shuffle_epi8(xor_hi, shuf_even);
        let odd_hi  = _mm_shuffle_epi8(xor_hi, shuf_odd);

        // Combine: [even_lo_8 | even_hi_8] and [odd_lo_8 | odd_hi_8]
        // Use _mm_unpacklo_epi64 to merge the two 8-byte halves
        let even = _mm_unpacklo_epi64(even_lo, even_hi);
        let odd  = _mm_unpacklo_epi64(odd_lo, odd_hi);

        // Pack: (even << 4) | odd
        // _mm_slli_epi16 shifts 16-bit lanes, so bits leak across byte
        // boundaries. Mask to keep only the high nibble per byte.
        let hi_nib_mask = _mm_set1_epi8(0xF0u8 as i8);
        let packed = _mm_or_si128(
            _mm_and_si128(_mm_slli_epi16(even, 4), hi_nib_mask),
            odd,
        );

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
            case_mask, ascii_0, seven, nine, ten,
        );
        let b_hex = hex_parse_sse(
            _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            case_mask, ascii_0, seven, nine, ten,
        );

        let bad = _mm_or_si128(
            _mm_cmpgt_epi8(a_hex, fifteen),
            _mm_cmpgt_epi8(b_hex, fifteen),
        );
        if _mm_testz_si128(bad, bad) == 0 {
            return Err("hex string contains invalid char");
        }

        let xor = _mm_xor_si128(a_hex, b_hex);
        acc = _mm_add_epi8(acc, _mm_shuffle_epi8(popcnt_table,
            _mm_and_si128(xor, popcnt_mask)));
        
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
