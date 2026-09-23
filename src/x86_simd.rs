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
#[target_feature(enable = "avx2", enable = "popcnt")]
pub unsafe fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut i = 0;

    // The AVX2 dispatcher only selects this function after AVX2+POPCNT are
    // detected, so small inputs can use an inline POPCNT path instead of
    // re-running SSE/POPCNT feature detection and tail-calling the SSE kernel.
    if length < 64 {
        return hamming_distance_bytes_popcnt(a, b, max_dist);
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

#[target_feature(enable = "popcnt")]
unsafe fn hamming_distance_bytes_popcnt(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut difference: u64 = 0;
    let mut i = 0;
    let max_dist_u64 = max_dist as u64;

    while i + 32 <= length {
        difference += popcnt_record_32(a.as_ptr().add(i), b.as_ptr().add(i));
        if max_dist >= 0 && difference > max_dist_u64 {
            return u64::MAX;
        }
        i += 32;
    }
    while i + 8 <= length {
        let av = core::ptr::read_unaligned(a.as_ptr().add(i) as *const u64);
        let bv = core::ptr::read_unaligned(b.as_ptr().add(i) as *const u64);
        difference += (av ^ bv).count_ones() as u64;
        i += 8;
    }
    while i < length {
        difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
        i += 1;
    }
    if max_dist >= 0 && difference > max_dist_u64 {
        u64::MAX
    } else {
        difference
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
    let mut bad_acc = _mm256_setzero_si256();

    // Process 64 hex chars × 2 iterations (128 chars) per batched SAD+accumulate.
    // Each nibble XOR produces max 4 set bits, and we accumulate popcount values
    // 0-4 per byte. Smaller batches keep parsed nibbles in registers on AVX2
    // targets with only 16 YMM registers.
    while i + 128 <= length {
        let mut acc = _mm256_setzero_si256();
        for _ in 0..2 {
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
            bad_acc = _mm256_or_si256(bad_acc, _mm256_or_si256(invalid, negative));

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
        bad_acc = _mm256_or_si256(bad_acc, _mm256_or_si256(invalid, negative));

        let xor_lo = _mm256_xor_si256(a_lo, b_lo);
        let xor_hi = _mm256_xor_si256(a_hi, b_hi);
        let cnt_lo = _mm256_shuffle_epi8(popcnt_lut, xor_lo);
        let cnt_hi = _mm256_shuffle_epi8(popcnt_lut, xor_hi);
        acc = _mm256_add_epi8(acc, _mm256_add_epi8(cnt_lo, cnt_hi));

        i += 64;
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(acc, zero));

    if _mm256_testz_si256(bad_acc, bad_acc) == 0 {
        return Err("hex string contains invalid char");
    }

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

const AVX512_HEX_TAG_LUT: [u8; 128] = {
    let mut table = [0x80u8; 128];
    let mut i = 0usize;
    while i < 10 {
        table[b'0' as usize + i] = i as u8;
        i += 1;
    }
    i = 0;
    while i < 6 {
        table[b'A' as usize + i] = (10 + i) as u8;
        table[b'a' as usize + i] = (10 + i) as u8;
        i += 1;
    }
    table
};

#[inline]
#[target_feature(enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn hex_parse_avx512_vbmi(chars: __m512i, lo: __m512i, hi: __m512i) -> __m512i {
    let idx = _mm512_and_si512(chars, _mm512_set1_epi8(0x7F));
    _mm512_permutex2var_epi8(lo, idx, hi)
}

#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vbmi",
    enable = "avx512bitalg",
    enable = "popcnt"
)]
pub unsafe fn hamming_distance_string_avx512_vbmi(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();
    if length < 16 {
        return hamming_distance_string_classic(a, b);
    }

    let lut_lo = _mm512_loadu_si512(AVX512_HEX_TAG_LUT.as_ptr() as *const __m512i);
    let lut_hi = _mm512_loadu_si512(AVX512_HEX_TAG_LUT.as_ptr().add(64) as *const __m512i);
    let tag = _mm512_set1_epi8(0x80u8 as i8);
    let zero = _mm512_setzero_si512();
    let mut bad = zero;
    let mut total = zero;
    let mut i = 0usize;

    const BATCH: usize = 32;
    while i + 64 <= length {
        let mut acc = zero;
        let mut n = 0usize;
        while n < BATCH && i + 64 <= length {
            let ac = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let bc = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let an = hex_parse_avx512_vbmi(ac, lut_lo, lut_hi);
            let bn = hex_parse_avx512_vbmi(bc, lut_lo, lut_hi);
            bad = _mm512_or_si512(
                bad,
                _mm512_or_si512(_mm512_or_si512(ac, bc), _mm512_or_si512(an, bn)),
            );
            acc = _mm512_add_epi8(acc, _mm512_popcnt_epi8(_mm512_xor_si512(an, bn)));
            i += 64;
            n += 1;
        }
        total = _mm512_add_epi64(total, _mm512_sad_epu8(acc, zero));
    }

    let remaining = length - i;
    if remaining > 0 {
        let mask = (1u64 << remaining) - 1;
        let ac = _mm512_maskz_loadu_epi8(mask, a.as_ptr().add(i) as *const i8);
        let bc = _mm512_maskz_loadu_epi8(mask, b.as_ptr().add(i) as *const i8);
        let an = hex_parse_avx512_vbmi(ac, lut_lo, lut_hi);
        let bn = hex_parse_avx512_vbmi(bc, lut_lo, lut_hi);
        // Masked-off lanes load as NUL, which parses to the invalid tag; only
        // the loaded lanes may flag an invalid char. (Their XOR is 0 ^ 0 of
        // equal tags, so they add nothing to the count.)
        bad = _mm512_or_si512(
            bad,
            _mm512_maskz_mov_epi8(
                mask,
                _mm512_or_si512(_mm512_or_si512(ac, bc), _mm512_or_si512(an, bn)),
            ),
        );
        let cnt = _mm512_popcnt_epi8(_mm512_xor_si512(an, bn));
        total = _mm512_add_epi64(total, _mm512_sad_epu8(cnt, zero));
    }

    if _mm512_movepi8_mask(_mm512_and_si512(bad, tag)) != 0 {
        return Err("hex string contains invalid char");
    }
    Ok(_mm512_reduce_add_epi64(total) as u64)
}

/// AVX-512 BITALG implementation for byte arrays.
/// XOR + VPOPCNTB for native per-byte popcount.
#[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
pub unsafe fn hamming_distance_bytes_avx512(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();
    let mut i = 0;

    // Below 32 bytes the masked-load tail (mask synthesis, two masked loads,
    // VPOPCNTB, VPSADBW and a horizontal reduce) costs more than at most
    // three 64-bit POPCNTs plus a byte tail.
    if length < 32 {
        return hamming_distance_bytes_popcnt(a, b, max_dist);
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
// AVX2/POPCNT fixed-width catalog scanners and pairwise kernels.
//
// AVX2-only hosts previously had no fixed-width scanner, so catalog scans made
// one indirect byte-kernel call per record; 16/32-byte records then re-entered
// the AVX2 small-input fallback and tail-called SSE. These loops keep the
// resolved AVX2+POPCNT feature contract but make the fixed widths straight-line
// record loops with explicit batching and no per-record function call.
// -----------------------------------------------------------------------------

#[inline(always)]
unsafe fn popcnt_record_8(record: *const u8, query: *const u8) -> u64 {
    let r0 = core::ptr::read_unaligned(record as *const u64);
    let q0 = core::ptr::read_unaligned(query as *const u64);
    (r0 ^ q0).count_ones() as u64
}

#[inline(always)]
unsafe fn popcnt_record_16(record: *const u8, query: *const u8) -> u64 {
    popcnt_record_8(record, query) + popcnt_record_8(record.add(8), query.add(8))
}

#[inline(always)]
unsafe fn popcnt_record_32(record: *const u8, query: *const u8) -> u64 {
    popcnt_record_16(record, query) + popcnt_record_16(record.add(16), query.add(16))
}

#[inline(always)]
unsafe fn popcnt_record_64(record: *const u8, query: *const u8) -> u64 {
    popcnt_record_32(record, query) + popcnt_record_32(record.add(32), query.add(32))
}

#[inline(always)]
unsafe fn avx2_distance_for_width<const WIDTH: usize>(record: *const u8, query: *const u8) -> u64 {
    match WIDTH {
        8 => popcnt_record_8(record, query),
        16 => popcnt_record_16(record, query),
        32 => popcnt_record_32(record, query),
        64 => popcnt_record_64(record, query),
        _ => core::hint::unreachable_unchecked(),
    }
}

#[inline(always)]
unsafe fn array_first_avx2<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    let count = big_array.len() / WIDTH;
    let query = small_array.as_ptr();
    let records = big_array.as_ptr();
    let mut index = 0usize;

    while index + 8 <= count {
        for lane in 0..8 {
            let d = avx2_distance_for_width::<WIDTH>(records.add((index + lane) * WIDTH), query);
            if within_fixed_threshold(d, max_dist) {
                return Some(index + lane);
            }
        }
        index += 8;
    }
    while index < count {
        let d = avx2_distance_for_width::<WIDTH>(records.add(index * WIDTH), query);
        if within_fixed_threshold(d, max_dist) {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[inline(always)]
unsafe fn array_best_avx2<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let query = small_array.as_ptr();
    let records = big_array.as_ptr();
    let mut best: Option<(u64, usize)> = None;
    let mut index = 0usize;

    while index + 8 <= count {
        for lane in 0..8 {
            let candidate_index = index + lane;
            let d = avx2_distance_for_width::<WIDTH>(records.add(candidate_index * WIDTH), query);
            let eligible = match best {
                Some((best_distance, _)) => d < best_distance,
                None => within_fixed_threshold(d, max_dist),
            };
            if eligible {
                best = Some((d, candidate_index));
                if d == 0 {
                    return best;
                }
            }
        }
        index += 8;
    }
    while index < count {
        let d = avx2_distance_for_width::<WIDTH>(records.add(index * WIDTH), query);
        let eligible = match best {
            Some((best_distance, _)) => d < best_distance,
            None => within_fixed_threshold(d, max_dist),
        };
        if eligible {
            best = Some((d, index));
            if d == 0 {
                return best;
            }
        }
        index += 1;
    }
    best
}

#[inline(always)]
unsafe fn array_all_avx2<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let query = small_array.as_ptr();
    let records = big_array.as_ptr();
    let mut out = Vec::new();
    let mut index = 0usize;

    while index + 8 <= count {
        for lane in 0..8 {
            let candidate_index = index + lane;
            let d = avx2_distance_for_width::<WIDTH>(records.add(candidate_index * WIDTH), query);
            if within_fixed_threshold(d, max_dist) {
                out.push((d, candidate_index));
            }
        }
        index += 8;
    }
    while index < count {
        let d = avx2_distance_for_width::<WIDTH>(records.add(index * WIDTH), query);
        if within_fixed_threshold(d, max_dist) {
            out.push((d, index));
        }
        index += 1;
    }
    out
}

macro_rules! avx2_scanner_fns {
    ($first:ident, $best:ident, $all:ident, $width:expr) => {
        #[target_feature(enable = "avx2", enable = "popcnt")]
        pub(crate) unsafe fn $first(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<usize> {
            array_first_avx2::<$width>(big_array, small_array, max_dist)
        }

        #[target_feature(enable = "avx2", enable = "popcnt")]
        pub(crate) unsafe fn $best(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<(u64, usize)> {
            array_best_avx2::<$width>(big_array, small_array, max_dist)
        }

        #[target_feature(enable = "avx2", enable = "popcnt")]
        pub(crate) unsafe fn $all(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Vec<(u64, usize)> {
            array_all_avx2::<$width>(big_array, small_array, max_dist)
        }
    };
}

avx2_scanner_fns!(array_first_avx2_8, array_best_avx2_8, array_all_avx2_8, 8);
avx2_scanner_fns!(
    array_first_avx2_16,
    array_best_avx2_16,
    array_all_avx2_16,
    16
);
avx2_scanner_fns!(
    array_first_avx2_32,
    array_best_avx2_32,
    array_all_avx2_32,
    32
);
avx2_scanner_fns!(
    array_first_avx2_64,
    array_best_avx2_64,
    array_all_avx2_64,
    64
);

macro_rules! avx2_dispatch_fns {
    ($first_dispatch:ident, $best_dispatch:ident, $all_dispatch:ident, $first:ident, $best:ident, $all:ident) => {
        #[inline]
        pub(crate) fn $first_dispatch(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<usize> {
            unsafe { $first(big_array, small_array, max_dist) }
        }

        #[inline]
        pub(crate) fn $best_dispatch(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<(u64, usize)> {
            unsafe { $best(big_array, small_array, max_dist) }
        }

        #[inline]
        pub(crate) fn $all_dispatch(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Vec<(u64, usize)> {
            unsafe { $all(big_array, small_array, max_dist) }
        }
    };
}

avx2_dispatch_fns!(
    array_first_avx2_8_dispatch,
    array_best_avx2_8_dispatch,
    array_all_avx2_8_dispatch,
    array_first_avx2_8,
    array_best_avx2_8,
    array_all_avx2_8
);
avx2_dispatch_fns!(
    array_first_avx2_16_dispatch,
    array_best_avx2_16_dispatch,
    array_all_avx2_16_dispatch,
    array_first_avx2_16,
    array_best_avx2_16,
    array_all_avx2_16
);
avx2_dispatch_fns!(
    array_first_avx2_32_dispatch,
    array_best_avx2_32_dispatch,
    array_all_avx2_32_dispatch,
    array_first_avx2_32,
    array_best_avx2_32,
    array_all_avx2_32
);
avx2_dispatch_fns!(
    array_first_avx2_64_dispatch,
    array_best_avx2_64_dispatch,
    array_all_avx2_64_dispatch,
    array_first_avx2_64,
    array_best_avx2_64,
    array_all_avx2_64
);

#[target_feature(enable = "avx2", enable = "popcnt")]
pub(crate) unsafe fn pairwise_avx2_fixed<const WIDTH: usize>(a: &[u8], b: &[u8], out: &mut [u8]) {
    let count = a.len() / WIDTH;
    let mut index = 0usize;
    while index + 8 <= count {
        for lane in 0..8 {
            let i = index + lane;
            let d = avx2_distance_for_width::<WIDTH>(
                a.as_ptr().add(i * WIDTH),
                b.as_ptr().add(i * WIDTH),
            );
            out.as_mut_ptr()
                .add(i * 8)
                .cast::<u64>()
                .write_unaligned(d.to_le());
        }
        index += 8;
    }
    while index < count {
        let d = avx2_distance_for_width::<WIDTH>(
            a.as_ptr().add(index * WIDTH),
            b.as_ptr().add(index * WIDTH),
        );
        out.as_mut_ptr()
            .add(index * 8)
            .cast::<u64>()
            .write_unaligned(d.to_le());
        index += 1;
    }
}

// -----------------------------------------------------------------------------
// AVX-512 VPOPCNTDQ block scanners for fixed-width catalogs (8/16/32/64 B).
//
// Each block of 16 records is XORed with the broadcast query and counted per
// 8-byte lane with VPOPCNTQ. A tree of VPERMT2D even/odd selections and adds
// then folds those partial counts into one ZMM holding the 16 per-record
// distances as u32 lanes in record order, so one unsigned compare yields a
// 16-bit hit mask and blocks without hits need no scalar work. Per 16 records
// the fold costs 1 (8 B), 5 (16 B), 13 (32 B) or 29 (64 B) shuffle/add uops
// on top of one load+XOR and one VPOPCNTQ per 64 bytes. (VPOPCNTQ instead of
// VPOPCNTB + VPSADBW saves one port-5 uop per 64 bytes on Intel.)
//
// Semantics match the other scanners:
//   * `first` returns the lowest matching index.
//   * `best` returns the minimum distance, the lowest index on ties, and stops
//     at an exact match.
//   * `all` returns matches in ascending index order.
//   * `max_dist < 0` means unlimited.
// Records after the last full block use scalar POPCNT.
// -----------------------------------------------------------------------------

#[inline(always)]
fn within_fixed_threshold(distance: u64, max_dist: i64) -> bool {
    max_dist < 0 || distance <= max_dist as u64
}

/// Largest accepted distance: `max_dist`, or any distance when unlimited,
/// capped at the record's bit count so it fits a u32 lane.
#[inline(always)]
fn block_limit<const WIDTH: usize>(max_dist: i64) -> u32 {
    let max_bits = (WIDTH * 8) as u64;
    if max_dist < 0 {
        max_bits as u32
    } else {
        (max_dist as u64).min(max_bits) as u32
    }
}

/// The even u32 lanes of `a` followed by the even u32 lanes of `b`.
#[inline(always)]
unsafe fn even_dwords(a: __m512i, b: __m512i) -> __m512i {
    let idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    _mm512_permutex2var_epi32(a, idx, b)
}

/// `a` and `b` hold consecutive records whose partial sums fill groups of an
/// even number of adjacent u32 lanes. Returns `a`'s records then `b`'s with
/// adjacent partial sums added: half the group size, same record order.
#[inline(always)]
unsafe fn fold_pairs(a: __m512i, b: __m512i) -> __m512i {
    let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    _mm512_add_epi32(even_dwords(a, b), _mm512_permutex2var_epi32(a, odd, b))
}

/// Bit counts of `records ^ query` for 64 bytes, one per 8-byte lane.
#[inline(always)]
unsafe fn lane_counts(records: *const u8, query: __m512i) -> __m512i {
    let x = _mm512_xor_si512(_mm512_loadu_si512(records as *const __m512i), query);
    _mm512_popcnt_epi64(x)
}

/// The query repeated once per record slot of a ZMM.
#[inline(always)]
unsafe fn broadcast_query<const WIDTH: usize>(query: *const u8) -> __m512i {
    match WIDTH {
        8 => _mm512_set1_epi64(core::ptr::read_unaligned(query as *const i64)),
        16 => _mm512_broadcast_i32x4(_mm_loadu_si128(query as *const __m128i)),
        32 => _mm512_broadcast_i64x4(_mm256_loadu_si256(query as *const __m256i)),
        64 => _mm512_loadu_si512(query as *const __m512i),
        _ => core::hint::unreachable_unchecked(),
    }
}

/// The 256 bytes at `records` reduced to 16 u32 lanes of per-record partial
/// sums in record order: `WIDTH / 16` lanes per record for WIDTH >= 16.
#[inline(always)]
unsafe fn fold256(records: *const u8, query: __m512i) -> __m512i {
    fold_pairs(
        even_dwords(
            lane_counts(records, query),
            lane_counts(records.add(64), query),
        ),
        even_dwords(
            lane_counts(records.add(128), query),
            lane_counts(records.add(192), query),
        ),
    )
}

/// Distances of the 16 records at `records`, as u32 lanes in record order.
///
/// SAFETY: `records` must be valid for `16 * WIDTH` readable bytes.
#[inline(always)]
unsafe fn block16_distances<const WIDTH: usize>(records: *const u8, query: __m512i) -> __m512i {
    match WIDTH {
        // Each 8-byte lane is one whole record.
        8 => even_dwords(
            lane_counts(records, query),
            lane_counts(records.add(64), query),
        ),
        16 => fold256(records, query),
        32 => fold_pairs(fold256(records, query), fold256(records.add(256), query)),
        64 => fold_pairs(
            fold_pairs(fold256(records, query), fold256(records.add(256), query)),
            fold_pairs(
                fold256(records.add(512), query),
                fold256(records.add(768), query),
            ),
        ),
        _ => core::hint::unreachable_unchecked(),
    }
}

#[inline(always)]
unsafe fn tail_distance<const WIDTH: usize>(record: *const u8, query: *const u8) -> u32 {
    avx2_distance_for_width::<WIDTH>(record, query) as u32
}

#[inline(always)]
unsafe fn scan_first_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    let count = big_array.len() / WIDTH;
    let records = big_array.as_ptr();
    let query = small_array.as_ptr();
    let limit = block_limit::<WIDTH>(max_dist);
    let query_v = broadcast_query::<WIDTH>(query);
    let limit_v = _mm512_set1_epi32(limit as i32);
    let mut index = 0;
    while index + 16 <= count {
        let distances = block16_distances::<WIDTH>(records.add(index * WIDTH), query_v);
        let hits = _mm512_cmple_epu32_mask(distances, limit_v);
        if hits != 0 {
            return Some(index + hits.trailing_zeros() as usize);
        }
        index += 16;
    }
    while index < count {
        if tail_distance::<WIDTH>(records.add(index * WIDTH), query) <= limit {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[inline(always)]
unsafe fn scan_best_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let records = big_array.as_ptr();
    let query = small_array.as_ptr();
    let query_v = broadcast_query::<WIDTH>(query);
    // After each hit the limit drops to best - 1. Lanes are visited in index
    // order, so equal distances keep the lowest index.
    let mut limit = block_limit::<WIDTH>(max_dist);
    let mut limit_v = _mm512_set1_epi32(limit as i32);
    let mut best = None;
    let mut index = 0;
    while index + 16 <= count {
        let distances = block16_distances::<WIDTH>(records.add(index * WIDTH), query_v);
        let mut hits = _mm512_cmple_epu32_mask(distances, limit_v);
        if hits != 0 {
            let mut lanes = [0u32; 16];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, distances);
            while hits != 0 {
                let lane = hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let distance = lanes[lane];
                if distance <= limit {
                    best = Some((distance as u64, index + lane));
                    if distance == 0 {
                        return best;
                    }
                    limit = distance - 1;
                }
            }
            limit_v = _mm512_set1_epi32(limit as i32);
        }
        index += 16;
    }
    while index < count {
        let distance = tail_distance::<WIDTH>(records.add(index * WIDTH), query);
        if distance <= limit {
            best = Some((distance as u64, index));
            if distance == 0 {
                return best;
            }
            limit = distance - 1;
        }
        index += 1;
    }
    best
}

#[inline(always)]
unsafe fn scan_all_avx512<const WIDTH: usize>(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    let count = big_array.len() / WIDTH;
    let records = big_array.as_ptr();
    let query = small_array.as_ptr();
    let limit = block_limit::<WIDTH>(max_dist);
    let query_v = broadcast_query::<WIDTH>(query);
    let limit_v = _mm512_set1_epi32(limit as i32);
    let mut out = Vec::new();
    let mut index = 0;
    while index + 16 <= count {
        let distances = block16_distances::<WIDTH>(records.add(index * WIDTH), query_v);
        let mut hits = _mm512_cmple_epu32_mask(distances, limit_v);
        if hits != 0 {
            let mut lanes = [0u32; 16];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, distances);
            while hits != 0 {
                let lane = hits.trailing_zeros() as usize;
                hits &= hits - 1;
                out.push((lanes[lane] as u64, index + lane));
            }
        }
        index += 16;
    }
    while index < count {
        let distance = tail_distance::<WIDTH>(records.add(index * WIDTH), query);
        if distance <= limit {
            out.push((distance as u64, index));
        }
        index += 1;
    }
    out
}

// -----------------------------------------------------------------------------
// Public (crate-visible) scanner entry points. `select_array_scanner_for_width`
// captures these as function pointers after checking the CPU features.
// -----------------------------------------------------------------------------

macro_rules! avx512_block_scanners {
    ($width:literal, $first:ident, $best:ident, $all:ident) => {
        /// # Safety
        /// The CPU must support AVX-512 F/VPOPCNTDQ and POPCNT, `small_array`
        /// must be one record long and `big_array` a whole number of records.
        #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq", enable = "popcnt")]
        pub(crate) unsafe fn $first(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<usize> {
            scan_first_avx512::<$width>(big_array, small_array, max_dist)
        }

        /// # Safety
        /// Same as the matching `array_first_avx512_*`.
        #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq", enable = "popcnt")]
        pub(crate) unsafe fn $best(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Option<(u64, usize)> {
            scan_best_avx512::<$width>(big_array, small_array, max_dist)
        }

        /// # Safety
        /// Same as the matching `array_first_avx512_*`.
        #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq", enable = "popcnt")]
        pub(crate) unsafe fn $all(
            big_array: &[u8],
            small_array: &[u8],
            max_dist: i64,
        ) -> Vec<(u64, usize)> {
            scan_all_avx512::<$width>(big_array, small_array, max_dist)
        }
    };
}

avx512_block_scanners!(
    8,
    array_first_avx512_8,
    array_best_avx512_8,
    array_all_avx512_8
);
avx512_block_scanners!(
    16,
    array_first_avx512_16,
    array_best_avx512_16,
    array_all_avx512_16
);
avx512_block_scanners!(
    32,
    array_first_avx512_32,
    array_best_avx512_32,
    array_all_avx512_32
);
avx512_block_scanners!(
    64,
    array_first_avx512_64,
    array_best_avx512_64,
    array_all_avx512_64
);

// -----------------------------------------------------------------------------
// Feature-checked trampolines used by the `ArrayScanner` function-pointer table
// in `api.rs`. Function pointers cannot carry `#[target_feature]`, so these
// safe wrappers re-check the feature at every call (cheap after the first
// invocation because `is_x86_feature_detected!` caches the result).
// -----------------------------------------------------------------------------

#[inline]
pub(crate) fn array_first_avx512_8_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    unsafe { array_first_avx512_8(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_best_avx512_8_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    unsafe { array_best_avx512_8(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_all_avx512_8_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    unsafe { array_all_avx512_8(big_array, small_array, max_dist) }
}

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

#[inline]
pub(crate) fn array_first_avx512_64_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<usize> {
    unsafe { array_first_avx512_64(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_best_avx512_64_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    unsafe { array_best_avx512_64(big_array, small_array, max_dist) }
}

#[inline]
pub(crate) fn array_all_avx512_64_dispatch(
    big_array: &[u8],
    small_array: &[u8],
    max_dist: i64,
) -> Vec<(u64, usize)> {
    unsafe { array_all_avx512_64(big_array, small_array, max_dist) }
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

    #[target_feature(enable = "avx512bw", enable = "avx512bitalg", enable = "popcnt")]
    unsafe fn hamming_distance_bytes_avx512_reference(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
        let mut difference = 0u64;
        let max_dist_u64 = max_dist as u64;
        for (x, y) in a.iter().zip(b.iter()) {
            difference += (x ^ y).count_ones() as u64;
            if max_dist >= 0 && difference > max_dist_u64 {
                return u64::MAX;
            }
        }
        difference
    }

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

    #[test]
    fn avx512_small_byte_shortcut_matches_reference() {
        if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512bitalg") {
            return;
        }
        for length in 0usize..40 {
            let mut rng = SplitMix::new(0xB17E_512 ^ length as u64);
            let a = rng.vec(length);
            let b = rng.vec(length);
            for max_dist in [-1i64, 0, 1, 7, 8, 31, 32, 1000] {
                let want = unsafe { hamming_distance_bytes_avx512_reference(&a, &b, max_dist) };
                let got = unsafe { hamming_distance_bytes_avx512(&a, &b, max_dist) };
                assert_eq!(got, want, "len={length} max_dist={max_dist}");
            }
        }
    }

    fn avx512_scanner_hw_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("popcnt")
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

    #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
    unsafe fn block16_distances_array<const WIDTH: usize>(
        records: &[u8],
        query: &[u8],
    ) -> [u32; 16] {
        let distances =
            block16_distances::<WIDTH>(records.as_ptr(), broadcast_query::<WIDTH>(query.as_ptr()));
        let mut lanes = [0u32; 16];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, distances);
        lanes
    }

    // The fold tree must yield every record's distance in record order,
    // including the extremes (exact match and bitwise complement).
    #[test]
    fn avx512_block16_distances_match_scalar() {
        fn check<const WIDTH: usize>(rng: &mut SplitMix) {
            let query = rng.vec(WIDTH);
            let mut records = rng.vec(16 * WIDTH);
            records[3 * WIDTH..4 * WIDTH].copy_from_slice(&query);
            for (dst, &q) in records[9 * WIDTH..10 * WIDTH].iter_mut().zip(&query) {
                *dst = !q;
            }
            let got = unsafe { block16_distances_array::<WIDTH>(&records, &query) };
            for (i, &distance) in got.iter().enumerate() {
                let want = scalar_byte_distance(&records[i * WIDTH..(i + 1) * WIDTH], &query);
                assert_eq!(distance as u64, want, "width={WIDTH} record={i}");
            }
        }
        if !avx512_scanner_hw_available() {
            return;
        }
        let mut rng = SplitMix::new(0xC0FFEE_D15EA5E);
        for _ in 0..16 {
            check::<8>(&mut rng);
            check::<16>(&mut rng);
            check::<32>(&mut rng);
            check::<64>(&mut rng);
        }
    }

    // Full-scanner semantic parity for widths 8/16/32/64 across a matrix of
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

        let w = width as i64;
        for &max_dist in &[
            -1i64,
            0,
            3,
            4,
            5,
            8,
            128,
            4 * w - 8,
            4 * w,
            8 * w - 1,
            8 * w,
            8 * w + 1,
        ] {
            let (efirst, ebest, eall) = oracle_first_best_all(&big, &small, max_dist);

            let (afirst, abest, aall) = match width {
                8 => unsafe {
                    (
                        array_first_avx512_8(&big, &small, max_dist),
                        array_best_avx512_8(&big, &small, max_dist),
                        array_all_avx512_8(&big, &small, max_dist),
                    )
                },
                16 => unsafe {
                    (
                        array_first_avx512_16(&big, &small, max_dist),
                        array_best_avx512_16(&big, &small, max_dist),
                        array_all_avx512_16(&big, &small, max_dist),
                    )
                },
                32 => unsafe {
                    (
                        array_first_avx512_32(&big, &small, max_dist),
                        array_best_avx512_32(&big, &small, max_dist),
                        array_all_avx512_32(&big, &small, max_dist),
                    )
                },
                64 => unsafe {
                    (
                        array_first_avx512_64(&big, &small, max_dist),
                        array_best_avx512_64(&big, &small, max_dist),
                        array_all_avx512_64(&big, &small, max_dist),
                    )
                },
                _ => unreachable!(),
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

    #[test]
    fn avx512_scanners_w8_w64_random_oracle_various_counts() {
        if !avx512_scanner_hw_available() {
            return;
        }
        for &width in &[8usize, 64] {
            for &count in &[1usize, 3, 15, 16, 17, 33, 70, 257] {
                assert_scanners_match_oracle(
                    width,
                    count,
                    0x5150_0000 ^ width as u64 ^ count as u64,
                );
            }
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

    fn avx2_hw_available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt")
    }

    fn assert_avx2_scanners_match_oracle(width: usize, count: usize, seed: u64) {
        let mut rng = SplitMix::new(seed);
        let small = rng.vec(width);
        let mut big = rng.vec(count * width);
        if count > 3 {
            big[width..2 * width].copy_from_slice(&small);
            big[(count - 1) * width..count * width].copy_from_slice(&small);
        }
        for &max_dist in &[-1i64, 0, 1, 7, 32, 64, 128, 256, 512] {
            let (efirst, ebest, eall) = oracle_first_best_all(&big, &small, max_dist);
            let (afirst, abest, aall) = unsafe {
                match width {
                    8 => (
                        array_first_avx2_8(&big, &small, max_dist),
                        array_best_avx2_8(&big, &small, max_dist),
                        array_all_avx2_8(&big, &small, max_dist),
                    ),
                    16 => (
                        array_first_avx2_16(&big, &small, max_dist),
                        array_best_avx2_16(&big, &small, max_dist),
                        array_all_avx2_16(&big, &small, max_dist),
                    ),
                    32 => (
                        array_first_avx2_32(&big, &small, max_dist),
                        array_best_avx2_32(&big, &small, max_dist),
                        array_all_avx2_32(&big, &small, max_dist),
                    ),
                    64 => (
                        array_first_avx2_64(&big, &small, max_dist),
                        array_best_avx2_64(&big, &small, max_dist),
                        array_all_avx2_64(&big, &small, max_dist),
                    ),
                    _ => unreachable!(),
                }
            };
            assert_eq!(
                afirst, efirst,
                "first width={width} count={count} max={max_dist}"
            );
            assert_eq!(
                abest, ebest,
                "best width={width} count={count} max={max_dist}"
            );
            assert_eq!(aall, eall, "all width={width} count={count} max={max_dist}");
        }
    }

    #[test]
    fn avx2_scanners_random_oracle_various_widths_counts() {
        if !avx2_hw_available() {
            return;
        }
        for &width in &[8usize, 16, 32, 64] {
            for &count in &[0usize, 1, 2, 7, 8, 9, 15, 16, 17, 33, 128] {
                assert_avx2_scanners_match_oracle(
                    width,
                    count,
                    0xA5A5_1234 ^ count as u64 ^ width as u64,
                );
            }
        }
    }

    #[test]
    fn avx2_pairwise_fixed_widths_match_scalar() {
        if !avx2_hw_available() {
            return;
        }
        for &width in &[8usize, 16, 32, 64] {
            for &count in &[0usize, 1, 7, 8, 9, 31] {
                let mut rng = SplitMix::new(0x55AA_7788 ^ width as u64 ^ count as u64);
                let a = rng.vec(width * count);
                let b = rng.vec(width * count);
                let mut out = vec![0u8; count * 8];
                unsafe {
                    match width {
                        8 => pairwise_avx2_fixed::<8>(&a, &b, &mut out),
                        16 => pairwise_avx2_fixed::<16>(&a, &b, &mut out),
                        32 => pairwise_avx2_fixed::<32>(&a, &b, &mut out),
                        64 => pairwise_avx2_fixed::<64>(&a, &b, &mut out),
                        _ => unreachable!(),
                    }
                }
                for i in 0..count {
                    let got = u64::from_le_bytes(out[i * 8..(i + 1) * 8].try_into().unwrap());
                    let want = scalar_byte_distance(
                        &a[i * width..(i + 1) * width],
                        &b[i * width..(i + 1) * width],
                    );
                    assert_eq!(got, want, "pairwise width={width} count={count} index={i}");
                }
            }
        }
    }

    #[test]
    fn avx512_vbmi_hex_matches_legacy_kernel() {
        if !(is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512bitalg")
            && is_x86_feature_detected!("avx512vbmi"))
        {
            return;
        }
        let invalids = [b'g', b'G', b'z', b'!', b'/', b':', b'@', 0x80, 0xFF];
        for len in 0usize..=600 {
            let mut rng = SplitMix::new(0xA11CE ^ len as u64);
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            const HEX: &[u8] = b"0123456789abcdefABCDEF";
            for i in 0..len {
                a[i] = HEX[rng.next() as usize % HEX.len()];
                b[i] = HEX[rng.next() as usize % HEX.len()];
            }
            for inject in [None, Some(len / 2), len.checked_sub(1)] {
                let mut aa = a.clone();
                if let Some(pos) = inject {
                    if pos < len {
                        aa[pos] = invalids[(len + pos) % invalids.len()];
                    }
                }
                let want = unsafe { hamming_distance_string_avx512(&aa, &b) };
                let got = unsafe { hamming_distance_string_avx512_vbmi(&aa, &b) };
                assert_eq!(got, want, "len={len} inject={inject:?}");
            }
        }
    }

    #[test]
    fn public_api_native_dispatch_matches_oracles() {
        let _ = crate::set_algorithm("native");
        for &width in &[1usize, 2, 7, 8, 15, 16, 31, 32, 63, 64, 65, 128] {
            for &count in &[0usize, 1, 2, 15, 16, 17, 31, 32, 33, 70, 257] {
                let mut rng = SplitMix::new(0xD15A_7000 ^ width as u64 ^ ((count as u64) << 16));
                let mut catalog = rng.vec(width * count);
                let query = rng.vec(width);
                if count > 2 {
                    catalog[width..2 * width].copy_from_slice(&query);
                }
                let limits = [
                    -1i64,
                    0,
                    1,
                    (width * 4) as i64,
                    (width * 8) as i64,
                    (width * 8 + 1) as i64,
                ];
                for max_dist in limits {
                    let (first, best, all) = oracle_first_best_all(&catalog, &query, max_dist);
                    assert_eq!(
                        crate::bytes_array_first_within_dist(&catalog, &query, max_dist).unwrap(),
                        first,
                        "first width={width} count={count} max={max_dist}"
                    );
                    assert_eq!(
                        crate::bytes_array_best_within_dist(&catalog, &query, max_dist).unwrap(),
                        best,
                        "best width={width} count={count} max={max_dist}"
                    );
                    assert_eq!(
                        crate::bytes_array_all_within_dist(&catalog, &query, max_dist).unwrap(),
                        all,
                        "all width={width} count={count} max={max_dist}"
                    );
                }

                let a = rng.vec(width * count);
                let b = rng.vec(width * count);
                let pairwise = crate::bytes_pairwise_distances(&a, &b, width).unwrap();
                let mut into = vec![0u8; count * 8];
                assert_eq!(
                    crate::bytes_pairwise_distances_into(&a, &b, width, &mut into).unwrap(),
                    count
                );
                for i in 0..count {
                    let want = scalar_byte_distance(
                        &a[i * width..(i + 1) * width],
                        &b[i * width..(i + 1) * width],
                    );
                    assert_eq!(
                        pairwise[i], want,
                        "pairwise width={width} count={count} i={i}"
                    );
                    assert_eq!(
                        u64::from_le_bytes(into[i * 8..(i + 1) * 8].try_into().unwrap()),
                        want
                    );
                }
            }

            for len in 0usize..=128 {
                let mut a = String::with_capacity(len);
                let mut b = String::with_capacity(len);
                const HEX: &[u8] = b"0123456789abcdef";
                for i in 0..len {
                    a.push(HEX[i % HEX.len()] as char);
                    b.push(HEX[(i * 7 + 3) % HEX.len()] as char);
                }
                let want = hamming_distance_string_classic(a.as_bytes(), b.as_bytes());
                assert_eq!(crate::hex_hamming_distance(&a, &b), want);
            }
            let _ = crate::set_algorithm("native");
        }
    }
}
