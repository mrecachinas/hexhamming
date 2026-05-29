// ARM64 NEON implementations

use crate::classic::hamming_distance_string_classic;
use crate::hex::hex_char_to_nibble;
use crate::native::hamming_distance_bytes_native;
use crate::LOOKUP;

use std::arch::aarch64::*;

/// NEON vectorized hamming distance for byte arrays.
/// Processes 64 B per iter via 4× vld1q_u8(veorq)+vcntq_u8, accumulating
/// into a uint8x16_t accumulator for up to 7 iterations (448 B) before
/// one horizontal sum via vaddlvq_u8.  (Each iter adds up to 4×8=32 per
/// lane; 7×32=224 < 255.)  Handles max_dist>=0 early-exit per §2.
#[inline]
pub(crate) unsafe fn hamming_distance_bytes_neon(a: &[u8], b: &[u8], max_dist: i64) -> u64 {
    let length = a.len();

    // For small inputs, delegate to native (SIMD setup not worthwhile)
    if length < 32 {
        return hamming_distance_bytes_native(a, b, max_dist);
    }

    let mut i = 0usize;
    let mut difference: u64 = 0;
    let zero = vdupq_n_u8(0);

    // Max safe inner iterations: 255 / (4*8) = 7  (7*32 = 224 < 255)
    const BATCH: usize = 7;

    if max_dist < 0 {
        // Full distance — batch BATCH iterations of 64 B per horizontal sum
        while i + 64 * BATCH <= length {
            let mut acc = zero;
            for _ in 0..BATCH {
                let a0 = vld1q_u8(a.as_ptr().add(i));
                let b0 = vld1q_u8(b.as_ptr().add(i));
                let a1 = vld1q_u8(a.as_ptr().add(i + 16));
                let b1 = vld1q_u8(b.as_ptr().add(i + 16));
                let a2 = vld1q_u8(a.as_ptr().add(i + 32));
                let b2 = vld1q_u8(b.as_ptr().add(i + 32));
                let a3 = vld1q_u8(a.as_ptr().add(i + 48));
                let b3 = vld1q_u8(b.as_ptr().add(i + 48));

                let cnt0 = vcntq_u8(veorq_u8(a0, b0));
                let cnt1 = vcntq_u8(veorq_u8(a1, b1));
                let cnt2 = vcntq_u8(veorq_u8(a2, b2));
                let cnt3 = vcntq_u8(veorq_u8(a3, b3));

                acc = vaddq_u8(acc, vaddq_u8(vaddq_u8(cnt0, cnt1), vaddq_u8(cnt2, cnt3)));
                i += 64;
            }
            difference += vaddlvq_u8(acc) as u64;
        }

        // Remaining 64-byte chunks (up to BATCH-1 iterations safe for u8 acc)
        let mut acc = zero;
        while i + 64 <= length {
            let a0 = vld1q_u8(a.as_ptr().add(i));
            let b0 = vld1q_u8(b.as_ptr().add(i));
            let a1 = vld1q_u8(a.as_ptr().add(i + 16));
            let b1 = vld1q_u8(b.as_ptr().add(i + 16));
            let a2 = vld1q_u8(a.as_ptr().add(i + 32));
            let b2 = vld1q_u8(b.as_ptr().add(i + 32));
            let a3 = vld1q_u8(a.as_ptr().add(i + 48));
            let b3 = vld1q_u8(b.as_ptr().add(i + 48));

            let cnt0 = vcntq_u8(veorq_u8(a0, b0));
            let cnt1 = vcntq_u8(veorq_u8(a1, b1));
            let cnt2 = vcntq_u8(veorq_u8(a2, b2));
            let cnt3 = vcntq_u8(veorq_u8(a3, b3));

            acc = vaddq_u8(acc, vaddq_u8(vaddq_u8(cnt0, cnt1), vaddq_u8(cnt2, cnt3)));
            i += 64;
        }
        difference += vaddlvq_u8(acc) as u64;

        // 16-byte chunks
        while i + 16 <= length {
            let a16 = vld1q_u8(a.as_ptr().add(i));
            let b16 = vld1q_u8(b.as_ptr().add(i));
            difference += vaddlvq_u8(vcntq_u8(veorq_u8(a16, b16))) as u64;
            i += 16;
        }

        // Scalar tail
        while i < length {
            difference += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
            i += 1;
        }
        difference
    } else {
        // Early-exit path — check every BATCH iters of 64 B (448 B)
        let max_dist_u64 = max_dist as u64;

        while i + 64 * BATCH <= length {
            let mut acc = zero;
            for _ in 0..BATCH {
                let a0 = vld1q_u8(a.as_ptr().add(i));
                let b0 = vld1q_u8(b.as_ptr().add(i));
                let a1 = vld1q_u8(a.as_ptr().add(i + 16));
                let b1 = vld1q_u8(b.as_ptr().add(i + 16));
                let a2 = vld1q_u8(a.as_ptr().add(i + 32));
                let b2 = vld1q_u8(b.as_ptr().add(i + 32));
                let a3 = vld1q_u8(a.as_ptr().add(i + 48));
                let b3 = vld1q_u8(b.as_ptr().add(i + 48));

                let cnt0 = vcntq_u8(veorq_u8(a0, b0));
                let cnt1 = vcntq_u8(veorq_u8(a1, b1));
                let cnt2 = vcntq_u8(veorq_u8(a2, b2));
                let cnt3 = vcntq_u8(veorq_u8(a3, b3));

                acc = vaddq_u8(acc, vaddq_u8(vaddq_u8(cnt0, cnt1), vaddq_u8(cnt2, cnt3)));
                i += 64;
            }
            difference += vaddlvq_u8(acc) as u64;
            if difference > max_dist_u64 {
                return u64::MAX;
            }
        }

        // Remaining 64-byte chunks
        let mut acc = zero;
        while i + 64 <= length {
            let a0 = vld1q_u8(a.as_ptr().add(i));
            let b0 = vld1q_u8(b.as_ptr().add(i));
            let a1 = vld1q_u8(a.as_ptr().add(i + 16));
            let b1 = vld1q_u8(b.as_ptr().add(i + 16));
            let a2 = vld1q_u8(a.as_ptr().add(i + 32));
            let b2 = vld1q_u8(b.as_ptr().add(i + 32));
            let a3 = vld1q_u8(a.as_ptr().add(i + 48));
            let b3 = vld1q_u8(b.as_ptr().add(i + 48));

            let cnt0 = vcntq_u8(veorq_u8(a0, b0));
            let cnt1 = vcntq_u8(veorq_u8(a1, b1));
            let cnt2 = vcntq_u8(veorq_u8(a2, b2));
            let cnt3 = vcntq_u8(veorq_u8(a3, b3));

            acc = vaddq_u8(acc, vaddq_u8(vaddq_u8(cnt0, cnt1), vaddq_u8(cnt2, cnt3)));
            i += 64;
        }
        difference += vaddlvq_u8(acc) as u64;

        // 16-byte chunks
        while i + 16 <= length {
            let a16 = vld1q_u8(a.as_ptr().add(i));
            let b16 = vld1q_u8(b.as_ptr().add(i));
            difference += vaddlvq_u8(vcntq_u8(veorq_u8(a16, b16))) as u64;
            i += 16;
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

/// NEON vectorized hamming distance for hex strings.
/// Processes 16 ASCII hex chars per iteration using:
///   - vqtbl1q_u8 for branchless hex→nibble conversion
///   - vcntq_u8 for parallel popcount
///   - vpaddlq cascade for horizontal summation
#[inline]
pub unsafe fn hamming_distance_string_neon(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 16 {
        return hamming_distance_string_classic(a, b);
    }

    // Hex→nibble lookup table for vqtbl1q_u8 (indices 0-15 map ASCII
    // low nibble to hex value; out-of-range produces 0xFF via saturation).
    // We split into two ranges: digits ('0'-'9') and letters ('A'-'F'/'a'-'f').
    //
    // Strategy: mask to low nibble, use vqtbl1q as a 16-entry LUT.
    // '0'(0x30)..'9'(0x39) have low nibbles 0x0..0x9 → identity
    // 'A'(0x41)..'F'(0x46) have low nibbles 0x1..0x6 → +9
    // 'a'(0x61)..'f'(0x66) have low nibbles 0x1..0x6 → +9
    // We detect digit vs letter via range comparison.

    let zero = vdupq_n_u8(0);
    let fifteen_u = vdupq_n_u8(15);
    let case_mask = vdupq_n_u8(0xDF); // clears bit 5 for case folding
    let ascii_0 = vdupq_n_u8(b'0');
    let seven = vdupq_n_u8(7);
    let nine = vdupq_n_u8(9);
    let ten = vdupq_n_u8(10);

    // Popcount lookup table: popcnt[i] = number of 1-bits in i, for i in 0..15
    let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());

    let mut i = 0usize;
    let mut total = vdupq_n_u64(0);

    // Process 64 chars at a time (4×16) to batch horizontal sums.
    // Per-byte accumulator holds at most 4*4 = 16 < 255, safe for u8 lanes.
    while i + 64 <= length {
        let mut acc = zero;

        for _ in 0..4 {
            let a16 = vld1q_u8(a.as_ptr().add(i));
            let b16 = vld1q_u8(b.as_ptr().add(i));

            let a_nib = hex_parse_neon(a16, case_mask, ascii_0, seven, nine, ten);
            let b_nib = hex_parse_neon(b16, case_mask, ascii_0, seven, nine, ten);

            // Validate: any lane > 15 means invalid char — single cmpgt(or(a,b), 15)
            let bad = vcgtq_u8(vorrq_u8(a_nib, b_nib), fifteen_u);
            if vmaxvq_u8(bad) != 0 {
                return Err("hex string contains invalid char");
            }

            // XOR nibbles → popcount via table lookup (values are 0-15, only low nibble used)
            let xor = veorq_u8(a_nib, b_nib);
            let cnt = vqtbl1q_u8(popcnt_tbl, xor);
            acc = vaddq_u8(acc, cnt);

            i += 16;
        }

        // Horizontal sum: u8→u16→u32→u64, add into total
        total = vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(acc)));
    }

    // Process remaining 16-byte chunks
    let mut acc = zero;
    while i + 16 <= length {
        let a16 = vld1q_u8(a.as_ptr().add(i));
        let b16 = vld1q_u8(b.as_ptr().add(i));

        let a_nib = hex_parse_neon(a16, case_mask, ascii_0, seven, nine, ten);
        let b_nib = hex_parse_neon(b16, case_mask, ascii_0, seven, nine, ten);

        let bad = vcgtq_u8(vorrq_u8(a_nib, b_nib), fifteen_u);
        if vmaxvq_u8(bad) != 0 {
            return Err("hex string contains invalid char");
        }

        let xor = veorq_u8(a_nib, b_nib);
        let cnt = vqtbl1q_u8(popcnt_tbl, xor);
        acc = vaddq_u8(acc, cnt);

        i += 16;
    }
    total = vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(acc)));

    let mut result = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);

    // Scalar tail for remaining chars
    while i < length {
        let val1 = hex_char_to_nibble(*a.get_unchecked(i));
        let val2 = hex_char_to_nibble(*b.get_unchecked(i));
        if (val1 | val2) & 0xF0 != 0 {
            return Err("hex string contains invalid char");
        }
        result += *LOOKUP.get_unchecked((val1 ^ val2) as usize) as u64;
        i += 1;
    }

    Ok(result)
}

/// Branchless vectorized hex ASCII → nibble (0-15) conversion.
/// Invalid chars produce values > 15 (for easy detection by caller).
///
/// Strategy (7 NEON instructions):
///   1. digit_val = c - '0': digits become 0-9
///   2. letter_val = (c & 0xDF) - '0' - 7: letters become 10-15
///   3. Select letter path where digit_val > 9
///   4. Force invalid where letter result < 10 (catches '@', '`')
#[inline(always)]
unsafe fn hex_parse_neon(
    chars: uint8x16_t,
    case_mask: uint8x16_t,
    ascii_0: uint8x16_t,
    seven: uint8x16_t,
    nine: uint8x16_t,
    ten: uint8x16_t,
) -> uint8x16_t {
    let digit_val = vsubq_u8(chars, ascii_0);
    let letter_val = vsubq_u8(vandq_u8(chars, case_mask), ascii_0);
    let is_letter = vcgtq_u8(digit_val, nine);
    let adjusted = vsubq_u8(letter_val, seven);
    let result = vbslq_u8(is_letter, adjusted, digit_val);
    // Force lanes invalid where letter path produced < 10 (e.g. '@' → 9)
    let bad_letter = vandq_u8(is_letter, vcltq_u8(adjusted, ten));
    vorrq_u8(result, bad_letter)
}

/// Parse 32 hex chars at `a`/`b`, XOR their nibbles, and pack the 32 nibble-XOR
/// results into 16 bytes (each byte holds two XOR'd nibbles).
///
/// Returns `(packed, bad)` where `packed` is ready for `vcntq_u8` popcount and
/// `bad` is a per-lane mask (non-zero lane ⇒ an invalid hex char was seen).
/// The caller is responsible for accumulating `bad` and validating once.
///
/// SAFETY: `a` and `b` must each be valid for 32 readable bytes.
#[inline(always)]
unsafe fn pack32_xor_neon(
    a: *const u8,
    b: *const u8,
    case_mask: uint8x16_t,
    ascii_0: uint8x16_t,
    seven: uint8x16_t,
    nine: uint8x16_t,
    ten: uint8x16_t,
    fifteen_u: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let a_lo = hex_parse_neon(vld1q_u8(a), case_mask, ascii_0, seven, nine, ten);
    let b_lo = hex_parse_neon(vld1q_u8(b), case_mask, ascii_0, seven, nine, ten);
    let a_hi = hex_parse_neon(vld1q_u8(a.add(16)), case_mask, ascii_0, seven, nine, ten);
    let b_hi = hex_parse_neon(vld1q_u8(b.add(16)), case_mask, ascii_0, seven, nine, ten);

    // Per-lane invalid mask — caller accumulates and checks once.
    let bad = vorrq_u8(
        vcgtq_u8(vorrq_u8(a_lo, b_lo), fifteen_u),
        vcgtq_u8(vorrq_u8(a_hi, b_hi), fifteen_u),
    );

    // XOR nibbles, then pack even/odd nibbles into bytes.
    let xor_lo = veorq_u8(a_lo, b_lo);
    let xor_hi = veorq_u8(a_hi, b_hi);
    let even = vuzp1q_u8(xor_lo, xor_hi);
    let odd = vuzp2q_u8(xor_lo, xor_hi);
    // Constant left-shift by 4 (nibbles are 0-15, so logical shift is correct).
    let packed = vorrq_u8(vshlq_n_u8(even, 4), odd);
    (packed, bad)
}
/// Alternative: parse 32 hex chars → pack into 16 bytes → use vcntq_u8.
/// Processes 32 hex chars per iteration (vs 16 in the nibble approach),
/// and replaces the vqtbl1q popcount lookup with the native vcntq_u8
/// instruction which counts all 8 bits per byte in a single cycle.
#[inline]
pub unsafe fn hamming_distance_string_neon_pack(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 32 {
        return hamming_distance_string_neon(a, b);
    }

    let fifteen_u = vdupq_n_u8(15);
    let case_mask = vdupq_n_u8(0xDF);
    let ascii_0 = vdupq_n_u8(b'0');
    let seven = vdupq_n_u8(7);
    let nine = vdupq_n_u8(9);
    let ten = vdupq_n_u8(10);
    let zero = vdupq_n_u8(0);

    let mut i = 0usize;
    let mut difference: u64 = 0;
    // Accumulate invalid-char masks across the whole SIMD region and check once.
    let mut bad_acc = zero;

    // Each 32-char iteration popcounts into a u8 lane (≤8 per packed byte), so
    // up to 31 iterations are safe before a lane overflows (31*8=248 < 256).
    // Batch BATCH iterations per horizontal reduction to keep the hot loop free
    // of cross-lane reductions.
    const BATCH: usize = 16;

    while i + 32 * BATCH <= length {
        let mut acc = zero;
        for _ in 0..BATCH {
            let (packed, bad) = pack32_xor_neon(
                a.as_ptr().add(i),
                b.as_ptr().add(i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
                fifteen_u,
            );
            bad_acc = vorrq_u8(bad_acc, bad);
            acc = vaddq_u8(acc, vcntq_u8(packed));
            i += 32;
        }
        difference += vaddlvq_u8(acc) as u64;
    }

    // Remaining 32-char chunks (< BATCH of them — still safe in u8 lanes).
    let mut acc = zero;
    while i + 32 <= length {
        let (packed, bad) = pack32_xor_neon(
            a.as_ptr().add(i),
            b.as_ptr().add(i),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
            fifteen_u,
        );
        bad_acc = vorrq_u8(bad_acc, bad);
        acc = vaddq_u8(acc, vcntq_u8(packed));
        i += 32;
    }
    difference += vaddlvq_u8(acc) as u64;

    // Single validation for the entire 32-char SIMD region.
    if vmaxvq_u8(bad_acc) != 0 {
        return Err("hex string contains invalid char");
    }

    // Handle remaining chars with the nibble-based approach
    // §13: popcnt_tbl loaded once at function entry scope, not per iteration
    let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());
    while i + 16 <= length {
        let a_nib = hex_parse_neon(
            vld1q_u8(a.as_ptr().add(i)),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_nib = hex_parse_neon(
            vld1q_u8(b.as_ptr().add(i)),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        // §8: single cmpgt(or(a,b), 15)
        let bad = vcgtq_u8(vorrq_u8(a_nib, b_nib), fifteen_u);
        if vmaxvq_u8(bad) != 0 {
            return Err("hex string contains invalid char");
        }
        let xor = veorq_u8(a_nib, b_nib);
        let cnt = vqtbl1q_u8(popcnt_tbl, xor);
        difference += vaddlvq_u8(cnt) as u64;
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

    Ok(difference)
}

/// Like hamming_distance_string_neon_pack, but with early-exit at max_dist.
/// Returns Ok(u64::MAX) when distance exceeds max_dist (caller treats as "not within").
#[inline]
pub unsafe fn hamming_distance_string_neon_pack_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();

    if length < 32 {
        // Fall back to scalar with early-exit for short inputs
        return hamming_distance_string_neon_with_max(a, b, max_dist);
    }

    let fifteen_u = vdupq_n_u8(15);
    let case_mask = vdupq_n_u8(0xDF);
    let ascii_0 = vdupq_n_u8(b'0');
    let seven = vdupq_n_u8(7);
    let nine = vdupq_n_u8(9);
    let ten = vdupq_n_u8(10);
    let zero = vdupq_n_u8(0);

    let mut i = 0usize;
    let mut difference: u64 = 0;
    let mut bad_acc = zero;

    // Process 32 hex chars at a time, batching cross-lane reductions and the
    // threshold check. Each packed byte popcount is ≤8, so BATCH iterations are
    // safe in u8 lanes (BATCH*8 < 256). Early exit is granular to one batch.
    const BATCH: usize = 16;
    while i + 32 <= length {
        let mut acc = zero;
        let mut n = 0;
        while n < BATCH && i + 32 <= length {
            let (packed, bad) = pack32_xor_neon(
                a.as_ptr().add(i),
                b.as_ptr().add(i),
                case_mask,
                ascii_0,
                seven,
                nine,
                ten,
                fifteen_u,
            );
            bad_acc = vorrq_u8(bad_acc, bad);
            acc = vaddq_u8(acc, vcntq_u8(packed));
            i += 32;
            n += 1;
        }
        // Invalid hex chars take precedence over the max_dist sentinel.
        if vmaxvq_u8(bad_acc) != 0 {
            return Err("hex string contains invalid char");
        }
        difference += vaddlvq_u8(acc) as u64;
        if difference > max_dist {
            return Ok(u64::MAX);
        }
    }

    // Handle remaining chars with the nibble-based approach
    let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());
    while i + 16 <= length {
        let a_nib = hex_parse_neon(
            vld1q_u8(a.as_ptr().add(i)),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let b_nib = hex_parse_neon(
            vld1q_u8(b.as_ptr().add(i)),
            case_mask,
            ascii_0,
            seven,
            nine,
            ten,
        );
        let bad = vcgtq_u8(vorrq_u8(a_nib, b_nib), fifteen_u);
        if vmaxvq_u8(bad) != 0 {
            return Err("hex string contains invalid char");
        }
        let xor = veorq_u8(a_nib, b_nib);
        let cnt = vqtbl1q_u8(popcnt_tbl, xor);
        difference += vaddlvq_u8(cnt) as u64;
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

/// Like hamming_distance_string_neon, but with early-exit at max_dist.
/// Used as fallback for inputs < 32 chars.
#[inline]
unsafe fn hamming_distance_string_neon_with_max(
    a: &[u8],
    b: &[u8],
    max_dist: u64,
) -> Result<u64, &'static str> {
    let length = a.len();

    let fifteen_u = vdupq_n_u8(15);
    let case_mask = vdupq_n_u8(0xDF);
    let ascii_0 = vdupq_n_u8(b'0');
    let seven = vdupq_n_u8(7);
    let nine = vdupq_n_u8(9);
    let ten = vdupq_n_u8(10);
    let popcnt_tbl = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());

    let mut i = 0usize;
    let mut difference: u64 = 0;

    while i + 16 <= length {
        let a16 = vld1q_u8(a.as_ptr().add(i));
        let b16 = vld1q_u8(b.as_ptr().add(i));

        let a_nib = hex_parse_neon(a16, case_mask, ascii_0, seven, nine, ten);
        let b_nib = hex_parse_neon(b16, case_mask, ascii_0, seven, nine, ten);

        let bad = vcgtq_u8(vorrq_u8(a_nib, b_nib), fifteen_u);
        if vmaxvq_u8(bad) != 0 {
            return Err("hex string contains invalid char");
        }

        let xor = veorq_u8(a_nib, b_nib);
        let cnt = vqtbl1q_u8(popcnt_tbl, xor);
        difference += vaddlvq_u8(cnt) as u64;

        if difference > max_dist {
            return Ok(u64::MAX);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neon_string_basic() {
        unsafe {
            assert_eq!(
                hamming_distance_string_neon(b"deadbeef", b"00000000").unwrap(),
                24
            );
            assert_eq!(hamming_distance_string_neon(b"ffff", b"0000").unwrap(), 16);
            assert_eq!(hamming_distance_string_neon(b"0000", b"0000").unwrap(), 0);
        }
    }

    #[test]
    fn neon_string_16_chars() {
        // Exactly 16 chars — one NEON iteration, no tail
        unsafe {
            let a = "f".repeat(16);
            let b = "0".repeat(16);
            assert_eq!(
                hamming_distance_string_neon(a.as_bytes(), b.as_bytes()).unwrap(),
                64
            );
        }
    }

    #[test]
    fn neon_string_64_chars() {
        // 64 chars — exercises the batched 4×16 loop
        unsafe {
            let a = "f".repeat(64);
            let b = "0".repeat(64);
            assert_eq!(
                hamming_distance_string_neon(a.as_bytes(), b.as_bytes()).unwrap(),
                256
            );
        }
    }

    #[test]
    fn neon_string_invalid() {
        unsafe {
            assert!(
                hamming_distance_string_neon(b"zzzzzzzzzzzzzzzz", b"0000000000000000").is_err()
            );
            assert!(
                hamming_distance_string_neon(b"@@@@@@@@@@@@@@@@", b"0000000000000000").is_err()
            );
        }
    }

    #[test]
    fn neon_pack_basic() {
        unsafe {
            let a = "f".repeat(32);
            let b = "0".repeat(32);
            assert_eq!(
                hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()).unwrap(),
                128
            );
        }
    }

    #[test]
    fn neon_pack_with_tail() {
        // 48 chars: 32-char pack loop + 16-char NEON tail
        unsafe {
            let a = "f".repeat(48);
            let b = "0".repeat(48);
            assert_eq!(
                hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()).unwrap(),
                192
            );
        }
    }

    #[test]
    fn neon_agrees_with_classic() {
        use crate::classic::hamming_distance_string_classic;
        let a = "0123456789abcdef".repeat(8); // 128 chars
        let b = "fedcba9876543210".repeat(8);
        unsafe {
            assert_eq!(
                hamming_distance_string_neon(a.as_bytes(), b.as_bytes()).unwrap(),
                hamming_distance_string_classic(a.as_bytes(), b.as_bytes()).unwrap()
            );
            assert_eq!(
                hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()).unwrap(),
                hamming_distance_string_classic(a.as_bytes(), b.as_bytes()).unwrap()
            );
        }
    }

    #[test]
    fn neon_pack_with_max_agrees_with_full() {
        // _with_max(max=u64::MAX) should return same result as full pass
        let lengths: &[usize] = &[64, 96, 128, 256, 1024];
        for &len in lengths {
            let a = "f".repeat(len);
            let b = "0".repeat(len);
            unsafe {
                let full = hamming_distance_string_neon_pack(a.as_bytes(), b.as_bytes()).unwrap();
                let with_max = hamming_distance_string_neon_pack_with_max(
                    a.as_bytes(),
                    b.as_bytes(),
                    u64::MAX,
                )
                .unwrap();
                assert_eq!(
                    full, with_max,
                    "mismatch at len={}: full={} with_max={}",
                    len, full, with_max
                );
            }
        }
    }

    #[test]
    fn neon_pack_with_max_returns_sentinel() {
        // Returns u64::MAX when actual distance > max_dist
        let lengths: &[usize] = &[64, 96, 128, 256, 1024];
        for &len in lengths {
            let a = "f".repeat(len);
            let b = "0".repeat(len);
            unsafe {
                let result =
                    hamming_distance_string_neon_pack_with_max(a.as_bytes(), b.as_bytes(), 1)
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
    }

    #[test]
    fn neon_pack_with_max_returns_actual() {
        // Returns actual distance when <= max_dist
        let lengths: &[usize] = &[64, 96, 128, 256, 1024];
        for &len in lengths {
            let a = "f".repeat(len);
            let b = "0".repeat(len);
            let expected = len as u64 * 4;
            unsafe {
                let result = hamming_distance_string_neon_pack_with_max(
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
    }

    #[test]
    fn neon_pack_with_max_invalid_chars() {
        unsafe {
            let a = "z".repeat(64);
            let b = "0".repeat(64);
            assert!(
                hamming_distance_string_neon_pack_with_max(a.as_bytes(), b.as_bytes(), 100)
                    .is_err()
            );
        }
    }
}
