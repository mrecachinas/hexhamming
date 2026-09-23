//! Block-vectorized NEON catalog scanners and fixed-width pairwise kernels.
//!
//! The per-record scanners these replace reduced every record's popcount
//! vector with `uaddlv` and moved the sum to a general register before
//! comparing it; that serial reduction held ~94% of the scan's samples. Here a
//! block of records is reduced with a `vpaddq` tree into one vector holding
//! every record's distance, compared against the threshold once, and
//! compressed into a scalar mask. Only blocks that contain a candidate leave
//! the vector unit.
//!
//! Every block covers 256 bytes: 16 records of 8 or 16 bytes, 8 records of 32
//! bytes, or 4 records of 64 bytes. Distances never exceed the lane type
//! (`u8` up to 128, `u16` up to 256, `u32` up to 512), so the tree is exact.

use std::arch::aarch64::*;

/// A fixed record width with a block reduction.
trait BlockScanner {
    /// Bytes per record.
    const WIDTH: usize;
    /// Records reduced per block.
    const RECORDS: usize;
    /// Largest possible distance for one record (`WIDTH * 8`).
    const MAX_DIST: u64;
    /// `mask` sets bit `lane << LANE_SHIFT` for every matching lane.
    const LANE_SHIFT: u32;
    type Query: Copy;
    type Totals: Copy;

    unsafe fn load_query(query: *const u8) -> Self::Query;
    /// Distances of the `RECORDS` records starting at `base`.
    unsafe fn totals(base: *const u8, query: Self::Query) -> Self::Totals;
    /// One bit per lane whose distance is at most `limit` (`limit <= MAX_DIST`).
    unsafe fn mask(totals: Self::Totals, limit: u64) -> u64;
    unsafe fn store(totals: Self::Totals, out: &mut [u32; 16]);
    unsafe fn distance(record: *const u8, query: Self::Query) -> u64;
}

#[inline(always)]
unsafe fn mask_u8(totals: uint8x16_t, limit: u64) -> u64 {
    let hits = vcleq_u8(totals, vdupq_n_u8(limit as u8));
    // Narrow each 0x00/0xFF lane to a nibble, then keep one bit per lane.
    let nibbles = vshrn_n_u16(vreinterpretq_u16_u8(hits), 4);
    vget_lane_u64::<0>(vreinterpret_u64_u8(nibbles)) & 0x1111_1111_1111_1111
}

#[inline(always)]
unsafe fn store_u8(totals: uint8x16_t, out: &mut [u32; 16]) {
    let lo = vmovl_u8(vget_low_u8(totals));
    let hi = vmovl_high_u8(totals);
    vst1q_u32(out.as_mut_ptr(), vmovl_u16(vget_low_u16(lo)));
    vst1q_u32(out.as_mut_ptr().add(4), vmovl_high_u16(lo));
    vst1q_u32(out.as_mut_ptr().add(8), vmovl_u16(vget_low_u16(hi)));
    vst1q_u32(out.as_mut_ptr().add(12), vmovl_high_u16(hi));
}

#[inline(always)]
unsafe fn popcount16(record: *const u8, query: uint8x16_t) -> uint8x16_t {
    vcntq_u8(veorq_u8(vld1q_u8(record), query))
}

/// 8-byte records: each vector holds two records, and a three-level pairwise
/// add tree over eight vectors leaves record `i`'s distance in lane `i`.
struct W8;

impl BlockScanner for W8 {
    const WIDTH: usize = 8;
    const RECORDS: usize = 16;
    const MAX_DIST: u64 = 64;
    const LANE_SHIFT: u32 = 2;
    type Query = uint8x16_t;
    type Totals = uint8x16_t;

    #[inline(always)]
    unsafe fn load_query(query: *const u8) -> uint8x16_t {
        vreinterpretq_u8_u64(vdupq_n_u64(query.cast::<u64>().read_unaligned()))
    }

    #[inline(always)]
    unsafe fn totals(base: *const u8, q: uint8x16_t) -> uint8x16_t {
        let v = |k: usize| popcount16(base.add(16 * k), q);
        let t0 = vpaddq_u8(v(0), v(1));
        let t1 = vpaddq_u8(v(2), v(3));
        let t2 = vpaddq_u8(v(4), v(5));
        let t3 = vpaddq_u8(v(6), v(7));
        vpaddq_u8(vpaddq_u8(t0, t1), vpaddq_u8(t2, t3))
    }

    #[inline(always)]
    unsafe fn mask(totals: uint8x16_t, limit: u64) -> u64 {
        mask_u8(totals, limit)
    }

    #[inline(always)]
    unsafe fn store(totals: uint8x16_t, out: &mut [u32; 16]) {
        store_u8(totals, out)
    }

    #[inline(always)]
    unsafe fn distance(record: *const u8, q: uint8x16_t) -> u64 {
        let query = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(q));
        (record.cast::<u64>().read_unaligned() ^ query).count_ones() as u64
    }
}

/// 16-byte records: a four-level pairwise add tree over sixteen popcount
/// vectors leaves record `i`'s distance in lane `i`.
struct W16;

impl BlockScanner for W16 {
    const WIDTH: usize = 16;
    const RECORDS: usize = 16;
    const MAX_DIST: u64 = 128;
    const LANE_SHIFT: u32 = 2;
    type Query = uint8x16_t;
    type Totals = uint8x16_t;

    #[inline(always)]
    unsafe fn load_query(query: *const u8) -> uint8x16_t {
        vld1q_u8(query)
    }

    #[inline(always)]
    unsafe fn totals(base: *const u8, q: uint8x16_t) -> uint8x16_t {
        let v = |k: usize| popcount16(base.add(16 * k), q);
        let t = |k: usize| vpaddq_u8(v(2 * k), v(2 * k + 1));
        let u0 = vpaddq_u8(t(0), t(1));
        let u1 = vpaddq_u8(t(2), t(3));
        let u2 = vpaddq_u8(t(4), t(5));
        let u3 = vpaddq_u8(t(6), t(7));
        vpaddq_u8(vpaddq_u8(u0, u1), vpaddq_u8(u2, u3))
    }

    #[inline(always)]
    unsafe fn mask(totals: uint8x16_t, limit: u64) -> u64 {
        mask_u8(totals, limit)
    }

    #[inline(always)]
    unsafe fn store(totals: uint8x16_t, out: &mut [u32; 16]) {
        store_u8(totals, out)
    }

    #[inline(always)]
    unsafe fn distance(record: *const u8, q: uint8x16_t) -> u64 {
        vaddlvq_u8(popcount16(record, q)) as u64
    }
}

/// 32-byte records: the two halves of each record are summed per lane, a
/// three-level tree leaves two `u8` partial sums per record, and a widening
/// pairwise add produces one `u16` distance per lane.
struct W32;

impl BlockScanner for W32 {
    const WIDTH: usize = 32;
    const RECORDS: usize = 8;
    const MAX_DIST: u64 = 256;
    const LANE_SHIFT: u32 = 3;
    type Query = (uint8x16_t, uint8x16_t);
    type Totals = uint16x8_t;

    #[inline(always)]
    unsafe fn load_query(query: *const u8) -> Self::Query {
        (vld1q_u8(query), vld1q_u8(query.add(16)))
    }

    #[inline(always)]
    unsafe fn totals(base: *const u8, q: Self::Query) -> uint16x8_t {
        let v = |k: usize| {
            vaddq_u8(
                popcount16(base.add(32 * k), q.0),
                popcount16(base.add(32 * k + 16), q.1),
            )
        };
        let t0 = vpaddq_u8(v(0), v(1));
        let t1 = vpaddq_u8(v(2), v(3));
        let t2 = vpaddq_u8(v(4), v(5));
        let t3 = vpaddq_u8(v(6), v(7));
        vpaddlq_u8(vpaddq_u8(vpaddq_u8(t0, t1), vpaddq_u8(t2, t3)))
    }

    #[inline(always)]
    unsafe fn mask(totals: uint16x8_t, limit: u64) -> u64 {
        let hits = vmovn_u16(vcleq_u16(totals, vdupq_n_u16(limit as u16)));
        vget_lane_u64::<0>(vreinterpret_u64_u8(hits)) & 0x0101_0101_0101_0101
    }

    #[inline(always)]
    unsafe fn store(totals: uint16x8_t, out: &mut [u32; 16]) {
        vst1q_u32(out.as_mut_ptr(), vmovl_u16(vget_low_u16(totals)));
        vst1q_u32(out.as_mut_ptr().add(4), vmovl_high_u16(totals));
    }

    #[inline(always)]
    unsafe fn distance(record: *const u8, q: Self::Query) -> u64 {
        vaddlvq_u8(vaddq_u8(
            popcount16(record, q.0),
            popcount16(record.add(16), q.1),
        )) as u64
    }
}

/// 64-byte records: four quarters are summed per lane, a two-level tree
/// leaves four `u8` partial sums per record, and two widening pairwise adds
/// produce one `u32` distance per lane.
struct W64;

impl BlockScanner for W64 {
    const WIDTH: usize = 64;
    const RECORDS: usize = 4;
    const MAX_DIST: u64 = 512;
    const LANE_SHIFT: u32 = 4;
    type Query = [uint8x16_t; 4];
    type Totals = uint32x4_t;

    #[inline(always)]
    unsafe fn load_query(query: *const u8) -> Self::Query {
        [
            vld1q_u8(query),
            vld1q_u8(query.add(16)),
            vld1q_u8(query.add(32)),
            vld1q_u8(query.add(48)),
        ]
    }

    #[inline(always)]
    unsafe fn totals(base: *const u8, q: Self::Query) -> uint32x4_t {
        let v = |k: usize| Self::lanes(base.add(64 * k), q);
        let x = vpaddq_u8(vpaddq_u8(v(0), v(1)), vpaddq_u8(v(2), v(3)));
        vpaddlq_u16(vpaddlq_u8(x))
    }

    #[inline(always)]
    unsafe fn mask(totals: uint32x4_t, limit: u64) -> u64 {
        let hits = vmovn_u32(vcleq_u32(totals, vdupq_n_u32(limit as u32)));
        vget_lane_u64::<0>(vreinterpret_u64_u16(hits)) & 0x0001_0001_0001_0001
    }

    #[inline(always)]
    unsafe fn store(totals: uint32x4_t, out: &mut [u32; 16]) {
        vst1q_u32(out.as_mut_ptr(), totals);
    }

    #[inline(always)]
    unsafe fn distance(record: *const u8, q: Self::Query) -> u64 {
        vaddlvq_u8(Self::lanes(record, q)) as u64
    }
}

impl W64 {
    /// Per-lane popcount of one record (at most 32 per lane).
    #[inline(always)]
    unsafe fn lanes(record: *const u8, q: [uint8x16_t; 4]) -> uint8x16_t {
        vaddq_u8(
            vaddq_u8(popcount16(record, q[0]), popcount16(record.add(16), q[1])),
            vaddq_u8(
                popcount16(record.add(32), q[2]),
                popcount16(record.add(48), q[3]),
            ),
        )
    }
}

/// Distance limit for `max_dist`, where negative means unlimited.
#[inline(always)]
fn limit_for<S: BlockScanner>(max_dist: i64) -> u64 {
    if max_dist < 0 {
        S::MAX_DIST
    } else {
        (max_dist as u64).min(S::MAX_DIST)
    }
}

/// SAFETY: `records.len()` is a multiple of `S::WIDTH` and
/// `query.len() == S::WIDTH`.
#[inline(always)]
unsafe fn first<S: BlockScanner>(records: &[u8], query: &[u8], max_dist: i64) -> Option<usize> {
    debug_assert_eq!(query.len(), S::WIDTH);
    let count = records.len() / S::WIDTH;
    if count == 0 {
        return None;
    }
    if max_dist < 0 || max_dist as u64 >= S::MAX_DIST {
        return Some(0);
    }
    let limit = max_dist as u64;
    let q = S::load_query(query.as_ptr());
    let base = records.as_ptr();
    let mut index = 0;
    while index + S::RECORDS <= count {
        let mask = S::mask(S::totals(base.add(index * S::WIDTH), q), limit);
        if mask != 0 {
            return Some(index + (mask.trailing_zeros() >> S::LANE_SHIFT) as usize);
        }
        index += S::RECORDS;
    }
    while index < count {
        if S::distance(base.add(index * S::WIDTH), q) <= limit {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// SAFETY: as for [`first`].
#[inline(always)]
unsafe fn best<S: BlockScanner>(
    records: &[u8],
    query: &[u8],
    max_dist: i64,
) -> Option<(u64, usize)> {
    debug_assert_eq!(query.len(), S::WIDTH);
    let count = records.len() / S::WIDTH;
    // A record is reported only if its distance is at most `limit`: first
    // `max_dist`, then one less than the best distance seen, so equal
    // distances keep the lowest index.
    let mut limit = limit_for::<S>(max_dist);
    let mut best = None;
    let q = S::load_query(query.as_ptr());
    let base = records.as_ptr();
    let mut totals = [0u32; 16];
    let mut index = 0;
    while index + S::RECORDS <= count {
        let block = S::totals(base.add(index * S::WIDTH), q);
        let mut mask = S::mask(block, limit);
        if mask != 0 {
            S::store(block, &mut totals);
            while mask != 0 {
                let lane = (mask.trailing_zeros() >> S::LANE_SHIFT) as usize;
                mask &= mask - 1;
                let distance = totals[lane] as u64;
                // Lanes were selected with the limit in force at the start of
                // the block; an earlier lane may have tightened it since.
                if distance <= limit {
                    best = Some((distance, index + lane));
                    if distance == 0 {
                        return best;
                    }
                    limit = distance - 1;
                }
            }
        }
        index += S::RECORDS;
    }
    while index < count {
        let distance = S::distance(base.add(index * S::WIDTH), q);
        if distance <= limit {
            best = Some((distance, index));
            if distance == 0 {
                return best;
            }
            limit = distance - 1;
        }
        index += 1;
    }
    best
}

/// SAFETY: as for [`first`].
#[inline(always)]
unsafe fn all<S: BlockScanner>(records: &[u8], query: &[u8], max_dist: i64) -> Vec<(u64, usize)> {
    debug_assert_eq!(query.len(), S::WIDTH);
    let count = records.len() / S::WIDTH;
    let limit = limit_for::<S>(max_dist);
    let mut matches = Vec::new();
    let q = S::load_query(query.as_ptr());
    let base = records.as_ptr();
    let mut totals = [0u32; 16];
    let mut index = 0;
    while index + S::RECORDS <= count {
        let block = S::totals(base.add(index * S::WIDTH), q);
        let mut mask = S::mask(block, limit);
        if mask != 0 {
            S::store(block, &mut totals);
            while mask != 0 {
                let lane = (mask.trailing_zeros() >> S::LANE_SHIFT) as usize;
                mask &= mask - 1;
                matches.push((totals[lane] as u64, index + lane));
            }
        }
        index += S::RECORDS;
    }
    while index < count {
        let distance = S::distance(base.add(index * S::WIDTH), q);
        if distance <= limit {
            matches.push((distance, index));
        }
        index += 1;
    }
    matches
}

macro_rules! scanners {
    ($scanner:ty, $first:ident, $best:ident, $all:ident) => {
        pub(crate) fn $first(records: &[u8], query: &[u8], max_dist: i64) -> Option<usize> {
            assert_eq!(query.len(), <$scanner>::WIDTH);
            unsafe { first::<$scanner>(records, query, max_dist) }
        }

        pub(crate) fn $best(records: &[u8], query: &[u8], max_dist: i64) -> Option<(u64, usize)> {
            assert_eq!(query.len(), <$scanner>::WIDTH);
            unsafe { best::<$scanner>(records, query, max_dist) }
        }

        pub(crate) fn $all(records: &[u8], query: &[u8], max_dist: i64) -> Vec<(u64, usize)> {
            assert_eq!(query.len(), <$scanner>::WIDTH);
            unsafe { all::<$scanner>(records, query, max_dist) }
        }
    };
}

scanners!(W8, array_first_8, array_best_8, array_all_8);
scanners!(W16, array_first_16, array_best_16, array_all_16);
scanners!(W32, array_first_32, array_best_32, array_all_32);
scanners!(W64, array_first_64, array_best_64, array_all_64);

/// Distance between two `WIDTH`-byte records for the pairwise widths.
///
/// SAFETY: `a` and `b` are valid for `WIDTH` bytes.
#[inline(always)]
unsafe fn pair_distance<const WIDTH: usize>(a: *const u8, b: *const u8) -> u64 {
    match WIDTH {
        8 => (a.cast::<u64>().read_unaligned() ^ b.cast::<u64>().read_unaligned()).count_ones()
            as u64,
        20 => {
            let head = vaddlvq_u8(vcntq_u8(veorq_u8(vld1q_u8(a), vld1q_u8(b)))) as u64;
            let tail =
                a.add(16).cast::<u32>().read_unaligned() ^ b.add(16).cast::<u32>().read_unaligned();
            head + tail.count_ones() as u64
        }
        _ => {
            // Multiples of 16 up to 64 bytes: at most 32 per lane.
            let mut lanes = vdupq_n_u8(0);
            for chunk in 0..WIDTH / 16 {
                let x = veorq_u8(vld1q_u8(a.add(16 * chunk)), vld1q_u8(b.add(16 * chunk)));
                lanes = vaddq_u8(lanes, vcntq_u8(x));
            }
            vaddlvq_u8(lanes) as u64
        }
    }
}

/// Call `emit(index, distance)` for every record pair, four pairs per
/// iteration so the independent reductions overlap.
#[inline(always)]
fn for_each_pair<const WIDTH: usize>(a: &[u8], b: &[u8], mut emit: impl FnMut(usize, u64)) {
    assert!(matches!(WIDTH, 8 | 16 | 20 | 32 | 64));
    assert_eq!(a.len(), b.len());
    let count = a.len() / WIDTH;
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    unsafe {
        while i + 4 <= count {
            let d0 = pair_distance::<WIDTH>(ap.add(i * WIDTH), bp.add(i * WIDTH));
            let d1 = pair_distance::<WIDTH>(ap.add((i + 1) * WIDTH), bp.add((i + 1) * WIDTH));
            let d2 = pair_distance::<WIDTH>(ap.add((i + 2) * WIDTH), bp.add((i + 2) * WIDTH));
            let d3 = pair_distance::<WIDTH>(ap.add((i + 3) * WIDTH), bp.add((i + 3) * WIDTH));
            emit(i, d0);
            emit(i + 1, d1);
            emit(i + 2, d2);
            emit(i + 3, d3);
            i += 4;
        }
        while i < count {
            emit(
                i,
                pair_distance::<WIDTH>(ap.add(i * WIDTH), bp.add(i * WIDTH)),
            );
            i += 1;
        }
    }
}

/// Widths with a dedicated pairwise kernel.
#[inline]
pub(crate) fn has_pairwise_kernel(width: usize) -> bool {
    matches!(width, 8 | 16 | 20 | 32 | 64)
}

macro_rules! dispatch_pair_width {
    ($width:expr, $body:ident) => {
        match $width {
            8 => $body!(8),
            16 => $body!(16),
            20 => $body!(20),
            32 => $body!(32),
            64 => $body!(64),
            _ => unreachable!("no pairwise kernel for width {}", $width),
        }
    };
}

/// Pairwise distances for a width accepted by [`has_pairwise_kernel`].
pub(crate) fn pairwise_distances(a: &[u8], b: &[u8], width: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len() / width);
    macro_rules! run {
        ($w:literal) => {
            for_each_pair::<$w>(a, b, |_, d| out.push(d))
        };
    }
    dispatch_pair_width!(width, run);
    out
}

/// Write little-endian pairwise distances into `out` (8 bytes per record) for
/// a width accepted by [`has_pairwise_kernel`].
pub(crate) fn pairwise_distances_into(a: &[u8], b: &[u8], width: usize, out: &mut [u8]) {
    assert_eq!(out.len(), a.len() / width * 8);
    macro_rules! run {
        ($w:literal) => {
            for_each_pair::<$w>(a, b, |i, d| {
                out[i * 8..i * 8 + 8].copy_from_slice(&d.to_le_bytes())
            })
        };
    }
    dispatch_pair_width!(width, run);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut v = state;
                v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((v ^ (v >> 31)) >> 56) as u8
            })
            .collect()
    }

    fn distance(a: &[u8], b: &[u8]) -> u64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x ^ y).count_ones() as u64)
            .sum()
    }

    fn within(d: u64, max_dist: i64) -> bool {
        max_dist < 0 || d <= max_dist as u64
    }

    type Scanners = (
        fn(&[u8], &[u8], i64) -> Option<usize>,
        fn(&[u8], &[u8], i64) -> Option<(u64, usize)>,
        fn(&[u8], &[u8], i64) -> Vec<(u64, usize)>,
    );

    fn scanners(width: usize) -> Scanners {
        match width {
            8 => (array_first_8, array_best_8, array_all_8),
            16 => (array_first_16, array_best_16, array_all_16),
            32 => (array_first_32, array_best_32, array_all_32),
            64 => (array_first_64, array_best_64, array_all_64),
            _ => unreachable!(),
        }
    }

    fn check(records: &[u8], query: &[u8], max_dist: i64, context: &str) {
        let width = query.len();
        let distances: Vec<u64> = records
            .chunks_exact(width)
            .map(|record| distance(record, query))
            .collect();
        let all: Vec<(u64, usize)> = distances
            .iter()
            .enumerate()
            .filter(|&(_, &d)| within(d, max_dist))
            .map(|(i, &d)| (d, i))
            .collect();
        let first = all.first().map(|&(_, i)| i);
        let best = all.iter().copied().min();
        let (scan_first, scan_best, scan_all) = scanners(width);
        assert_eq!(
            scan_first(records, query, max_dist),
            first,
            "first {context}"
        );
        assert_eq!(scan_best(records, query, max_dist), best, "best {context}");
        assert_eq!(scan_all(records, query, max_dist), all, "all {context}");
    }

    #[test]
    fn scanners_match_oracle() {
        for width in [8usize, 16, 32, 64] {
            let bits = width as i64 * 8;
            let limits = [
                0,
                1,
                2,
                3,
                bits / 4,
                bits / 2,
                bits - 1,
                bits,
                bits + 1,
                255,
                256,
                1000,
                i64::MAX,
                -1,
            ];
            for count in [
                0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 100,
            ] {
                for seed in 0..4u64 {
                    let query = pseudo_random(width, 0x51 + seed + 16 * width as u64);
                    let mut records = pseudo_random(count * width, 0xA1 + seed + 16 * width as u64);
                    // Exact duplicates exercise the lowest-index tie break and
                    // the exact-match short circuit; the complement record has
                    // the maximum distance; the near records sit a few bits
                    // either side of small limits.
                    for (i, record) in records.chunks_exact_mut(width).enumerate() {
                        match (i + seed as usize) % 7 {
                            0 if seed % 2 == 0 => record.copy_from_slice(&query),
                            1 => record.iter_mut().zip(&query).for_each(|(r, q)| *r = !q),
                            2 | 3 => {
                                record.copy_from_slice(&query);
                                record[i % width] ^= 0b1011 >> (i % 3);
                            }
                            _ => {}
                        }
                    }
                    for &max_dist in &limits {
                        check(
                            &records,
                            &query,
                            max_dist,
                            &format!("width={width} count={count} seed={seed} max_dist={max_dist}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn best_prefers_lowest_index_among_equal_distances() {
        for width in [8usize, 16, 32, 64] {
            let query = vec![0u8; width];
            let mut records = vec![0xFFu8; width * 40];
            for &index in &[5usize, 18, 33] {
                records[index * width] = 0b0000_0011; // distance 2 from the query
                records[index * width + 1..(index + 1) * width].fill(0);
            }
            let (_, scan_best, _) = scanners(width);
            assert_eq!(
                scan_best(&records, &query, -1),
                Some((2, 5)),
                "width={width}"
            );
        }
    }

    #[test]
    fn pairwise_matches_oracle() {
        for width in [8usize, 16, 20, 32, 64] {
            for count in [0usize, 1, 3, 4, 5, 8, 17, 40] {
                let a = pseudo_random(count * width, 0x11 + width as u64);
                let mut b = pseudo_random(count * width, 0x22 + width as u64);
                if count > 1 {
                    b[..width]
                        .iter_mut()
                        .zip(&a[..width])
                        .for_each(|(y, x)| *y = !x);
                }
                let want: Vec<u64> = a
                    .chunks_exact(width)
                    .zip(b.chunks_exact(width))
                    .map(|(x, y)| distance(x, y))
                    .collect();
                assert_eq!(pairwise_distances(&a, &b, width), want, "width={width}");
                let mut out = vec![0u8; count * 8];
                pairwise_distances_into(&a, &b, width, &mut out);
                let got: Vec<u64> = out
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                assert_eq!(got, want, "into width={width}");
            }
        }
        assert!(!has_pairwise_kernel(24));
    }
}
