//! Persistent catalogs and optional multi-index hashing (MIH).
//!
//! `Catalog` owns a fixed-width byte catalog once and serves repeated queries
//! without reacquiring or revalidating the catalog buffer. `Catalog::with_index`
//! additionally builds a Multi-Index Hashing index using flat CSR bucket tables:
//! for each of `m` substrings, memory is
//!
//! `4 * ((2^s_j + 1) + N)` bytes,
//!
//! where `s_j` is that substring's bit width and `N` is the record count. The
//! default planner chooses `s ≈ ceil(log2(N)) + 2`, capped at 22 bits to keep
//! 256-bit million-record catalogs around half the prototype's memory while
//! still giving large wins for small radii. Widths that would require more
//! than the supported number of substring tables simply keep a linear catalog.

use crate::api::select_array_scanner_for_width;
use crate::par;
use crate::{
    bytes_array_all_many_within_dist, bytes_array_all_within_dist,
    bytes_array_best_many_within_dist, bytes_array_best_within_dist,
    bytes_array_first_many_within_dist, bytes_array_first_within_dist,
    select_bytes_kernel_for_width, BytesKernel,
};
use std::mem::size_of;

const MAX_M: usize = 64;
const MAX_SUB_BITS: usize = 22;
/// Largest per-substring probe radius `enumerate_neighbors` supports.
const MAX_PROBE_RADIUS: usize = 4;

// Planner cost model, in nanoseconds per query. The constants are fitted to a
// sweep of probe radii 0..=4 on 8/16/32-byte catalogs of 10k-1M records
// (Apple M4 Max, block scanners): a bucket probe, including verifying its
// ~N/2^s records, costs 1.2-5.6 ns, rising with index size as probes miss
// cache and TLB. A linear scan costs about 0.11 ns per record or 0.0133 ns
// per byte, whichever is larger; split across threads it runs about 1.4x
// faster at 512 KiB, 2.5x at 1 MiB and 4-6x from 2 MiB (modelled as up to 4x).
// Slower linear scans elsewhere only make the planner more conservative.
const QUERY_OVERHEAD_NS: f64 = 50.0;
const PROBE_BASE_NS: f64 = 1.5;
const PROBE_CACHE_MISS_NS: f64 = 4.0;
const PROBE_MISS_SATURATION_BYTES: f64 = (128u64 << 20) as f64;
const SCAN_NS_PER_RECORD: f64 = 0.11;
const SCAN_NS_PER_BYTE: f64 = 0.0133;
const PARALLEL_SPEEDUP_BYTES: f64 = (400u64 << 10) as f64;
const MAX_PARALLEL_SPEEDUP: f64 = 4.0;

/// Estimated cost of scanning `n` records of `width` bytes linearly.
fn linear_scan_ns(n: usize, width: usize) -> f64 {
    let bytes = n.saturating_mul(width);
    let serial = (n as f64 * SCAN_NS_PER_RECORD).max(bytes as f64 * SCAN_NS_PER_BYTE);
    let threshold = crate::api::parallel_threshold(select_array_scanner_for_width(width).is_some());
    if bytes >= threshold {
        serial / (bytes as f64 / PARALLEL_SPEEDUP_BYTES).clamp(1.0, MAX_PARALLEL_SPEEDUP)
    } else {
        serial
    }
}

/// Number of keys within Hamming distance `t` of an `s`-bit key.
fn ball_size(s: u32, t: u32) -> f64 {
    let mut term = 1.0;
    let mut total = 1.0;
    for i in 1..=t.min(s) {
        term = term * f64::from(s - i + 1) / f64::from(i);
        total += term;
    }
    total
}

#[derive(Clone, Copy, Debug)]
struct MihParams {
    m: usize,
    s_bits: [u8; MAX_M],
    s_off: [u16; MAX_M],
}

impl MihParams {
    fn plan(bits: usize, n: usize) -> Option<Self> {
        if bits == 0 || bits > (u16::MAX as usize) || bits > MAX_M * MAX_SUB_BITS {
            return None;
        }
        let log2n = ((n as f64).max(2.0)).log2().ceil() as usize;
        let target = (log2n + 2).clamp(4, MAX_SUB_BITS);
        let m = bits.div_ceil(target).clamp(1, MAX_M);
        Self::with_m(bits, m)
    }

    fn with_m(bits: usize, m: usize) -> Option<Self> {
        if m == 0 || m > MAX_M || bits.div_ceil(m) > MAX_SUB_BITS {
            return None;
        }
        let mut s_bits = [0u8; MAX_M];
        let mut s_off = [0u16; MAX_M];
        let base = bits / m;
        let extra = bits % m;
        let mut off = 0u16;
        for j in 0..m {
            let w = base + usize::from(j < extra);
            s_bits[j] = w as u8;
            s_off[j] = off;
            off += w as u16;
        }
        Some(Self { m, s_bits, s_off })
    }
}

#[inline]
fn extract_substring(code: &[u8], bit_off: usize, width: usize) -> u32 {
    debug_assert!(width > 0 && width <= MAX_SUB_BITS);
    let start_byte = bit_off / 8;
    let bit_in_byte = bit_off & 7;
    let end_bit = bit_off + width;
    let end_byte = end_bit.div_ceil(8);
    let mut acc = 0u64;
    for &byte in &code[start_byte..end_byte] {
        acc = (acc << 8) | u64::from(byte);
    }
    let total_bits = (end_byte - start_byte) * 8;
    let shift = total_bits - bit_in_byte - width;
    let mask = (1u32 << width) - 1;
    ((acc >> shift) as u32) & mask
}

struct BucketTable {
    offsets: Vec<u32>,
    records: Vec<u32>,
    s_bits: u8,
    bit_off: u16,
}

impl BucketTable {
    fn build(records: &[u8], width: usize, s_bits: u8, bit_off: u16) -> Self {
        let n = records.len() / width;
        let cardinality = 1usize << usize::from(s_bits);
        let mut offsets = vec![0u32; cardinality + 1];
        for record in records.chunks_exact(width) {
            let key = extract_substring(record, bit_off as usize, s_bits as usize) as usize;
            offsets[key + 1] += 1;
        }
        for i in 1..=cardinality {
            offsets[i] += offsets[i - 1];
        }

        let mut placed = offsets.clone();
        let mut ids = vec![0u32; n];
        for (record_id, record) in records.chunks_exact(width).enumerate() {
            let key = extract_substring(record, bit_off as usize, s_bits as usize) as usize;
            let slot = placed[key] as usize;
            ids[slot] = record_id as u32;
            placed[key] += 1;
        }
        Self {
            offsets,
            records: ids,
            s_bits,
            bit_off,
        }
    }

    #[inline]
    fn bucket(&self, key: u32) -> &[u32] {
        let start = self.offsets[key as usize] as usize;
        let end = self.offsets[key as usize + 1] as usize;
        &self.records[start..end]
    }
}

struct MihIndex {
    params: MihParams,
    tables: Vec<BucketTable>,
}

impl MihIndex {
    fn build(records: &[u8], width: usize) -> Option<Self> {
        let n = records.len() / width;
        if n == 0 || n > u32::MAX as usize {
            return None;
        }
        let params = MihParams::plan(width.checked_mul(8)?, n)?;
        let mut tables = Vec::with_capacity(params.m);
        for j in 0..params.m {
            tables.push(BucketTable::build(
                records,
                width,
                params.s_bits[j],
                params.s_off[j],
            ));
        }
        Some(Self { params, tables })
    }

    /// Estimated cost of one query probing every table at radius `t`.
    fn query_ns(&self, t: u32) -> f64 {
        let probes: f64 = self
            .tables
            .iter()
            .map(|table| ball_size(u32::from(table.s_bits), t))
            .sum();
        let miss_share =
            (self.memory_overhead_bytes() as f64 / PROBE_MISS_SATURATION_BYTES).min(1.0);
        QUERY_OVERHEAD_NS + probes * (PROBE_BASE_NS + PROBE_CACHE_MISS_NS * miss_share)
    }

    /// Probe radius to use for `max_dist`, or `None` when a linear scan of the
    /// `n`-record catalog is expected to be at least as fast. Records within
    /// `max_dist` differ from the query by at most `max_dist / m` bits in some
    /// substring (pigeonhole), so that radius finds all of them.
    fn probe_radius(&self, max_dist: i64, n: usize, width: usize) -> Option<u32> {
        if max_dist < 0 {
            return None;
        }
        let t = max_dist as usize / self.params.m;
        if t > MAX_PROBE_RADIUS {
            return None;
        }
        let t = t as u32;
        (self.query_ns(t) < linear_scan_ns(n, width)).then_some(t)
    }

    #[inline]
    fn memory_overhead_bytes(&self) -> usize {
        self.tables
            .iter()
            .map(|table| (table.offsets.len() + table.records.len()) * size_of::<u32>())
            .sum()
    }

    #[inline]
    fn extract_query_subs(&self, query: &[u8]) -> [u32; MAX_M] {
        let mut subs = [0u32; MAX_M];
        for (j, table) in self.tables.iter().enumerate() {
            subs[j] = extract_substring(query, table.bit_off as usize, table.s_bits as usize);
        }
        subs
    }

    /// Calls `f` with every record that shares a substring within `t` bits
    /// of the query's. By pigeonhole this includes every record within
    /// `(t + 1) * m - 1` bits, in no particular order and possibly repeated.
    #[inline]
    fn for_each_candidate<F: FnMut(u32)>(&self, query: &[u8], t: u32, mut f: F) {
        let subs = self.extract_query_subs(query);
        for (j, table) in self.tables.iter().enumerate() {
            enumerate_neighbors(subs[j], u32::from(table.s_bits), t, |key| {
                for &record_id in table.bucket(key) {
                    f(record_id);
                }
            });
        }
    }
}

#[inline]
fn enumerate_neighbors<F: FnMut(u32)>(key: u32, width: u32, radius: u32, mut f: F) {
    f(key);
    if radius >= 1 {
        for i in 0..width {
            f(key ^ (1u32 << i));
        }
    }
    if radius >= 2 {
        for i in 0..width {
            for j in (i + 1)..width {
                f(key ^ (1u32 << i) ^ (1u32 << j));
            }
        }
    }
    if radius >= 3 {
        for i in 0..width {
            for j in (i + 1)..width {
                for k in (j + 1)..width {
                    f(key ^ (1u32 << i) ^ (1u32 << j) ^ (1u32 << k));
                }
            }
        }
    }
    if radius >= 4 {
        for i in 0..width {
            for j in (i + 1)..width {
                for k in (j + 1)..width {
                    for l in (k + 1)..width {
                        f(key ^ (1u32 << i) ^ (1u32 << j) ^ (1u32 << k) ^ (1u32 << l));
                    }
                }
            }
        }
    }
}

/// Owned fixed-width byte catalog with an optional MIH index.
pub struct Catalog {
    records: Vec<u8>,
    width: usize,
    index: Option<MihIndex>,
}

impl Catalog {
    /// Copy `records` into a reusable fixed-width catalog without an index.
    pub fn new(records: &[u8], width: usize) -> Result<Catalog, &'static str> {
        validate_records(records, width)?;
        Ok(Catalog {
            records: records.to_vec(),
            width,
            index: None,
        })
    }

    /// Copy `records` and build an MIH index when the planner can bound memory.
    pub fn with_index(records: &[u8], width: usize) -> Result<Catalog, &'static str> {
        validate_records(records, width)?;
        let index = MihIndex::build(records, width);
        Ok(Catalog {
            records: records.to_vec(),
            width,
            index,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.records.len() / self.width
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn has_index(&self) -> bool {
        self.index.is_some()
    }

    #[inline]
    pub fn records(&self) -> &[u8] {
        &self.records
    }

    /// Bytes used by MIH CSR tables, excluding the owned record bytes.
    #[inline]
    pub fn index_memory_overhead_bytes(&self) -> usize {
        self.index
            .as_ref()
            .map(MihIndex::memory_overhead_bytes)
            .unwrap_or(0)
    }

    pub fn first_within(&self, query: &[u8], max_dist: i64) -> Result<Option<usize>, &'static str> {
        self.validate_query(query)?;
        if let Some((index, t)) = self.index_for_radius(max_dist) {
            return Ok(self.indexed_first(index, query, max_dist, t));
        }
        bytes_array_first_within_dist(&self.records, query, max_dist)
    }

    pub fn best_within(
        &self,
        query: &[u8],
        max_dist: i64,
    ) -> Result<Option<(u64, usize)>, &'static str> {
        self.validate_query(query)?;
        if let Some((index, t)) = self.index_for_radius(max_dist) {
            return Ok(self.indexed_best(index, query, max_dist, t));
        }
        bytes_array_best_within_dist(&self.records, query, max_dist)
    }

    pub fn all_within(
        &self,
        query: &[u8],
        max_dist: i64,
    ) -> Result<Vec<(u64, usize)>, &'static str> {
        self.validate_query(query)?;
        if let Some((index, t)) = self.index_for_radius(max_dist) {
            return Ok(self.indexed_all(index, query, max_dist, t));
        }
        bytes_array_all_within_dist(&self.records, query, max_dist)
    }

    pub fn first_many_within(
        &self,
        queries: &[u8],
        max_dist: i64,
    ) -> Result<Vec<Option<usize>>, &'static str> {
        self.validate_queries(queries)?;
        if self.index_for_radius(max_dist).is_none() {
            return bytes_array_first_many_within_dist(
                &self.records,
                queries,
                self.width,
                max_dist,
            );
        }
        self.map_indexed_queries(queries, max_dist, |query| {
            self.first_within(query, max_dist)
        })
    }

    pub fn best_many_within(
        &self,
        queries: &[u8],
        max_dist: i64,
    ) -> Result<Vec<Option<(u64, usize)>>, &'static str> {
        self.validate_queries(queries)?;
        if self.index_for_radius(max_dist).is_none() {
            return bytes_array_best_many_within_dist(&self.records, queries, self.width, max_dist);
        }
        self.map_indexed_queries(queries, max_dist, |query| self.best_within(query, max_dist))
    }

    pub fn all_many_within(
        &self,
        queries: &[u8],
        max_dist: i64,
    ) -> Result<Vec<Vec<(u64, usize)>>, &'static str> {
        self.validate_queries(queries)?;
        if self.index_for_radius(max_dist).is_none() {
            return bytes_array_all_many_within_dist(&self.records, queries, self.width, max_dist);
        }
        self.map_indexed_queries(queries, max_dist, |query| self.all_within(query, max_dist))
    }

    /// Run `f` for every query in order, spreading queries across threads when
    /// their estimated total cost matches a linear scan worth splitting.
    fn map_indexed_queries<R, F>(
        &self,
        queries: &[u8],
        max_dist: i64,
        f: F,
    ) -> Result<Vec<R>, &'static str>
    where
        R: Send,
        F: Fn(&[u8]) -> Result<R, &'static str> + Sync,
    {
        let count = queries.len() / self.width;
        let query = |q: usize| &queries[q * self.width..(q + 1) * self.width];
        let bytes_equivalent = (self.estimated_query_ns(max_dist) / SCAN_NS_PER_BYTE) as usize;
        let threshold = crate::api::parallel_threshold(true);
        let Some(plan) = par::plan_items(count, bytes_equivalent, threshold) else {
            return (0..count).map(|q| f(query(q))).collect();
        };
        par::map_chunks(plan.chunks, |chunk| {
            let (start, end) = plan.range(chunk, count);
            (start..end).map(|q| f(query(q))).collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// Estimated nanoseconds for one query at `max_dist` with the strategy the
    /// catalog will choose (index or linear scan).
    pub fn estimated_query_ns(&self, max_dist: i64) -> f64 {
        match self.index_for_radius(max_dist) {
            Some((index, t)) => index.query_ns(t),
            None => linear_scan_ns(self.len(), self.width),
        }
    }

    #[inline]
    fn validate_query(&self, query: &[u8]) -> Result<(), &'static str> {
        if query.len() != self.width {
            return Err("query size must equal catalog width");
        }
        Ok(())
    }

    #[inline]
    fn validate_queries(&self, queries: &[u8]) -> Result<(), &'static str> {
        if queries.len() % self.width != 0 {
            return Err("queries size must be multiplier of catalog width");
        }
        Ok(())
    }

    #[inline]
    fn index_for_radius(&self, max_dist: i64) -> Option<(&MihIndex, u32)> {
        let index = self.index.as_ref()?;
        let t = index.probe_radius(max_dist, self.len(), self.width)?;
        Some((index, t))
    }

    #[inline]
    fn record(&self, record_id: u32) -> &[u8] {
        let start = record_id as usize * self.width;
        &self.records[start..start + self.width]
    }

    // Candidates arrive unordered and possibly repeated, so ties are broken
    // by record index explicitly instead of sorting the candidate list.

    #[inline]
    fn indexed_first(
        &self,
        index: &MihIndex,
        query: &[u8],
        max_dist: i64,
        probe_radius: u32,
    ) -> Option<usize> {
        let kernel = select_bytes_kernel_for_width(self.width);
        let mut first: Option<u32> = None;
        index.for_each_candidate(query, probe_radius, |record_id| {
            if first.is_some_and(|found| record_id >= found) {
                return;
            }
            if kernel(self.record(record_id), query, max_dist) != u64::MAX {
                first = Some(record_id);
            }
        });
        first.map(|record_id| record_id as usize)
    }

    #[inline]
    fn indexed_best(
        &self,
        index: &MihIndex,
        query: &[u8],
        max_dist: i64,
        probe_radius: u32,
    ) -> Option<(u64, usize)> {
        let kernel = select_bytes_kernel_for_width(self.width);
        let mut best: Option<(u64, u32)> = None;
        index.for_each_candidate(query, probe_radius, |record_id| {
            // Admit ties so a lower index at the same distance can win.
            let limit = best.map_or(max_dist, |(distance, _)| distance as i64);
            let distance = kernel(self.record(record_id), query, limit);
            if distance == u64::MAX {
                return;
            }
            let better = match best {
                None => true,
                Some((best_distance, best_id)) => {
                    distance < best_distance || (distance == best_distance && record_id < best_id)
                }
            };
            if better {
                best = Some((distance, record_id));
            }
        });
        best.map(|(distance, record_id)| (distance, record_id as usize))
    }

    #[inline]
    fn indexed_all(
        &self,
        index: &MihIndex,
        query: &[u8],
        max_dist: i64,
        probe_radius: u32,
    ) -> Vec<(u64, usize)> {
        let kernel: BytesKernel = select_bytes_kernel_for_width(self.width);
        let mut matches = Vec::new();
        index.for_each_candidate(query, probe_radius, |record_id| {
            let distance = kernel(self.record(record_id), query, max_dist);
            if distance != u64::MAX {
                matches.push((distance, record_id as usize));
            }
        });
        matches.sort_unstable_by_key(|&(_, record_id)| record_id);
        matches.dedup_by_key(|&mut (_, record_id)| record_id);
        matches
    }
}

fn validate_records(records: &[u8], width: usize) -> Result<(), &'static str> {
    if width == 0 {
        return Err("width must be >0");
    }
    if records.len() % width != 0 {
        return Err("records size must be multiplier of width");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Catalog;
    use crate::{
        bytes_array_all_many_within_dist, bytes_array_all_within_dist,
        bytes_array_best_many_within_dist, bytes_array_best_within_dist,
        bytes_array_first_many_within_dist, bytes_array_first_within_dist,
    };

    fn make_records(width: usize, count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = vec![0u8; width * count];
        for byte in &mut out {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 32) as u8;
        }
        out
    }

    fn flip_bits(buf: &mut [u8], count: usize, seed: u64) {
        let mut state = seed;
        for _ in 0..count {
            state = state
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            let bit = (state as usize) % (buf.len() * 8);
            buf[bit / 8] ^= 1 << (bit & 7);
        }
    }

    // The planner only probes when it expects a win, which small test catalogs
    // rarely trigger at larger radii. Drive the indexed paths directly at every
    // probe radius, with duplicate records and near neighbours planted so ties
    // and repeated candidates occur.
    #[test]
    fn indexed_paths_match_linear_at_every_probe_radius() {
        for &(width, count) in &[(2usize, 300usize), (3, 257), (8, 400), (16, 300), (32, 200)] {
            let mut records = make_records(width, count, 0x5EED ^ width as u64);
            let query = records[7 * width..8 * width].to_vec();
            for (k, target) in [3usize, 40, 41, 150, count - 1].into_iter().enumerate() {
                let mut near = query.clone();
                flip_bits(&mut near, k, 0xB17 + k as u64);
                records[target * width..(target + 1) * width].copy_from_slice(&near);
            }
            let cat = Catalog::with_index(&records, width).unwrap();
            let index = cat.index.as_ref().expect("indexable width");
            let m = index.params.m;
            for t in 0..=super::MAX_PROBE_RADIUS as u32 {
                for max_dist in [t as i64 * m as i64, (t as i64 + 1) * m as i64 - 1] {
                    let label = format!("width={width} t={t} max_dist={max_dist}");
                    assert_eq!(
                        cat.indexed_first(index, &query, max_dist, t),
                        bytes_array_first_within_dist(&records, &query, max_dist).unwrap(),
                        "first {label}"
                    );
                    assert_eq!(
                        cat.indexed_best(index, &query, max_dist, t),
                        bytes_array_best_within_dist(&records, &query, max_dist).unwrap(),
                        "best {label}"
                    );
                    assert_eq!(
                        cat.indexed_all(index, &query, max_dist, t),
                        bytes_array_all_within_dist(&records, &query, max_dist).unwrap(),
                        "all {label}"
                    );
                }
            }
        }
    }

    #[test]
    fn catalog_validation_and_accessors() {
        assert!(Catalog::new(b"abc", 0).is_err());
        assert!(Catalog::new(b"abc", 2).is_err());
        let cat = Catalog::new(b"abcd", 2).unwrap();
        assert_eq!(cat.len(), 2);
        assert!(!cat.is_empty());
        assert_eq!(cat.width(), 2);
        assert_eq!(cat.records(), b"abcd");
        assert!(!cat.has_index());
    }

    #[test]
    fn catalog_tie_breaks_match_linear() {
        let width = 8;
        let mut records = vec![0xFF; width * 8];
        records[3 * width..4 * width].fill(0);
        records[6 * width..7 * width].fill(0);
        let query = vec![0; width];
        let cat = Catalog::with_index(&records, width).unwrap();
        assert_eq!(cat.first_within(&query, 0).unwrap(), Some(3));
        assert_eq!(cat.best_within(&query, 0).unwrap(), Some((0, 3)));
        assert_eq!(cat.all_within(&query, 0).unwrap(), vec![(0, 3), (0, 6)]);
    }

    #[test]
    fn catalog_differential_widths_and_radii() {
        for &width in &[1usize, 2, 3, 7, 8, 16, 32, 33, 64] {
            let count = if width <= 8 { 2048 } else { 768 };
            let mut records = make_records(width, count, width as u64 * 17 + 3);
            let planted = count / 3;
            let query_index = count / 2;
            let mut query = records[query_index * width..(query_index + 1) * width].to_vec();
            flip_bits(&mut query, (width * 8 / 16).max(1), 99 + width as u64);
            records[planted * width..(planted + 1) * width].copy_from_slice(&query);

            let cat_index = Catalog::with_index(&records, width).unwrap();
            let cat_linear = Catalog::new(&records, width).unwrap();
            let bits = (width * 8) as i64;
            for radius in [0, 1, 2, 4, 8, bits / 4, bits, bits + 100, -1] {
                let radius = radius.max(-1);
                let expected_first =
                    bytes_array_first_within_dist(&records, &query, radius).unwrap();
                let expected_best = bytes_array_best_within_dist(&records, &query, radius).unwrap();
                let expected_all = bytes_array_all_within_dist(&records, &query, radius).unwrap();
                assert_eq!(
                    cat_index.first_within(&query, radius).unwrap(),
                    expected_first,
                    "first width={width} r={radius}"
                );
                assert_eq!(
                    cat_index.best_within(&query, radius).unwrap(),
                    expected_best,
                    "best width={width} r={radius}"
                );
                assert_eq!(
                    cat_index.all_within(&query, radius).unwrap(),
                    expected_all,
                    "all width={width} r={radius}"
                );
                assert_eq!(
                    cat_linear.first_within(&query, radius).unwrap(),
                    expected_first
                );
                assert_eq!(
                    cat_linear.best_within(&query, radius).unwrap(),
                    expected_best
                );
                assert_eq!(cat_linear.all_within(&query, radius).unwrap(), expected_all);
            }
        }
    }

    #[test]
    fn catalog_many_variants_match_free_functions() {
        let width = 8;
        let records = make_records(width, 2048, 123);
        let mut queries = Vec::new();
        queries.extend_from_slice(&records[10 * width..11 * width]);
        let mut near = records[99 * width..100 * width].to_vec();
        flip_bits(&mut near, 2, 77);
        queries.extend_from_slice(&near);
        queries.extend_from_slice(&make_records(width, 1, 456));
        let cat = Catalog::with_index(&records, width).unwrap();
        for radius in [0, 2, 8, 64, -1] {
            assert_eq!(
                cat.first_many_within(&queries, radius).unwrap(),
                bytes_array_first_many_within_dist(&records, &queries, width, radius).unwrap()
            );
            assert_eq!(
                cat.best_many_within(&queries, radius).unwrap(),
                bytes_array_best_many_within_dist(&records, &queries, width, radius).unwrap()
            );
            assert_eq!(
                cat.all_many_within(&queries, radius).unwrap(),
                bytes_array_all_many_within_dist(&records, &queries, width, radius).unwrap()
            );
        }
    }
}
