use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexhamming::{
    bytes_hamming_distance, bytes_within_dist,
    bytes_array_first_within_dist, bytes_array_best_within_dist, bytes_array_all_within_dist,
    hex_hamming_distance, set_algorithm,
};

#[cfg(target_arch = "aarch64")]
use hexhamming::hex_hamming_distance_pack;

const HEX_SIZES: [usize; 5] = [16, 32, 64, 128, 254];
const BYTE_SIZES: [usize; 5] = [8, 16, 32, 64, 127];

/// Benchmark hex hamming distance across all available algorithms
fn bench_hex_by_algo(c: &mut Criterion) {
    let algos: &[&str] = if cfg!(target_arch = "x86_64") {
        &["classic", "sse", "avx2", "avx512"]
    } else if cfg!(target_arch = "aarch64") {
        &["classic", "neon"]
    } else {
        &["classic", "native"]
    };

    for &algo in algos {
        if set_algorithm(algo).is_err() {
            continue; // skip unsupported algos on this CPU
        }
        let mut group = c.benchmark_group(format!("hex_string/{algo}"));
        for size in HEX_SIZES {
            let a = "f".repeat(size);
            let b = "0".repeat(size);
            group.bench_function(format!("{size} chars"), |bencher| {
                bencher.iter(|| hex_hamming_distance(black_box(&a), black_box(&b)))
            });
        }
        group.finish();
    }

    // Reset to auto-detect
    set_algorithm("native").ok();
}

/// Benchmark bytes hamming distance across all available algorithms
fn bench_bytes_by_algo(c: &mut Criterion) {
    let algos: &[&str] = if cfg!(target_arch = "x86_64") {
        &["classic", "sse", "avx2", "avx512"]
    } else if cfg!(target_arch = "aarch64") {
        &["classic", "neon"]
    } else {
        &["classic", "native"]
    };

    for &algo in algos {
        if set_algorithm(algo).is_err() {
            continue;
        }
        let mut group = c.benchmark_group(format!("bytes/{algo}"));
        for size in BYTE_SIZES {
            let a = vec![0xFFu8; size];
            let b = vec![0x00u8; size];
            group.bench_function(format!("{size} bytes"), |bencher| {
                bencher.iter(|| bytes_hamming_distance(black_box(&a), black_box(&b)))
            });
        }
        group.finish();
    }

    set_algorithm("native").ok();
}

/// Benchmark check_bytes_within_dist
fn bench_bytes_within_dist(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_within_dist");
    for size in BYTE_SIZES {
        let a = vec![0xFFu8; size];
        let b = vec![0x00u8; size];
        group.bench_function(format!("{size} bytes"), |bencher| {
            bencher.iter(|| bytes_within_dist(black_box(&a), black_box(&b), black_box(100)))
        });
    }
    group.finish();
}

/// Benchmark array API: first, best, all
fn bench_array_api(c: &mut Criterion) {
    // 512 elements of 16 bytes, match at position 0
    let elem_size = 16usize;
    let num_elements = 512usize;
    let small = vec![0x00u8; elem_size];
    let mut big = vec![0x03u8; elem_size * num_elements];
    big[..elem_size].copy_from_slice(&small);

    let mut group = c.benchmark_group("array_api/512x16_match_at_0");
    group.bench_function("first", |bencher| {
        bencher.iter(|| bytes_array_first_within_dist(black_box(&big), black_box(&small), black_box(1)))
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| bytes_array_best_within_dist(black_box(&big), black_box(&small), black_box(1)))
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| bytes_array_all_within_dist(black_box(&big), black_box(&small), black_box(1)))
    });
    group.finish();

    // Match at end
    let mut big_end = vec![0x03u8; elem_size * num_elements];
    big_end[(num_elements - 1) * elem_size..].copy_from_slice(&small);

    let mut group = c.benchmark_group("array_api/512x16_match_at_end");
    group.bench_function("first", |bencher| {
        bencher.iter(|| bytes_array_first_within_dist(black_box(&big_end), black_box(&small), black_box(1)))
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| bytes_array_best_within_dist(black_box(&big_end), black_box(&small), black_box(1)))
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| bytes_array_all_within_dist(black_box(&big_end), black_box(&small), black_box(1)))
    });
    group.finish();

    // 16384 elements of 64 bytes, match at middle
    let elem_size_lg = 64usize;
    let num_elements_lg = 16384usize;
    let small_lg = vec![0x00u8; elem_size_lg];
    let mut big_lg = vec![0x03u8; elem_size_lg * num_elements_lg];
    let mid = num_elements_lg / 2;
    big_lg[mid * elem_size_lg..(mid + 1) * elem_size_lg].copy_from_slice(&small_lg);

    let mut group = c.benchmark_group("array_api/16384x64_match_at_mid");
    group.bench_function("first", |bencher| {
        bencher.iter(|| bytes_array_first_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1)))
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| bytes_array_best_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1)))
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| bytes_array_all_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1)))
    });
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_hex_string_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("hex_string/neon_pack");
    for size in [32, 64, 128, 254] {
        let a = "f".repeat(size);
        let b = "0".repeat(size);
        group.bench_function(format!("{size} chars"), |bencher| {
            bencher.iter(|| hex_hamming_distance_pack(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
criterion_group!(benches, bench_hex_by_algo, bench_bytes_by_algo, bench_bytes_within_dist, bench_array_api, bench_hex_string_pack);
#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, bench_hex_by_algo, bench_bytes_by_algo, bench_bytes_within_dist, bench_array_api);
criterion_main!(benches);
