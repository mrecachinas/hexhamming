use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexhamming::{
    bytes_array_all_within_dist, bytes_array_best_within_dist, bytes_array_first_within_dist,
    bytes_hamming_distance, bytes_within_dist, hex_hamming_distance, set_algorithm,
};

#[cfg(target_arch = "aarch64")]
use hexhamming::hex_hamming_distance_pack;

// Hex sizes are character counts; byte sizes are the corresponding decoded lengths.
const HEX_SIZES: [usize; 5] = [16, 32, 64, 128, 254];
const BYTE_SIZES: [usize; 5] = [8, 16, 32, 64, 127];

fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((value ^ (value >> 31)) >> 56) as u8
        })
        .collect()
}

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
        // Each group measures one implementation on max-distance hex pairs.
        let mut group = c.benchmark_group(format!("hex_string/{algo}"));
        for size in HEX_SIZES {
            // "f" vs "0" makes every nibble differ, stressing the bit-count path.
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
        &["classic", "native", "neon"]
    } else {
        &["classic", "native"]
    };

    for &algo in algos {
        if set_algorithm(algo).is_err() {
            continue;
        }
        // Raw-byte groups isolate byte Hamming distance from hex parsing costs.
        let mut group = c.benchmark_group(format!("bytes/{algo}"));
        for size in BYTE_SIZES {
            // 0xFF vs 0x00 makes every bit differ for each byte length.
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
    // Measures the threshold predicate over byte arrays as distances cross 100 bits.
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
    // The large buffer is a contiguous catalog of fixed-size byte entries; small is the query.
    // 0x03 differs from 0x00 by two bits per byte, while copied entries are exact matches.
    // These groups compare finding the first match, best match, or all matches within distance 1.
    // 512 elements of 16 bytes, match at position 0
    let elem_size = 16usize;
    let num_elements = 512usize;
    let small = vec![0x00u8; elem_size];
    let mut big = vec![0x03u8; elem_size * num_elements];
    big[..elem_size].copy_from_slice(&small);

    let mut group = c.benchmark_group("array_api/512x16_match_at_0");
    group.bench_function("first", |bencher| {
        bencher.iter(|| {
            bytes_array_first_within_dist(black_box(&big), black_box(&small), black_box(1))
        })
    });
    group.bench_function("best", |bencher| {
        bencher
            .iter(|| bytes_array_best_within_dist(black_box(&big), black_box(&small), black_box(1)))
    });
    group.bench_function("all", |bencher| {
        bencher
            .iter(|| bytes_array_all_within_dist(black_box(&big), black_box(&small), black_box(1)))
    });
    group.finish();

    // Match at end
    let mut big_end = vec![0x03u8; elem_size * num_elements];
    big_end[(num_elements - 1) * elem_size..].copy_from_slice(&small);

    let mut group = c.benchmark_group("array_api/512x16_match_at_end");
    group.bench_function("first", |bencher| {
        bencher.iter(|| {
            bytes_array_first_within_dist(black_box(&big_end), black_box(&small), black_box(1))
        })
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| {
            bytes_array_best_within_dist(black_box(&big_end), black_box(&small), black_box(1))
        })
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| {
            bytes_array_all_within_dist(black_box(&big_end), black_box(&small), black_box(1))
        })
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
        bencher.iter(|| {
            bytes_array_first_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1))
        })
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| {
            bytes_array_best_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1))
        })
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| {
            bytes_array_all_within_dist(black_box(&big_lg), black_box(&small_lg), black_box(1))
        })
    });
    group.finish();

    // 100_000 elements of 128 bytes — large batch to showcase parallel speedup
    let elem_size_xl = 128usize;
    let num_elements_xl = 100_000usize;
    let small_xl = vec![0x00u8; elem_size_xl];
    let mut big_xl = vec![0x03u8; elem_size_xl * num_elements_xl];
    // Scatter matches at various positions
    for &idx in &[100, 5_000, 25_000, 50_000, 75_000, 99_999] {
        big_xl[idx * elem_size_xl..(idx + 1) * elem_size_xl].copy_from_slice(&small_xl);
    }

    let mut group = c.benchmark_group("array_api/100000x128_parallel");
    group.sample_size(10);
    group.bench_function("first", |bencher| {
        bencher.iter(|| {
            bytes_array_first_within_dist(black_box(&big_xl), black_box(&small_xl), black_box(1))
        })
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| {
            bytes_array_best_within_dist(black_box(&big_xl), black_box(&small_xl), black_box(1))
        })
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| {
            bytes_array_all_within_dist(black_box(&big_xl), black_box(&small_xl), black_box(1))
        })
    });
    group.finish();
}

fn bench_array_random_and_boundaries(c: &mut Criterion) {
    let small_16 = pseudo_random_bytes(16, 1);
    let big_512x16 = pseudo_random_bytes(512 * 16, 2);
    let mut group = c.benchmark_group("array_api/512x16_random_no_match");
    group.bench_function("first", |bencher| {
        bencher.iter(|| {
            bytes_array_first_within_dist(
                black_box(&big_512x16),
                black_box(&small_16),
                black_box(0),
            )
        })
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| {
            bytes_array_best_within_dist(black_box(&big_512x16), black_box(&small_16), black_box(0))
        })
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| {
            bytes_array_all_within_dist(black_box(&big_512x16), black_box(&small_16), black_box(0))
        })
    });
    group.finish();

    let small_64 = pseudo_random_bytes(64, 3);
    let big_16384x64 = pseudo_random_bytes(16_384 * 64, 4);
    let mut group = c.benchmark_group("array_api/16384x64_random_no_match");
    group.sample_size(30);
    group.bench_function("first", |bencher| {
        bencher.iter(|| {
            bytes_array_first_within_dist(
                black_box(&big_16384x64),
                black_box(&small_64),
                black_box(0),
            )
        })
    });
    group.bench_function("best", |bencher| {
        bencher.iter(|| {
            bytes_array_best_within_dist(
                black_box(&big_16384x64),
                black_box(&small_64),
                black_box(0),
            )
        })
    });
    group.bench_function("all", |bencher| {
        bencher.iter(|| {
            bytes_array_all_within_dist(
                black_box(&big_16384x64),
                black_box(&small_64),
                black_box(0),
            )
        })
    });
    group.finish();

    let mut group = c.benchmark_group("array_api/parallel_boundary_64");
    group.sample_size(30);
    for num_elements in [4095usize, 4096, 65_536, 81_920, 131_072] {
        let big = pseudo_random_bytes(num_elements * 64, 10 + num_elements as u64);
        group.bench_function(format!("{num_elements} elements/all"), |bencher| {
            bencher.iter(|| {
                bytes_array_all_within_dist(black_box(&big), black_box(&small_64), black_box(0))
            })
        });
        if num_elements >= 65_536 {
            group.bench_function(format!("{num_elements} elements/best"), |bencher| {
                bencher.iter(|| {
                    bytes_array_best_within_dist(
                        black_box(&big),
                        black_box(&small_64),
                        black_box(0),
                    )
                })
            });
        }
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_hex_string_pack(c: &mut Criterion) {
    // AArch64-only group for the packed NEON hex-string path.
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
criterion_group!(
    benches,
    bench_hex_by_algo,
    bench_bytes_by_algo,
    bench_bytes_within_dist,
    bench_array_api,
    bench_array_random_and_boundaries,
    bench_hex_string_pack
);
#[cfg(not(target_arch = "aarch64"))]
criterion_group!(
    benches,
    bench_hex_by_algo,
    bench_bytes_by_algo,
    bench_bytes_within_dist,
    bench_array_api,
    bench_array_random_and_boundaries
);
criterion_main!(benches);
