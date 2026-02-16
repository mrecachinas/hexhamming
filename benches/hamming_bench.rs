use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexhamming::{bytes_hamming_distance, hex_hamming_distance, set_algorithm};

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
criterion_group!(benches, bench_hex_by_algo, bench_bytes_by_algo, bench_hex_string_pack);
#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, bench_hex_by_algo, bench_bytes_by_algo);
criterion_main!(benches);
