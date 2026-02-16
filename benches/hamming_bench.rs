use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexhamming::{bytes_hamming_distance, hex_hamming_distance};

fn bench_hex_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("hex_hamming_distance");
    for size in [16, 32, 64, 128, 254] {
        let a = "f".repeat(size);
        let b = "0".repeat(size);
        group.bench_function(format!("{size} chars"), |bencher| {
            bencher.iter(|| hex_hamming_distance(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_hamming_distance");
    for size in [8, 16, 32, 64, 127] {
        let a = vec![0xFFu8; size];
        let b = vec![0x00u8; size];
        group.bench_function(format!("{size} bytes"), |bencher| {
            bencher.iter(|| bytes_hamming_distance(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hex_string, bench_bytes);
criterion_main!(benches);
