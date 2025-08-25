use criterion::{criterion_group, criterion_main, Criterion};
use rust_matrix_multiply::{Matrix, Dot, CPU, CPUNaive};

fn optimized_cpu_benchmark(c: &mut Criterion) {
    let m1 = Matrix::<CPU>::new(1000, 1000).unwrap();
    let m2 = Matrix::<CPU>::new(1000, 1000).unwrap();
    c.bench_function("cpu optimized matrix multiply 1000x1000", |b| b.iter(|| m1.dot(&m2)));
}

fn naive_cpu_benchmark(c: &mut Criterion) {
    let m1 = Matrix::<CPUNaive>::new(1000, 1000).unwrap();
    let m2 = Matrix::<CPUNaive>::new(1000, 1000).unwrap();
    c.bench_function("cpu naive matrix multiply 1000x1000", |b| b.iter(|| m1.dot(&m2)));
}


criterion_group!(benches, optimized_cpu_benchmark, naive_cpu_benchmark);
criterion_main!(benches);
