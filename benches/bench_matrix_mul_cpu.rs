use criterion::{criterion_group, criterion_main, Criterion};
use rust_matrix_multiply::Matrix;

fn criterion_benchmark(c: &mut Criterion) {
    let m1 = Matrix::<f32>::new(100, 100);
    let m2 = Matrix::<f32>::new(100, 100);
    c.bench_function("matrix multiply 100x100", |b| b.iter(|| m1.mul_cpu(&m2)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
