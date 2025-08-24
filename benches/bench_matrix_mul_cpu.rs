use criterion::{criterion_group, criterion_main, Criterion};
use rust_matrix_multiply::{Matrix, Dot, CPU};

fn criterion_benchmark(c: &mut Criterion) {
    let m1 = Matrix::<CPU>::new(100, 100).unwrap();
    let m2 = Matrix::<CPU>::new(100, 100).unwrap();
    c.bench_function("matrix multiply 100x100", |b| b.iter(|| m1.dot(&m2)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
