use crate::backends::Backend;

#[derive(Debug)]
pub struct CPU;

// Numeric operations grouped in a separate impl with focused bounds
impl Backend for CPU {
    type Buf = Vec<f32>;
    type Ctx = ();
    type Err = String;

    fn alloc(size: usize) -> Result<Self::Buf, Self::Err> {
        Ok(vec![0.0; size])
    }

    fn upload(host: &[f32]) -> Result<Self::Buf, Self::Err> {
        Ok(host.to_vec())
    }

    fn download(device: &Self::Buf, host: &mut [f32]) -> Result<(), Self::Err> {
        host.copy_from_slice(device);
        Ok(())
    }

    fn gemm(
        m: usize,
        k: usize,
        n: usize,
        a: &Self::Buf,
        b: &Self::Buf,
        c: &mut Self::Buf,
    ) -> Result<(), Self::Err> {
        if n != k {
            return Err("Matrix dimensions do not match".to_string());
        }

        // transpose b to be in column major order (this ensures memory access is contiguous)
        // K x N -> N x K
        let mut bt = b.clone();
        for row in 0..k {
            let b_row_start = row * n;
            for col in 0..n {
                bt[col * k + row] = b[b_row_start + col]
            }
        }

        const TILE_SIZE: usize = 8;

        for i in 0..m {
            let row_a = &a[i * k..(i + 1) * k];
            let c_row_start = i * n;
            for j in 0..n {
                let mut sum = 0.0;
                for t in (0..k).step_by(TILE_SIZE) {
                    let end = (t + TILE_SIZE).min(k);
                    let a_tile = &row_a[t..end];
                    let bt_tile = &bt[j * k + t..j * k + end];
                    sum += a_tile.iter().zip(bt_tile).map(|(a, b)| a * b).sum::<f32>();
                }
                c[c_row_start + j] = sum;
            }
        }

        Ok(())
    }
}