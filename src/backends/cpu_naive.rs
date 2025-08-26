use crate::backends::Backend;

#[derive(Debug)]
pub struct CPUNaive;

// Numeric operations grouped in a separate impl with focused bounds
impl Backend for CPUNaive {
    type Buf = Vec<f32>;
    type Ctx = ();
    type Err = String;

    fn alloc(_ctx: &Self::Ctx, size: usize) -> Result<Self::Buf, Self::Err> {
        Ok(vec![0.0; size])
    }

    fn upload(_ctx: &Self::Ctx, host: &[f32]) -> Result<Self::Buf, Self::Err> {
        Ok(host.to_vec())
    }

    fn download(_ctx: &Self::Ctx, device: &Self::Buf, host: &mut [f32]) -> Result<(), Self::Err> {
        host.copy_from_slice(device);
        Ok(())
    }

    fn gemm(
        _ctx: &Self::Ctx,
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

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k {
                    sum += a[i * k + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }
}
