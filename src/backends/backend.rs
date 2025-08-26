use std::fmt::Debug;

pub trait Backend {
    type Buf;
    type Ctx;
    type Err: Debug;

    fn alloc(ctx: &Self::Ctx, size: usize) -> Result<Self::Buf, Self::Err>;
    fn upload(ctx: &Self::Ctx, host: &[f32]) -> Result<Self::Buf, Self::Err>;
    fn download(ctx: &Self::Ctx, device: &Self::Buf, host: &mut [f32]) -> Result<(), Self::Err>;

    fn gemm(
        ctx: &Self::Ctx,
        m: usize,
        k: usize,
        n: usize,
        a: &Self::Buf,
        b: &Self::Buf,
        c: &mut Self::Buf,
    ) -> Result<(), Self::Err>;
}
