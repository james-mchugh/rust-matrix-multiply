use cust::context::Context;
use cust::prelude::DeviceBuffer;
use crate::backends::Backend;

pub struct GPU;

impl Backend for GPU {
    type Buf = DeviceBuffer<f32>;
    type Ctx = Context;
    type Err = String;

    fn alloc(ctx: &Self::Ctx, size: usize) -> Result<Self::Buf, Self::Err> {
        todo!()
    }

    fn upload(ctx: &Self::Ctx, host: &[f32]) -> Result<Self::Buf, Self::Err> {
        todo!()
    }

    fn download(ctx: &Self::Ctx, device: &Self::Buf, host: &mut [f32]) -> Result<(), Self::Err> {
        todo!()
    }

    fn gemm(ctx: &Self::Ctx, m: usize, k: usize, n: usize, a: &Self::Buf, b: &Self::Buf, c: &mut Self::Buf) -> Result<(), Self::Err> {
        todo!()
    }
}