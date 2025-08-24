use std::convert::{TryFrom, TryInto};
use std::fmt::Display;
use std::str::FromStr;

use crate::backends::Backend;

#[derive(Debug)]
pub struct Matrix<B: Backend> {
    buf: B::Buf,
    rows: usize,
    cols: usize,
    _b: std::marker::PhantomData<B>,
}

pub trait Dot {
    type Err;
    type Output;
    fn dot(&self, other: &Self) -> Result<Self::Output, Self::Err>;
}

impl<B: Backend> Matrix<B> {
    pub fn new_zeros(rows: usize, cols: usize) -> Result<Self, B::Err> {
        let buf = B::alloc(rows * cols)?;
        Ok(Self {
            buf,
            rows,
            cols,
            _b: std::marker::PhantomData,
        })
    }

    pub fn from_host(rows: usize, cols: usize, data: &[f32]) -> Result<Self, B::Err> {
        assert_eq!(data.len(), rows * cols);
        let buf = B::upload(data)?;
        Ok(Self {
            buf,
            rows,
            cols,
            _b: std::marker::PhantomData,
        })
    }

    pub fn to_host(&self) -> Result<Vec<f32>, B::Err> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        B::download(&self.buf, &mut out)?;
        Ok(out)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<B: Backend> TryFrom<Vec<Vec<f32>>> for Matrix<B> {
    type Error = String;

    fn try_from(value: Vec<Vec<f32>>) -> Result<Self, Self::Error> {
        let rows = value.len();
        if rows == 0 {
            return Err("Matrix is empty".to_string());
        }
        let cols = value[0].len();
        if cols == 0 {
            return Err("Matrix is empty".to_string());
        }

        if !value.iter().all(|row| row.len() == cols) {
            return Err("Matrix dimensions do not match".to_string());
        }

        let data: Vec<f32> = value.into_iter().flatten().collect();

        // todo fix error message
        Matrix::from_host(rows, cols, &data).map_err(|_| "Matrix allocation failed".to_string())
    }
}

impl<B: Backend, const R: usize, const C: usize> TryFrom<[[f32; C]; R]> for Matrix<B> {
    type Error = B::Err;

    fn try_from(value: [[f32; C]; R]) -> Result<Self, Self::Error> {
        let flat_data: Vec<f32> = value.iter().flatten().copied().collect();
        Matrix::from_host(R, C, &flat_data)
    }
}

impl<B: Backend> FromStr for Matrix<B> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(row, line)| {
                line.split_whitespace()
                    .enumerate()
                    .map(|(col, tok)| {
                        tok.parse::<f32>()
                            .map_err(|_| format!("parse error at row {row} col {col}"))
                    })
                    .collect::<Result<Vec<f32>, _>>()
            })
            .collect::<Result<Vec<Vec<f32>>, _>>()?
            .try_into()
    }
}

impl<B: Backend> Display for Matrix<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_host().map_err(|_| std::fmt::Error)?;
        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            let line = data[start..end]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

// General inherent methods that don't need "numeric" bounds
impl<B: Backend> Matrix<B> {
    pub fn new(rows: usize, cols: usize) -> Result<Self, B::Err> {
        Matrix::new_zeros(rows, cols)
    }
}

// Numeric operations grouped in a separate impl with focused bounds
impl<B: Backend> Dot for Matrix<B> {
    type Err = String;
    type Output = Self;

    fn dot(&self, other: &Self) -> Result<Self, Self::Err> {
        if self.cols != other.rows {
            return Err("Matrix dimensions do not match".to_string());
        }

        // todo: fix error message
        let mut c =
            Self::new_zeros(self.cols, self.rows).map_err(|_| "Matrix allocation failed")?;

        // todo: fix error message
        B::gemm(
            self.rows, self.cols, other.cols, &self.buf, &other.buf, &mut c.buf,
        )
        .map_err(|_| "Matrix multiplication failed")?;

        Ok(c)
    }
}

impl<B: Backend> PartialEq for Matrix<B> {
    fn eq(&self, other: &Self) -> bool {
        self.to_host().unwrap() == other.to_host().unwrap()
    }
}

// todo: move these tests to a separate file
#[cfg(test)]
mod test {
    use crate::backends::CPU;
    use crate::matrix::Dot;
    use crate::Matrix;

    #[test]
    fn test_mul_cpu() {
        #[rustfmt::skip]
        let matrix_a = Matrix::<CPU>::try_from(
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ]
        ).unwrap();

        #[rustfmt::skip]
        let matrix_b = Matrix::try_from(
            [
                [5.0, 6.0],
                [7.0, 8.0]
            ]
        ).unwrap();

        #[rustfmt::skip]
        let expected = Matrix::try_from(
            [
                [19.0, 22.0],
                [43.0, 50.0]
            ]
        ).unwrap();

        let result = matrix_a.dot(&matrix_b).unwrap();

        assert_eq!(result, expected)
    }
}
