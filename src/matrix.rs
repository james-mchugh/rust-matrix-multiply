use std::convert::{TryFrom, TryInto};
use std::fmt::Display;
use std::ops::{AddAssign, Index, IndexMut, Mul};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    data: Box<[T]>,
    rows: usize,
    cols: usize,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.offset(index)]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.offset(index)]
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = String;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
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

        let data: Vec<T> = value.into_iter().flatten().collect();

        Ok(Matrix {
            data: data.into_boxed_slice(),
            rows,
            cols,
        })
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        let data: Vec<T> = value.into_iter().flatten().collect();

        Matrix {
            data: data.into_boxed_slice(),
            rows: R,
            cols: C,
        }
    }
}

impl<T> FromStr for Matrix<T>
where
    T: FromStr,
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(row, line)| {
                line.split_whitespace()
                    .enumerate()
                    .map(|(col, tok)| {
                        tok.parse::<T>()
                            .map_err(|_| format!("parse error at row {row} col {col}"))
                    })
                    .collect::<Result<Vec<T>, _>>()
            })
            .collect::<Result<Vec<Vec<T>>, _>>()?
            .try_into()
    }
}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            let line = self.data[start..end]
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
impl<T> Matrix<T> {
    /// Calculates the single-dimensional array offset for a given two-dimensional index (row, col).
    fn offset(&self, (row, col): (usize, usize)) -> usize {
        row * self.cols + col
    }

    pub fn new(rows: usize, cols: usize) -> Self
    where
        T: Default + Clone,
    {
        let data = vec![T::default(); rows * cols].into_boxed_slice();
        Matrix { data, rows, cols }
    }
}

// Numeric operations grouped in a separate impl with focused bounds
impl<T> Matrix<T>
where
    T: Default + Copy + AddAssign + Mul<Output = T>,
{
    pub fn mul_cpu(&self, other: &Self) -> Result<Self, String> {
        if self.cols != other.rows {
            return Err("Matrix dimensions do not match".to_string());
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for row in 0..self.rows {
            for col in 0..other.cols {
                for i in 0..self.cols {
                    result[(row, col)] += self[(row, i)] * other[(i, col)];
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use crate::Matrix;

    #[test]
    fn test_mul_cpu() {
        #[rustfmt::skip]
        let matrix_a = Matrix::try_from(
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

        let result = matrix_a.mul_cpu(&matrix_b).unwrap();

        assert_eq!(result, expected)
    }
}