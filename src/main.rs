use std::fmt::Display;
use std::fs::read_to_string;
use std::ops::{Index, IndexMut};
use std::str::FromStr;



#[derive(Debug, Clone, PartialEq, Default)]
struct Matrix {
    data: Box<[f32]>,
    rows: usize,
    cols: usize,
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.offset(index)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.offset(index)]
    }
}

impl TryFrom<Vec<Vec<f32>>> for Matrix {
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

        Ok(Matrix {
            data: data.into_boxed_slice(),
            rows,
            cols,
        })
    }
}

impl<const R: usize, const C: usize> TryFrom<[[f32; C]; R]> for Matrix {
    type Error = String;

    fn try_from(value: [[f32; C]; R]) -> Result<Self, Self::Error> {
        if R == 0 || C == 0 {
            return Err("Matrix is empty".to_string());
        }

        let data: Vec<f32> = value.into_iter().flatten().collect();

        Ok(Matrix {
            data: data.into_boxed_slice(),
            rows: R,
            cols: C,
        })
    }
}


impl FromStr for Matrix {
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
                            .map_err(|e| format!("parse error at row {row} col {col}: {e}"))
                    })
                    .collect::<Result<Vec<f32>, _>>()
            })
            .collect::<Result<Vec<Vec<f32>>, _>>()?
            .try_into()
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            let line = self.data[start..end]
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

impl Matrix {
    /// Calculates the single-dimensional array offset for a given two-dimensional index (row, col).
    ///
    /// This function is used to map a 2D index into a 1D array offset, assuming that the array
    /// elements are stored in row-major order. It multiplies the specified row index by the
    /// number of columns (`self.cols`) and adds the column index to determine the offset.
    ///
    /// Row major order is used as it is the default choice for C and CUDA.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0-based) of the element in the 2D representation.
    /// * `col` - The column index (0-based) of the element in the 2D representation.
    ///
    /// # Returns
    ///
    /// The computed offset as a `usize` representing the position of the element in a
    /// flat 1-dimensional array.
    ///
    fn offset(&self, (row, col): (usize, usize)) -> usize {
        row * self.cols + col
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols].into_boxed_slice();
        Matrix { data, rows, cols }
    }

    pub fn mul_cpu(&self, other: &Matrix) -> Result<Matrix, String> {
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

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (Some(path_a), Some(path_b)) = (args.next(), args.next()) else {
        Err("Missing arguments".to_string())?
    };

    let matrix_a = read_to_string(path_a)
        .map_err(|e| e.to_string())?
        .parse::<Matrix>()?;
    let matrix_b = read_to_string(path_b)
        .map_err(|e| e.to_string())?
        .parse::<Matrix>()?;
    let result = matrix_a.mul_cpu(&matrix_b)?;

    println!("{}", result);

    Ok(())
}
