use std::fmt::Display;
use std::fs::read_to_string;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

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
