use rust_matrix_multiply::Matrix;
use std::fs::read_to_string;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (Some(path_a), Some(path_b)) = (args.next(), args.next()) else {
        Err("Missing arguments".to_string())?
    };

    let matrix_a = read_to_string(path_a)
        .map_err(|e| e.to_string())?
        .parse::<Matrix<f32>>()?;
    let matrix_b = read_to_string(path_b)
        .map_err(|e| e.to_string())?
        .parse::<Matrix<f32>>()?;
    let result = matrix_a.mul_cpu(&matrix_b)?;

    println!("{}", result);

    Ok(())
}
