use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <model.onnx> <sample.npz>", args[0]);
        std::process::exit(1);
    }
    let onnx_path = PathBuf::from(&args[1]);
    let sample = &args[2];

    println!("loading {}", onnx_path.display());
    let t0 = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(&onnx_path)?
        .into_optimized()?
        .into_runnable()?;
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut npz = NpzReader::new(File::open(sample).context("open npz")?)?;
    let input_ids: ArrayD<i64> = npz.by_name("input_ids.npy")?;
    let audio_mask: ArrayD<bool> = npz.by_name("audio_mask.npy")?;
    let attention_mask: ArrayD<bool> = npz.by_name("attention_mask.npy")?;
    let position_ids: ArrayD<i64> = npz.by_name("position_ids.npy")?;
    println!("input_ids: {:?}", input_ids.shape());

    let inputs = tvec!(
        input_ids.into_tvalue(),
        audio_mask.into_tvalue(),
        attention_mask.into_tvalue(),
        position_ids.into_tvalue(),
    );

    println!("warmup...");
    let _ = model.run(inputs.clone())?;
    println!("timing...");
    let n = 3;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = model.run(inputs.clone())?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("forward: {:.1} ms/iter (n={})", ms, n);
    Ok(())
}
