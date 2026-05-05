use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use openvino::{Core, ElementType, Shape, Tensor};
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn read_npz(path: &str) -> Result<(
    ArrayD<i64>, ArrayD<bool>, ArrayD<bool>, ArrayD<i64>,
)> {
    let mut npz = NpzReader::new(File::open(path).context("open npz")?)?;
    let input_ids: ArrayD<i64> = npz.by_name("input_ids.npy")?;
    let audio_mask: ArrayD<bool> = npz.by_name("audio_mask.npy")?;
    let attention_mask: ArrayD<bool> = npz.by_name("attention_mask.npy")?;
    let position_ids: ArrayD<i64> = npz.by_name("position_ids.npy")?;
    Ok((input_ids, audio_mask, attention_mask, position_ids))
}

fn arr_to_tensor_i64(a: &ArrayD<i64>) -> Result<Tensor> {
    let shape: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
    let mut t = Tensor::new(ElementType::I64, &Shape::new(&shape)?)?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            a.as_slice().context("contig")?.as_ptr() as *const u8,
            a.len() * std::mem::size_of::<i64>(),
        )
    };
    t.get_raw_data_mut()?.copy_from_slice(bytes);
    Ok(t)
}

fn arr_to_tensor_bool(a: &ArrayD<bool>) -> Result<Tensor> {
    let shape: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
    let mut t = Tensor::new(ElementType::Boolean, &Shape::new(&shape)?)?;
    let raw = t.get_raw_data_mut()?;
    for (i, &b) in a.as_slice().context("contig")?.iter().enumerate() {
        raw[i] = if b { 1 } else { 0 };
    }
    Ok(t)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <ir.xml> <sample.npz> [threads]", args[0]);
        std::process::exit(1);
    }
    let ir = PathBuf::from(&args[1]);
    let bin = ir.with_extension("bin");
    let sample = &args[2];
    let threads: i32 = args.get(3).map(|s| s.parse().unwrap_or(8)).unwrap_or(8);

    let mut core = Core::new()?;
    println!("OpenVINO devices: {:?}", core.available_devices()?);

    let mut model = core.read_model_from_file(
        ir.to_str().context("ir path")?,
        bin.to_str().context("bin path")?,
    )?;
    println!("Loaded IR: {}", ir.display());

    let mut compiled = core.compile_model(&mut model, "CPU".into())?;

    let (input_ids, audio_mask, attention_mask, position_ids) = read_npz(sample)?;
    println!(
        "sample shapes: input_ids={:?}, audio_mask={:?}, attention_mask={:?}, position_ids={:?}",
        input_ids.shape(), audio_mask.shape(), attention_mask.shape(), position_ids.shape()
    );

    let mut req = compiled.create_infer_request()?;
    req.set_tensor("input_ids", &arr_to_tensor_i64(&input_ids)?)?;
    req.set_tensor("audio_mask", &arr_to_tensor_bool(&audio_mask)?)?;
    req.set_tensor("attention_mask", &arr_to_tensor_bool(&attention_mask)?)?;
    req.set_tensor("position_ids", &arr_to_tensor_i64(&position_ids)?)?;

    let _ = threads; // not directly settable via this API path

    // Warmup
    req.infer()?;
    req.infer()?;

    let n = 5;
    let t0 = Instant::now();
    for _ in 0..n {
        req.infer()?;
    }
    let elapsed = t0.elapsed();
    let ms_per = elapsed.as_secs_f64() * 1000.0 / n as f64;
    println!("forward: {:.1} ms/iter (n={})", ms_per, n);

    let out = req.get_tensor("logits")?;
    let shape = out.get_shape()?;
    println!("logits dims: {:?}", shape.get_dimensions());

    Ok(())
}
