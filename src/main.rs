mod app;
mod audio;
mod colormap;
mod cuda;
mod cwt;
mod wavelet;

use app::WaveletApp;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // -----------------------------------------------------------------------
    // 1. Compile CUDA kernel at startup
    // -----------------------------------------------------------------------
    const KERNEL_SRC: &str = include_str!("kernel.cu");
    let ptx_code = compile_cuda_kernel(KERNEL_SRC)?;
    log::info!("CUDA kernel compiled successfully ({} bytes PTX)", ptx_code.len());

    // -----------------------------------------------------------------------
    // 2. Launch egui application
    // -----------------------------------------------------------------------
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Wavelet Audio Analyzer")
            .with_inner_size([1440.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Wavelet Audio Analyzer",
        native_options,
        Box::new(move |cc| Ok(Box::new(WaveletApp::new(cc, ptx_code)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Runtime CUDA kernel compilation
// ---------------------------------------------------------------------------

fn compile_cuda_kernel(src: &str) -> anyhow::Result<String> {
    use std::process::Command;

    let cuda_root = std::env::var("CUDA_ROOT")
        .unwrap_or_else(|_| "/usr/local/cuda-13.1".to_string());
    let nvcc = format!("{}/bin/nvcc", cuda_root);

    let tmp   = std::env::temp_dir();
    let cu_p  = tmp.join("wavelet_kernel.cu");
    let ptx_p = tmp.join("wavelet_kernel.ptx");

    std::fs::write(&cu_p, src)?;

    let out = Command::new(&nvcc)
        .args([
            "--gpu-architecture=sm_75",
            "--allow-unsupported-compiler",
            "-std=c++14",
            "--use_fast_math",
            "-O2",
            "--ptx",
            "-o", ptx_p.to_str().unwrap(),
            cu_p.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run nvcc ({}): {}", nvcc, e))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("nvcc compilation failed:\n{}", stderr);
    }

    // Print any warnings
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.is_empty() {
        log::warn!("nvcc warnings:\n{}", stderr);
    }

    Ok(std::fs::read_to_string(&ptx_p)?)
}
