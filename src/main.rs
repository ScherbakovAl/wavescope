mod analysis;
mod app;
mod audio;
mod colormap;
mod cwt;
mod gpu;
mod wavelet;

use app::WaveletApp;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Wavescope")
            .with_inner_size([1440.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Wavescope",
        native_options,
        Box::new(|cc| Ok(Box::new(WaveletApp::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}
