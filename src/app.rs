use std::path::Path;
use std::sync::{Arc, mpsc};

use egui::TextureHandle;

use crate::audio::AudioFile;
use crate::colormap::{ColorMap, scalogram_to_rgba};
use crate::cuda::CudaContext;
use crate::cwt::CwtEngine;
use crate::wavelet::CwtParams;

// ---------------------------------------------------------------------------
// Worker thread messages
// ---------------------------------------------------------------------------

pub enum WorkerMsg {
    LoadFile { path: String },
    Compute(ComputeRequest),
}

pub struct ComputeRequest {
    pub channels:     Vec<Vec<f32>>,
    pub sample_rate:  u32,
    pub t_start:      usize,
    pub t_end:        usize,
    pub width_pixels: usize,
    pub params:       CwtParams,
}

pub enum WorkerResult {
    Loaded(AudioFile),
    Computed(Vec<Vec<f32>>),   // one scalogram per channel
    Error(String),
}

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Viewport {
    pub t_start: f64,   // visible window start in sample indices
    pub t_end:   f64,   // visible window end   in sample indices
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

pub struct WaveletApp {
    // Audio
    audio:             Option<AudioFile>,

    // Scalogram data  [scale * width + col], one per channel
    scalograms:        Vec<Vec<f32>>,
    scalogram_width:   usize,
    textures:          Vec<Option<TextureHandle>>,

    // Viewport & parameters
    viewport:   Viewport,
    params:     CwtParams,
    colormap:   ColorMap,
    log_scale:  bool,

    // Worker thread communication
    work_tx:    mpsc::SyncSender<WorkerMsg>,
    result_rx:  mpsc::Receiver<WorkerResult>,

    // UI state
    computing:         bool,
    pending_compute:   bool,
    last_width_px:     usize,
    error_msg:         Option<String>,
    status:            String,
    hover_info:        Option<(f64, f64)>,  // (time_s, freq_hz)
}

impl WaveletApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, ptx_code: String) -> Self {
        let (work_tx, work_rx) = mpsc::sync_channel::<WorkerMsg>(8);
        let (res_tx,  res_rx)  = mpsc::sync_channel::<WorkerResult>(8);

        std::thread::spawn(move || worker_thread(work_rx, res_tx, ptx_code));

        WaveletApp {
            audio:           None,
            scalograms:      Vec::new(),
            scalogram_width: 0,
            textures:        Vec::new(),
            viewport:        Viewport { t_start: 0.0, t_end: 1.0 },
            params:          CwtParams::default(),
            colormap:        ColorMap::Plasma,
            log_scale:       true,
            work_tx,
            result_rx:       res_rx,
            computing:       false,
            pending_compute: false,
            last_width_px:   1024,
            error_msg:       None,
            status:          "Open an audio file to begin.".into(),
            hover_info:      None,
        }
    }

    // -----------------------------------------------------------------------
    // Trigger a compute with the current viewport / params
    // -----------------------------------------------------------------------
    fn trigger_compute(&mut self, width_px: usize) {
        let audio = match &self.audio { Some(a) => a, None => return };
        let t0 = self.viewport.t_start as usize;
        let t1 = (self.viewport.t_end as usize).min(audio.num_samples());
        if t0 >= t1 { return; }
        let req = ComputeRequest {
            channels:     audio.channels.clone(),
            sample_rate:  audio.sample_rate,
            t_start:      t0,
            t_end:        t1,
            width_pixels: width_px,
            params:       self.params.clone(),
        };
        let _ = self.work_tx.try_send(WorkerMsg::Compute(req));
        self.computing       = true;
        self.pending_compute = false;
        self.status          = "Computing…".into();
    }

    // -----------------------------------------------------------------------
    // Rebuild egui textures from scalogram data
    // -----------------------------------------------------------------------
    fn rebuild_textures(&mut self, ctx: &egui::Context) {
        let width      = self.scalogram_width;
        let num_scales = self.params.num_scales;
        self.textures.clear();
        for sc in &self.scalograms {
            let rgba = scalogram_to_rgba(sc, width, num_scales, self.colormap, self.log_scale);
            let img  = egui::ColorImage::from_rgba_unmultiplied([width, num_scales], &rgba);
            let tex  = ctx.load_texture("scalogram", img, egui::TextureOptions::LINEAR);
            self.textures.push(Some(tex));
        }
    }

    // -----------------------------------------------------------------------
    // Side-panel controls
    // -----------------------------------------------------------------------
    fn show_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Wavelet Analyzer");
        ui.separator();

        // File open
        if ui.button("📂 Open file…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Audio", &["wav", "flac", "wave"])
                .pick_file()
            {
                let _ = self.work_tx.try_send(WorkerMsg::LoadFile {
                    path: path.to_string_lossy().into_owned(),
                });
                self.computing = true;
                self.status    = "Loading…".into();
                self.error_msg = None;
            }
        }

        // File info
        if let Some(a) = &self.audio {
            ui.separator();
            let fname = Path::new(&a.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            ui.label(format!("File:     {}", fname));
            ui.label(format!("Rate:     {} Hz", a.sample_rate));
            ui.label(format!("Channels: {}", a.channels.len()));
            ui.label(format!("Duration: {:.2} s", a.duration_secs));
        }

        ui.separator();
        ui.heading("CWT Parameters");

        let mut changed = false;

        ui.label("ω₀ (Morlet):");
        changed |= ui.add(
            egui::Slider::new(&mut self.params.omega0, 4.0..=12.0).text("ω₀"),
        ).changed();

        ui.label("Scales:");
        let mut ns = self.params.num_scales as u32;
        if ui.add(egui::Slider::new(&mut ns, 64..=512).text("n")).changed() {
            self.params.num_scales = ns as usize;
            changed = true;
        }

        ui.label("f min (Hz):");
        changed |= ui.add(
            egui::Slider::new(&mut self.params.f_min, 1.0..=1000.0)
                .logarithmic(true)
                .text("Hz"),
        ).changed();

        ui.label("f max (Hz):");
        changed |= ui.add(
            egui::Slider::new(&mut self.params.f_max, 500.0..=22050.0)
                .logarithmic(true)
                .text("Hz"),
        ).changed();

        ui.separator();
        ui.heading("Display");

        egui::ComboBox::from_label("Colormap")
            .selected_text(self.colormap.name())
            .show_ui(ui, |ui| {
                for cm in [
                    ColorMap::Plasma, ColorMap::Viridis,
                    ColorMap::Magma,  ColorMap::Inferno, ColorMap::Hot,
                ] {
                    if ui.selectable_value(&mut self.colormap, cm, cm.name()).changed() {
                        let w = self.scalogram_width;
                        if w > 0 { self.rebuild_textures(ctx); }
                    }
                }
            });

        if ui.checkbox(&mut self.log_scale, "Log amplitude").changed() {
            let w = self.scalogram_width;
            if w > 0 { self.rebuild_textures(ctx); }
        }

        ui.separator();

        if ui.button("⟳ Recompute").clicked() || (changed && self.audio.is_some()) {
            if self.computing {
                self.pending_compute = true;
            } else {
                self.trigger_compute(self.last_width_px);
            }
        }

        ui.separator();
        ui.label("Interactions:");
        ui.label("Scroll        – zoom time axis");
        ui.label("Ctrl+Scroll   – zoom freq axis");
        ui.label("Drag          – pan");
        ui.label("Right-click   – reset view");

        // Status / spinner at bottom
        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
            if let Some(e) = &self.error_msg {
                ui.colored_label(egui::Color32::RED, e);
            } else {
                ui.label(&self.status.clone());
            }
            if self.computing { ui.spinner(); }
            if let Some((t, f)) = self.hover_info {
                ui.label(format!("t = {:.3} s  |  f = {:.1} Hz", t, f));
            }
        });
    }

    // -----------------------------------------------------------------------
    // Central panel – scalogram display
    // -----------------------------------------------------------------------
    fn show_scalograms(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let available    = ui.available_size();
        let render_width = (available.x - 4.0).max(64.0) as usize;

        // Detect significant width change → trigger recompute
        if render_width != self.last_width_px
            && self.last_width_px > 0
            && !self.computing
            && self.audio.is_some()
        {
            let diff = (render_width as isize - self.last_width_px as isize).unsigned_abs();
            if diff > 32 {
                self.last_width_px = render_width;
                self.trigger_compute(render_width);
            }
        }
        self.last_width_px = render_width;

        if self.audio.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    "Open a WAV or FLAC audio file to begin.\n\n\
                     Scroll to zoom time • Ctrl+Scroll to zoom frequency\n\
                     Drag to pan • Right-click to reset view",
                );
            });
            return;
        }

        let audio        = self.audio.as_ref().unwrap();
        let num_ch       = audio.channels.len();
        let total_samp   = audio.num_samples();
        let sr           = audio.sample_rate as f64;
        let _num_scales  = self.params.num_scales;
        let f_min        = self.params.f_min as f64;
        let f_max        = self.params.f_max as f64;

        let ch_height = ((available.y / num_ch.max(1) as f32) - 30.0).max(80.0);

        self.hover_info = None;
        let mut viewport_changed = false;
        let mut new_viewport     = self.viewport.clone();
        let mut new_f_min        = self.params.f_min;
        let mut new_f_max        = self.params.f_max;

        for ch in 0..num_ch {
            let label = match num_ch {
                1 => "Mono".to_string(),
                2 => if ch == 0 { "L".to_string() } else { "R".to_string() },
                _ => format!("Ch {}", ch + 1),
            };
            ui.label(label);

            let img_size = egui::vec2(available.x - 4.0, ch_height);
            let (rect, response) =
                ui.allocate_exact_size(img_size, egui::Sense::click_and_drag());

            // Background
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            ui.painter().rect_stroke(
                rect, 0.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            );

            // Draw scalogram texture (if available)
            if ch < self.textures.len() {
                if let Some(tex) = &self.textures[ch] {
                    ui.painter().image(
                        tex.id(),
                        rect,
                        egui::Rect::from_min_max(
                            egui::pos2(0.0, 0.0),
                            egui::pos2(1.0, 1.0),
                        ),
                        egui::Color32::WHITE,
                    );
                }
            }

            // Computing overlay
            if self.computing && self.scalograms.is_empty() {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Computing…",
                    egui::FontId::proportional(18.0),
                    egui::Color32::WHITE,
                );
            }

            // --- Interaction (use ch==0 viewport for all channels) ---
            if ch == 0 {
                // Hover: crosshair + info
                if let Some(pos) = response.hover_pos() {
                    let rx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
                    let ry = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
                    let t_samp = new_viewport.t_start
                        + rx * (new_viewport.t_end - new_viewport.t_start);
                    let t_sec  = t_samp / sr;
                    let log_lo = f_min.ln();
                    let log_hi = f_max.ln();
                    let freq   = (log_hi - ry * (log_hi - log_lo)).exp();
                    self.hover_info = Some((t_sec, freq));

                    // Crosshair lines
                    let p = ui.painter();
                    let dim = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 60);
                    p.line_segment(
                        [egui::pos2(pos.x, rect.min.y), egui::pos2(pos.x, rect.max.y)],
                        egui::Stroke::new(1.0, dim),
                    );
                    p.line_segment(
                        [egui::pos2(rect.min.x, pos.y), egui::pos2(rect.max.x, pos.y)],
                        egui::Stroke::new(1.0, dim),
                    );
                }

                // Scroll: zoom time or frequency
                if response.hovered() {
                    let (scroll, ctrl) =
                        ctx.input(|i| (i.smooth_scroll_delta, i.modifiers.ctrl));

                    if scroll.y.abs() > 0.5 {
                        let zoom = if scroll.y > 0.0 { 0.85f64 } else { 1.0 / 0.85 };

                        if ctrl {
                            // Zoom frequency axis (log space)
                            let ry = response
                                .hover_pos()
                                .map(|p| {
                                    ((p.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64
                                })
                                .unwrap_or(0.5);
                            let log_lo  = f_min.ln();
                            let log_hi  = f_max.ln();
                            let log_cur = log_hi - ry * (log_hi - log_lo);
                            let new_lo  = log_cur - (log_cur - log_lo) * zoom;
                            let new_hi  = log_cur + (log_hi - log_cur) * zoom;
                            new_f_min = (new_lo.exp() as f32).clamp(0.5, 20_000.0);
                            new_f_max = (new_hi.exp() as f32).clamp(1.0, 22_050.0);
                            if new_f_min >= new_f_max { new_f_min = new_f_max * 0.5; }
                        } else {
                            // Zoom time axis
                            let rx = response
                                .hover_pos()
                                .map(|p| {
                                    ((p.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64
                                })
                                .unwrap_or(0.5);
                            let cur = new_viewport.t_start
                                + rx * (new_viewport.t_end - new_viewport.t_start);
                            new_viewport.t_start =
                                (cur - (cur - new_viewport.t_start) * zoom).max(0.0);
                            new_viewport.t_end =
                                (cur + (new_viewport.t_end - cur) * zoom)
                                    .min(total_samp as f64);
                            // Enforce minimum window (8 samples)
                            if new_viewport.t_end - new_viewport.t_start < 8.0 {
                                new_viewport.t_end = new_viewport.t_start + 8.0;
                            }
                        }
                        viewport_changed = true;
                    }
                }

                // Drag: pan time axis
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    let len   = new_viewport.t_end - new_viewport.t_start;
                    let dt    = -(delta.x as f64 / rect.width() as f64) * len;
                    new_viewport.t_start = (new_viewport.t_start + dt).max(0.0);
                    new_viewport.t_end   = new_viewport.t_start + len;
                    if new_viewport.t_end > total_samp as f64 {
                        new_viewport.t_end   = total_samp as f64;
                        new_viewport.t_start = new_viewport.t_end - len;
                    }
                }

                // Drag released → trigger recompute with panned view
                if response.drag_stopped() {
                    viewport_changed = true;
                }

                // Right-click: reset viewport
                if response.clicked_by(egui::PointerButton::Secondary) {
                    new_viewport = Viewport {
                        t_start: 0.0,
                        t_end:   total_samp as f64,
                    };
                    new_f_min = 20.0;
                    new_f_max = 20_000.0;
                    viewport_changed = true;
                }
            }

            // Time-axis labels
            let t0 = self.viewport.t_start / sr;
            let t1 = self.viewport.t_end   / sr;
            ui.horizontal(|ui| {
                ui.label(format!("{:.3} s", t0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{:.3} s", t1));
                });
            });
        }

        // Apply viewport / freq-range updates
        if viewport_changed {
            self.viewport      = new_viewport;
            self.params.f_min  = new_f_min;
            self.params.f_max  = new_f_max;
            if self.computing {
                self.pending_compute = true;
            } else {
                self.trigger_compute(render_width);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App implementation
// ---------------------------------------------------------------------------

impl eframe::App for WaveletApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll worker results
        while let Ok(result) = self.result_rx.try_recv() {
            match result {
                WorkerResult::Loaded(audio) => {
                    let n = audio.num_samples();
                    self.viewport = Viewport {
                        t_start: 0.0,
                        t_end:   n as f64,
                    };
                    self.params.f_min = 20.0;
                    self.params.f_max = (audio.sample_rate as f32 / 2.0).min(20_000.0);
                    self.textures = vec![None; audio.channels.len()];
                    self.scalograms.clear();
                    self.audio = Some(audio);
                    // Trigger initial compute
                    self.computing = false;
                    self.trigger_compute(self.last_width_px);
                }
                WorkerResult::Computed(scals) => {
                    self.computing       = false;
                    self.scalograms      = scals;
                    self.scalogram_width = self.last_width_px;
                    self.rebuild_textures(ctx);
                    self.status = "Ready.".into();
                    if self.pending_compute {
                        self.pending_compute = false;
                        let w = self.last_width_px;
                        self.trigger_compute(w);
                    }
                }
                WorkerResult::Error(e) => {
                    self.computing = false;
                    self.error_msg = Some(e.clone());
                    self.status    = format!("Error: {}", e);
                    log::error!("{}", e);
                }
            }
        }

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(250.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.show_controls(ui, ctx);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_scalograms(ui, ctx);
        });

        if self.computing {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

fn worker_thread(
    rx: mpsc::Receiver<WorkerMsg>,
    tx: mpsc::SyncSender<WorkerResult>,
    ptx_code: String,
) {
    let ctx = match CudaContext::init() {
        Ok(c)  => Arc::new(c),
        Err(e) => {
            let _ = tx.send(WorkerResult::Error(format!("CUDA init: {}", e)));
            return;
        }
    };
    let module = match ctx.load_ptx(&ptx_code) {
        Ok(m)  => m,
        Err(e) => {
            let _ = tx.send(WorkerResult::Error(format!("PTX load: {}", e)));
            return;
        }
    };
    let engine = match CwtEngine::new(Arc::clone(&ctx), &module) {
        Ok(e)  => e,
        Err(e) => {
            let _ = tx.send(WorkerResult::Error(format!("CwtEngine: {}", e)));
            return;
        }
    };

    for msg in rx.iter() {
        // For Compute messages, drain the queue and keep only the latest
        let msg = if matches!(msg, WorkerMsg::Compute(_)) {
            let mut latest = msg;
            while let Ok(newer) = rx.try_recv() {
                latest = newer;
            }
            latest
        } else {
            msg
        };

        match msg {
            WorkerMsg::LoadFile { path } => {
                match AudioFile::load(Path::new(&path)) {
                    Ok(a)  => { let _ = tx.send(WorkerResult::Loaded(a)); }
                    Err(e) => { let _ = tx.send(WorkerResult::Error(e.to_string())); }
                }
            }
            WorkerMsg::Compute(req) => {
                let mut results = Vec::with_capacity(req.channels.len());
                let mut err     = None;
                for ch in &req.channels {
                    match engine.compute(
                        ch,
                        req.sample_rate,
                        req.t_start,
                        req.t_end,
                        req.width_pixels,
                        &req.params,
                    ) {
                        Ok(s)  => results.push(s),
                        Err(e) => { err = Some(e.to_string()); break; }
                    }
                }
                if let Some(e) = err {
                    let _ = tx.send(WorkerResult::Error(e));
                } else {
                    let _ = tx.send(WorkerResult::Computed(results));
                }
            }
        }
    }
}
