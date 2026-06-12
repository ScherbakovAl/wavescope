use std::path::Path;
use std::sync::{mpsc, Arc};

use egui::TextureHandle;

use crate::audio::AudioFile;
use crate::colormap::{
    combined_to_rgba, instfreq_to_rgba, phase_to_rgba, scalogram_to_rgba,
    ColorMap, DisplayMode, InstFreqBaseline,
};
use crate::gpu::GpuContext;
use crate::cwt::{synchrosqueeze, synchrosqueeze_with_phase, CrossOutput, CwtEngine, CwtOutput};
use crate::wavelet::{CwtParams, WaveletKind};

// ---------------------------------------------------------------------------
// Worker thread messages
// ---------------------------------------------------------------------------

pub enum WorkerMsg {
    LoadFile { path: String },
    Compute(ComputeRequest),
}

pub struct ComputeRequest {
    pub channels:     Arc<Vec<Vec<f32>>>,
    pub sample_rate:  u32,
    pub t_start:      usize,
    pub t_end:        usize,
    pub width_pixels: usize,
    pub params:       CwtParams,
    pub antialias:    bool,
    pub unweighted:   bool,
    /// Superlet order (extra sharpness passes); 1 = plain CWT.
    pub superlet_order: u32,
}

pub enum WorkerResult {
    Loaded(AudioFile),
    Computed {
        /// (amplitude, phase, coherence, inst_dev) per channel.
        channels: Vec<CwtOutput>,
        /// Cross spectrum of the first two channels (stereo input only).
        cross: Option<CrossOutput>,
    },
    Error(String),
}

/// Ridge seed picked by double-click, in physical coordinates so it survives
/// pans / zooms / recomputes.
#[derive(Clone, Copy)]
struct RidgeSeed {
    t_sec: f64,
    f_hz:  f64,
    ch:    usize,
}

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Viewport {
    pub t_start: f64,   // visible window start in sample indices
    pub t_end:   f64,   // visible window end   in sample indices
}

/// Log-space frequency zoom around relative cursor height `ry` (0 = top).
/// `zoom < 1` shrinks the range (zoom in). Returns the new (f_min, f_max).
fn zoom_freq(ry: f64, zoom: f64, f_min: f64, f_max: f64) -> (f32, f32) {
    let log_lo  = f_min.ln();
    let log_hi  = f_max.ln();
    let log_cur = log_hi - ry * (log_hi - log_lo);
    let new_lo  = log_cur - (log_cur - log_lo) * zoom;
    let new_hi  = log_cur + (log_hi - log_cur) * zoom;
    let mut lo = (new_lo.exp() as f32).clamp(0.5, 20_000.0);
    let hi = (new_hi.exp() as f32).clamp(1.0, 22_050.0);
    if lo >= hi { lo = hi * 0.5; }
    (lo, hi)
}

/// Log-space frequency pan. `pan_pts` > 0 shifts the window toward higher
/// frequency. The span is preserved; the shift is clamped at the limits so the
/// window cannot grow or shrink. Returns the new (f_min, f_max).
fn pan_freq(pan_pts: f64, f_min: f64, f_max: f64) -> (f32, f32) {
    let log_lo = f_min.ln();
    let log_hi = f_max.ln();
    let lo_lim = 0.5f64.ln();
    let hi_lim = 22_050.0f64.ln();
    let d = (pan_pts * 0.002 * (log_hi - log_lo))
        .min(hi_lim - log_hi)
        .max(lo_lim - log_lo);
    ((log_lo + d).exp() as f32, (log_hi + d).exp() as f32)
}

// What the user is currently dragging on the time scrollbar.
#[derive(Clone, Copy, PartialEq)]
enum ScrollbarDrag {
    None,
    Pan,         // move the window (body grabbed)
    ResizeLeft,  // change zoom by moving the left edge
    ResizeRight, // change zoom by moving the right edge
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

pub struct WaveletApp {
    // Audio
    audio:             Option<AudioFile>,

    // Scalogram data  [scale * width + col], one per channel
    scalograms:        Vec<Vec<f32>>,
    phases:            Vec<Vec<f32>>,
    coherences:        Vec<Vec<f32>>,
    inst_devs:         Vec<Vec<f32>>,   // relative inst-freq deviation (f_inst−f_i)/f_i
    cross:             Option<CrossOutput>, // L/R cross spectrum (stereo only)
    scalogram_width:   usize,
    textures:          Vec<Option<TextureHandle>>,

    // Ridge tracking (double-click to pick, follows recomputes)
    ridge_seed:        Option<RidgeSeed>,
    ridge_rows:        Option<Vec<usize>>,  // scale row per column

    // Viewport & parameters
    viewport:     Viewport,         // currently shown (effective) time window
    tex_viewport: Viewport,         // window the current texture was computed for
    pending_compute_viewport: Viewport, // window of the in-flight compute
    panning:      bool,             // smooth pan in progress (drag + its recompute)
    scrollbar_drag: ScrollbarDrag,  // active time-scrollbar interaction
    params:       CwtParams,
    colormap:     ColorMap,
    display_mode: DisplayMode,
    phase_gamma:   f32,    // coherence→saturation shaping for phase views
    instfreq_baseline:    InstFreqBaseline, // what the inst-freq view subtracts
    instfreq_range:       f32,    // ± full-scale (relative) of the diverging map
    instfreq_detrend_win: usize,  // detrend moving-average window (pixels)
    log_amount:    f32,    // 0.0 = linear, 1.0 = logarithmic brightness
    vmin:          f32,    // brightness window low end (raw amplitude)
    vmax:          f32,    // brightness window high end (raw amplitude)
    data_min:      f32,    // captured data extent → slider bounds
    data_max:      f32,
    norm_captured: bool,   // frozen after first compute; no auto-rescale
    antialias:     bool,
    unweighted:    bool,   // unweighted Δφ estimator (see ExtractParams)
    superlet_order: u32,   // passes in the Superlet view (cost ∝ order)

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
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (work_tx, work_rx) = mpsc::sync_channel::<WorkerMsg>(8);
        let (res_tx,  res_rx)  = mpsc::sync_channel::<WorkerResult>(8);

        std::thread::spawn(move || worker_thread(work_rx, res_tx));

        WaveletApp {
            audio:           None,
            scalograms:      Vec::new(),
            phases:          Vec::new(),
            coherences:      Vec::new(),
            inst_devs:       Vec::new(),
            cross:           None,
            scalogram_width: 0,
            textures:        Vec::new(),
            ridge_seed:      None,
            ridge_rows:      None,
            viewport:        Viewport { t_start: 0.0, t_end: 1.0 },
            tex_viewport:    Viewport { t_start: 0.0, t_end: 1.0 },
            pending_compute_viewport: Viewport { t_start: 0.0, t_end: 1.0 },
            panning:         false,
            scrollbar_drag:  ScrollbarDrag::None,
            params:          CwtParams::default(),
            colormap:        ColorMap::Plasma,
            display_mode:    DisplayMode::Amplitude,
            phase_gamma:     2.0,
            instfreq_baseline:    InstFreqBaseline::Nominal,
            instfreq_range:       0.02,   // ±2 % relative deviation full-scale
            instfreq_detrend_win: 32,
            log_amount:      1.0,
            vmin:            0.0,
            vmax:            1.0,
            data_min:        0.0,
            data_max:        1.0,
            norm_captured:   false,
            antialias:       true,
            unweighted:      false,
            superlet_order:  3,
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
        let sr = match &self.audio { Some(a) => a.sample_rate, None => return };
        // Keep f_max safely below Nyquist so the computed rows match the
        // labelled f_min..f_max axis exactly (no silent f_max_eff clamp).
        let nyq_cap = sr as f32 * 0.49;
        self.params.f_max = self.params.f_max.min(nyq_cap).max(2.0);
        self.params.f_min = self.params.f_min.min(self.params.f_max * 0.99);
        let audio = self.audio.as_ref().unwrap();
        let t0 = self.viewport.t_start as usize;
        let t1 = (self.viewport.t_end as usize).min(audio.num_samples());
        if t0 >= t1 { return; }
        let req = ComputeRequest {
            channels:     audio.channels.clone(),   // Arc: no data copy
            sample_rate:  audio.sample_rate,
            t_start:      t0,
            t_end:        t1,
            width_pixels: width_px,
            params:       self.params.clone(),
            antialias:    self.antialias,
            unweighted:   self.unweighted,
            // Extra superlet passes are paid only while the view needs them.
            superlet_order: if self.display_mode == DisplayMode::Superlet {
                self.superlet_order
            } else {
                1
            },
        };
        let _ = self.work_tx.try_send(WorkerMsg::Compute(req));
        self.pending_compute_viewport = self.viewport.clone();
        self.computing       = true;
        self.pending_compute = false;
        self.status          = "Computing…".into();
    }

    // -----------------------------------------------------------------------
    // Rebuild egui textures from scalogram data
    // -----------------------------------------------------------------------
    /// True when the cross-phase view is selected and cross data exists.
    fn cross_active(&self) -> bool {
        self.display_mode == DisplayMode::CrossPhase && self.cross.is_some()
    }

    fn rebuild_textures(&mut self, ctx: &egui::Context) {
        let width = self.scalogram_width;
        self.textures.clear();
        if width == 0 { return; }
        let (vmin, vmax) = (self.vmin, self.vmax);
        // Phase hue must not be bilinearly blended (wraps +π/−π through the
        // whole wheel ⇒ false bands), so use NEAREST for phase-based views.
        let opts = match self.display_mode {
            DisplayMode::Amplitude | DisplayMode::InstFreq
            | DisplayMode::Synchro | DisplayMode::Superlet => egui::TextureOptions::LINEAR,
            _                                              => egui::TextureOptions::NEAREST,
        };

        if self.cross_active() {
            let cr = self.cross.as_ref().unwrap();
            let num_scales = cr.phase.len() / width;
            if num_scales == 0 { return; }
            let rgba = combined_to_rgba(
                &cr.amplitude, &cr.phase, &cr.coherence, width, num_scales,
                vmin, vmax, self.log_amount, self.phase_gamma,
            );
            let img = egui::ColorImage::from_rgba_unmultiplied([width, num_scales], &rgba);
            self.textures.push(Some(ctx.load_texture("scalogram", img, opts)));
            return;
        }

        for (i, sc) in self.scalograms.iter().enumerate() {
            let num_scales = sc.len() / width;
            if num_scales == 0 { continue; }
            let rgba = match (self.display_mode, self.phases.get(i), self.coherences.get(i)) {
                (DisplayMode::Phase, Some(ph), Some(co)) =>
                    phase_to_rgba(ph, co, width, num_scales, self.phase_gamma),
                (DisplayMode::Combined, Some(ph), Some(co)) =>
                    combined_to_rgba(sc, ph, co, width, num_scales, vmin, vmax, self.log_amount, self.phase_gamma),
                (DisplayMode::InstFreq, _, _) if self.inst_devs.get(i).is_some() =>
                    instfreq_to_rgba(
                        &self.inst_devs[i], sc, width, num_scales,
                        self.instfreq_baseline, self.instfreq_range,
                        self.instfreq_detrend_win, vmin, vmax, self.log_amount,
                    ),
                (DisplayMode::Synchro, _, _) if self.inst_devs.get(i).is_some() => {
                    // Rows are log-spaced over [f_min, f_max] (see cwt.rs),
                    // so the reassignment needs only ln(f_max/f_min).
                    let log_ratio = (self.params.f_max / self.params.f_min).ln();
                    let sq = synchrosqueeze(
                        sc, &self.inst_devs[i], width, num_scales, log_ratio,
                    );
                    scalogram_to_rgba(
                        &sq, width, num_scales, self.colormap, vmin, vmax, self.log_amount,
                    )
                }
                (DisplayMode::SynchroPhase, Some(ph), _) if self.inst_devs.get(i).is_some() => {
                    let log_ratio = (self.params.f_max / self.params.f_min).ln();
                    let (sq, sph, sco) = synchrosqueeze_with_phase(
                        sc, ph, &self.inst_devs[i], width, num_scales, log_ratio,
                    );
                    combined_to_rgba(
                        &sq, &sph, &sco, width, num_scales,
                        vmin, vmax, self.log_amount, self.phase_gamma,
                    )
                }
                _ =>
                    scalogram_to_rgba(sc, width, num_scales, self.colormap, vmin, vmax, self.log_amount),
            };
            let img  = egui::ColorImage::from_rgba_unmultiplied([width, num_scales], &rgba);
            let tex  = ctx.load_texture("scalogram", img, opts);
            self.textures.push(Some(tex));
        }
    }

    // -----------------------------------------------------------------------
    // Global amplitude (vmin, vmax) over all current channels — used as the
    // frozen normalisation reference so brightness does not auto-rescale.
    // -----------------------------------------------------------------------
    fn compute_norm(&self) -> (f32, f32) {
        let mut vmin = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for sc in &self.scalograms {
            for &v in sc {
                if v < vmin { vmin = v; }
                if v > vmax { vmax = v; }
            }
        }
        if !vmin.is_finite() { vmin = 0.0; }
        if !vmax.is_finite() { vmax = 1.0; }
        (vmin, vmax)
    }

    /// Capture the data extent and reset the brightness window to span it.
    /// Also positions the vmin/vmax sliders (their bounds and values).
    fn capture_norm(&mut self) {
        let (lo, hi) = self.compute_norm();
        self.data_min = lo;
        self.data_max = hi;
        self.vmin     = lo;
        self.vmax     = hi;
    }

    // -----------------------------------------------------------------------
    // Ridge tracking & export
    // -----------------------------------------------------------------------

    /// (Re)extract the ridge from the current scalogram data: greedy
    /// local-maximum tracking from the seed, ±2 rows per column step.
    /// Cleared when the seed lies outside the computed window.
    fn update_ridge(&mut self) {
        self.ridge_rows = None;
        let Some(seed) = self.ridge_seed else { return };
        let Some(audio) = &self.audio else { return };
        let width = self.scalogram_width;
        if width == 0 { return; }
        let Some(amp) = self.scalograms.get(seed.ch) else { return };
        let ns = amp.len() / width;
        if ns < 2 { return; }
        let sr   = audio.sample_rate as f64;
        let vp   = &self.tex_viewport;
        let span = vp.t_end - vp.t_start;
        if span <= 0.0 { return; }
        let colf = (seed.t_sec * sr - vp.t_start) / span * width as f64;
        if colf < 0.0 || colf >= width as f64 { return; }
        let col0 = colf as usize;
        let lf0 = (self.params.f_min as f64).ln();
        let lf1 = (self.params.f_max as f64).ln();
        if lf1 <= lf0 { return; }
        let row_guess = ((seed.f_hz.ln() - lf0) / (lf1 - lf0) * (ns - 1) as f64)
            .round()
            .clamp(0.0, (ns - 1) as f64) as usize;

        let argmax = |col: usize, center: usize, radius: usize| -> usize {
            let lo = center.saturating_sub(radius);
            let hi = (center + radius).min(ns - 1);
            (lo..=hi)
                .max_by(|&a, &b| {
                    amp[a * width + col]
                        .partial_cmp(&amp[b * width + col])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(center)
        };

        let mut rows = vec![0usize; width];
        let r0 = argmax(col0, row_guess, 3);
        let mut r = r0;
        for (col, slot) in rows.iter_mut().enumerate().skip(col0) {
            r = argmax(col, r, 2);
            *slot = r;
        }
        r = r0;
        for (col, slot) in rows.iter_mut().enumerate().take(col0).rev() {
            r = argmax(col, r, 2);
            *slot = r;
        }
        self.ridge_rows = Some(rows);
    }

    /// Save the tracked ridge as CSV (one row per column of the current view).
    fn export_ridge_csv(&mut self) {
        let Some(seed) = self.ridge_seed else { return };
        let Some(rows) = &self.ridge_rows else { return };
        let Some(audio) = &self.audio else { return };
        let width = self.scalogram_width;
        if width == 0 || rows.len() != width { return; }
        let (Some(amp), Some(ph), Some(co), Some(dev)) = (
            self.scalograms.get(seed.ch),
            self.phases.get(seed.ch),
            self.coherences.get(seed.ch),
            self.inst_devs.get(seed.ch),
        ) else { return };
        let ns = amp.len() / width;
        if ns < 2 { return; }

        let sr   = audio.sample_rate as f64;
        let vp   = &self.tex_viewport;
        let span = vp.t_end - vp.t_start;
        let lf0  = (self.params.f_min as f64).ln();
        let lf1  = (self.params.f_max as f64).ln();

        let mut csv =
            String::from("time_s,f_row_hz,f_inst_hz,rel_dev,amplitude,phase_rad,coherence\n");
        for (col, &row) in rows.iter().enumerate() {
            let t = (vp.t_start + (col as f64 + 0.5) / width as f64 * span) / sr;
            let f_row = (lf0 + row as f64 / (ns - 1) as f64 * (lf1 - lf0)).exp();
            let i = row * width + col;
            let d = dev[i] as f64;
            let f_inst = f_row * (1.0 + d);
            csv.push_str(&format!(
                "{t:.6},{f_row:.3},{f_inst:.3},{d:.6},{:.6},{:.4},{:.4}\n",
                amp[i], ph[i], co[i],
            ));
        }

        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ridge.csv")
            .save_file()
        {
            match std::fs::write(&path, csv) {
                Ok(())  => self.status = format!("Ridge exported: {}", path.display()),
                Err(e)  => self.error_msg = Some(format!("CSV write failed: {e}")),
            }
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
                .add_filter("Audio", &["wav", "WAV", "wave", "WAVE", "flac", "FLAC"])
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

        // Wavelet family selector.
        egui::ComboBox::from_label("Wavelet")
            .selected_text(self.params.kind.name())
            .show_ui(ui, |ui| {
                for k in [
                    WaveletKind::Morlet,
                    WaveletKind::Morse,
                    WaveletKind::Bump,
                    WaveletKind::Paul,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.params.kind, k, k.name())
                        .changed();
                }
            });

        // Per-family shape controls.
        match self.params.kind {
            WaveletKind::Morlet => {
                ui.label("ω₀ (low→high freq):");
                changed |= ui.add(
                    egui::Slider::new(&mut self.params.omega0_low, 4.0..=200.0).text("ω₀ @f_min"),
                ).changed();
                changed |= ui.add(
                    egui::Slider::new(&mut self.params.omega0_high, 4.0..=200.0).text("ω₀ @f_max"),
                ).changed();
            }
            WaveletKind::Morse => {
                ui.label("Generalized Morse (β sharpness, γ symmetry):");
                changed |= ui.add(
                    egui::Slider::new(&mut self.params.morse_beta, 1.0..=60.0).text("β"),
                ).changed();
                changed |= ui.add(
                    egui::Slider::new(&mut self.params.morse_gamma, 1.0..=8.0).text("γ"),
                ).changed();
            }
            WaveletKind::Bump => {
                ui.label("Bump (σ = bandwidth; smaller ⇒ sharper freq):");
                changed |= ui.add(
                    egui::Slider::new(&mut self.params.bump_sigma, 0.1..=2.0).text("σ"),
                ).changed();
            }
            WaveletKind::Paul => {
                ui.label("Paul (order m; larger ⇒ sharper freq):");
                let mut m = self.params.paul_order.round() as u32;
                if ui.add(egui::Slider::new(&mut m, 1..=40).text("m")).changed() {
                    self.params.paul_order = m as f32;
                    changed = true;
                }
            }
        }

        ui.label("Scales:");
        let mut ns = self.params.num_scales as u32;
        if ui.add(egui::Slider::new(&mut ns, 64..=2048).logarithmic(true).text("n")).changed() {
            self.params.num_scales = ns as usize;
            changed = true;
        }

        ui.label("f min (Hz):");
        if ui.add(
            egui::Slider::new(&mut self.params.f_min, 1.0..=22050.0)
                .logarithmic(true)
                .text("Hz"),
        ).changed() {
            self.params.f_min = self.params.f_min.min(self.params.f_max * 0.99);
            changed = true;
        }

        ui.label("f max (Hz):");
        if ui.add(
            egui::Slider::new(&mut self.params.f_max, 1.0..=22050.0)
                .logarithmic(true)
                .text("Hz"),
        ).changed() {
            self.params.f_max = self.params.f_max.max(self.params.f_min * 1.01);
            changed = true;
        }

        ui.separator();
        ui.heading("Display");

        let stereo = self.audio.as_ref().is_some_and(|a| a.channels.len() >= 2);
        let mut modes = vec![
            DisplayMode::Amplitude,
            DisplayMode::Synchro,
            DisplayMode::SynchroPhase,
            DisplayMode::Superlet,
            DisplayMode::Phase,
            DisplayMode::Combined,
            DisplayMode::InstFreq,
        ];
        if stereo { modes.push(DisplayMode::CrossPhase); }
        let prev_mode = self.display_mode;
        let mut mode_changed = false;
        egui::ComboBox::from_label("Mode")
            .selected_text(self.display_mode.name())
            .show_ui(ui, |ui| {
                for dm in modes {
                    mode_changed |= ui
                        .selectable_value(&mut self.display_mode, dm, dm.name())
                        .changed();
                }
            });
        if mode_changed {
            // Superlet amplitudes come from extra compute passes, so entering
            // or leaving that view recomputes; any other switch re-renders.
            let superlet_edge = (prev_mode == DisplayMode::Superlet)
                != (self.display_mode == DisplayMode::Superlet);
            if superlet_edge && self.audio.is_some() {
                if self.computing {
                    self.pending_compute = true;
                } else {
                    self.trigger_compute(self.last_width_px);
                }
            } else if self.scalogram_width > 0 {
                self.rebuild_textures(ctx);
            }
        }

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

        if self.display_mode == DisplayMode::Superlet {
            ui.label("Superlet order (CWT passes; sharpness ×1…×order):");
            let mut o = self.superlet_order;
            if ui.add(egui::Slider::new(&mut o, 2..=10).text("order")).changed() {
                self.superlet_order = o;
                if self.audio.is_some() {
                    if self.computing {
                        self.pending_compute = true;
                    } else {
                        self.trigger_compute(self.last_width_px);
                    }
                }
            }
        }

        if matches!(
            self.display_mode,
            DisplayMode::Phase | DisplayMode::Combined | DisplayMode::CrossPhase
            | DisplayMode::SynchroPhase
        ) {
            ui.label("Phase coherence γ (fade unresolved phase):");
            if ui.add(
                egui::Slider::new(&mut self.phase_gamma, 0.0..=6.0).text("γ"),
            ).changed() {
                let w = self.scalogram_width;
                if w > 0 { self.rebuild_textures(ctx); }
            }
        }

        if self.display_mode == DisplayMode::InstFreq {
            egui::ComboBox::from_label("Baseline")
                .selected_text(self.instfreq_baseline.name())
                .show_ui(ui, |ui| {
                    for b in [InstFreqBaseline::Nominal, InstFreqBaseline::Detrend] {
                        if ui.selectable_value(&mut self.instfreq_baseline, b, b.name()).changed() {
                            let w = self.scalogram_width;
                            if w > 0 { self.rebuild_textures(ctx); }
                        }
                    }
                });

            ui.label("Deviation full-scale (±, relative):");
            if ui.add(
                egui::Slider::new(&mut self.instfreq_range, 0.002..=0.5)
                    .logarithmic(true)
                    .text("±rel"),
            ).changed() {
                let w = self.scalogram_width;
                if w > 0 { self.rebuild_textures(ctx); }
            }

            if self.instfreq_baseline == InstFreqBaseline::Detrend {
                ui.label("Detrend window (px):");
                let mut win = self.instfreq_detrend_win as u32;
                if ui.add(
                    egui::Slider::new(&mut win, 3..=256).logarithmic(true).text("px"),
                ).changed() {
                    self.instfreq_detrend_win = win as usize;
                    let w = self.scalogram_width;
                    if w > 0 { self.rebuild_textures(ctx); }
                }
            }

            // Phase slips coincide with amplitude dips; the default estimator
            // weights increments by |W|² and underweights exactly those
            // moments. Unweighted = pure circular mean of Δφ.
            if ui.checkbox(&mut self.unweighted, "Unweighted Δφ (slip-sensitive)")
                .changed() && self.audio.is_some()
            {
                if self.computing {
                    self.pending_compute = true;
                } else {
                    self.trigger_compute(self.last_width_px);
                }
            }
        }

        ui.label("Amplitude brightness (linear ↔ log):");
        if ui.add(
            egui::Slider::new(&mut self.log_amount, 0.0..=1.0).text("log amount"),
        ).changed() {
            let w = self.scalogram_width;
            if w > 0 { self.rebuild_textures(ctx); }
        }

        ui.label("Brightness range:");
        let lo = self.data_min;
        let hi = self.data_max.max(self.data_min + 1e-12);
        let mut range_changed = false;
        // Shift = ultra-fine vmin tuning when dragging over the value box.
        let vmin_speed = if ui.input(|i| i.modifiers.shift) { 0.000001 } else { 0.00001 };
        let vmin_slider = egui::Slider::new(&mut self.vmin, lo..=hi).text("vmin")
            .drag_value_speed(vmin_speed);
        range_changed |= ui.add(vmin_slider).changed();
        range_changed |= ui.add(
            egui::Slider::new(&mut self.vmax, lo..=hi).text("vmax"),
        ).changed();
        if range_changed {
            if self.vmin > self.vmax { self.vmin = self.vmax; }
            if self.scalogram_width > 0 { self.rebuild_textures(ctx); }
        }

        if ui.button("Auto-normalize brightness").clicked() && self.scalogram_width > 0 {
            self.capture_norm();
            self.rebuild_textures(ctx);
        }

        if ui.checkbox(&mut self.antialias, "Anti-aliasing filter").changed()
            && self.audio.is_some()
        {
            if self.computing {
                self.pending_compute = true;
            } else {
                self.trigger_compute(self.last_width_px);
            }
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
        ui.heading("Ridge");
        if let Some(seed) = self.ridge_seed {
            ui.label(format!("Seed: {:.3} s, {:.1} Hz (ch {})",
                seed.t_sec, seed.f_hz, seed.ch + 1));
            if self.ridge_rows.is_some() {
                if ui.button("💾 Export ridge CSV…").clicked() {
                    self.export_ridge_csv();
                }
            } else {
                ui.label("(outside current view)");
            }
            if ui.button("✖ Clear ridge").clicked() {
                self.ridge_seed = None;
                self.ridge_rows = None;
            }
        } else {
            ui.label("Double-click a ridge on the scalogram to track it; the trace \
                      (t, f_inst, amplitude, phase) can then be exported as CSV.");
        }

        ui.separator();
        ui.label("Interactions:");
        ui.label("Scroll              – zoom time axis");
        ui.label("Ctrl+Scroll         – zoom freq axis");
        ui.label("Shift+Scroll        – pan freq");
        ui.label("Alt+Scroll          – pan time");
        ui.label("Drag                – pan time");
        ui.label("Double-click        – pick ridge");
        ui.label("Right-click         – reset view");

        // Status / spinner at bottom
        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
            if let Some(e) = &self.error_msg {
                ui.colored_label(egui::Color32::RED, e);
            } else {
                ui.label(self.status.clone());
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
                     Shift+Scroll to pan freq • Alt+Scroll to pan time • Drag to pan time\n\
                     Double-click to pick a ridge • Right-click to reset view",
                );
            });
            return;
        }

        let audio        = self.audio.as_ref().unwrap();
        let num_ch       = audio.channels.len();
        let total_samp   = audio.num_samples();
        let sr           = audio.sample_rate as f64;
        let f_min        = self.params.f_min as f64;
        let f_max        = self.params.f_max as f64;
        let cross_active = self.cross_active();
        let panels       = if cross_active { 1 } else { num_ch };

        // Reserve room at the bottom for the time scrollbar (height + spacing)
        // so it stays inside the window below the scalograms.
        let scrollbar_room = 44.0;
        let ch_height =
            (((available.y - scrollbar_room) / panels.max(1) as f32) - 30.0).max(80.0);

        self.hover_info = None;
        let mut viewport_changed = false;
        let mut new_viewport     = self.viewport.clone();

        // While panning, slide the existing texture under the cursor instead of
        // waiting for a recompute. Offset = how far the texture's window sits
        // from the window we are now showing (in time), drawn as a pixel shift.
        let pan_dt  = if self.panning {
            self.tex_viewport.t_start - self.viewport.t_start
        } else { 0.0 };
        let pan_len = (self.viewport.t_end - self.viewport.t_start).max(1e-9);
        let mut new_f_min        = self.params.f_min;
        let mut new_f_max        = self.params.f_max;

        for ch in 0..panels {
            let label = if cross_active {
                "Δφ L−R".to_string()
            } else {
                match num_ch {
                    1 => "Mono".to_string(),
                    2 => if ch == 0 { "L".to_string() } else { "R".to_string() },
                    _ => format!("Ch {}", ch + 1),
                }
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

            // Draw scalogram texture (if available), shifted by the pan offset.
            // Revealed strips stay on the black background until the recompute
            // for the new window arrives.
            if ch < self.textures.len() {
                if let Some(tex) = &self.textures[ch] {
                    let w   = rect.width();
                    let off = (pan_dt / pan_len * w as f64) as f32;
                    if off.abs() < w {
                        let (x0, x1, u0, u1) = if off >= 0.0 {
                            (rect.min.x + off, rect.max.x, 0.0, (w - off) / w)
                        } else {
                            (rect.min.x, rect.max.x + off, -off / w, 1.0)
                        };
                        ui.painter().image(
                            tex.id(),
                            egui::Rect::from_min_max(
                                egui::pos2(x0, rect.min.y),
                                egui::pos2(x1, rect.max.y),
                            ),
                            egui::Rect::from_min_max(
                                egui::pos2(u0, 0.0),
                                egui::pos2(u1, 1.0),
                            ),
                            egui::Color32::WHITE,
                        );
                    }
                }
            }

            // Ridge overlay: drawn on the seed's channel panel (panel 0 in
            // cross mode), shifted by the same pan offset as the texture.
            if let (Some(seed), Some(rows)) = (self.ridge_seed, &self.ridge_rows) {
                let on_this_panel = if cross_active { ch == 0 } else { ch == seed.ch };
                if on_this_panel && self.scalogram_width > 0 {
                    if let Some(sc) = self.scalograms.get(seed.ch) {
                        let width = self.scalogram_width;
                        let ns = sc.len() / width;
                        if ns > 1 && rows.len() == width {
                            let w   = rect.width();
                            let off = (pan_dt / pan_len * w as f64) as f32;
                            let pts: Vec<egui::Pos2> = rows
                                .iter()
                                .enumerate()
                                .map(|(col, &row)| {
                                    let x = rect.min.x
                                        + (col as f32 + 0.5) / width as f32 * w + off;
                                    let y = rect.min.y
                                        + (1.0 - row as f32 / (ns - 1) as f32) * rect.height();
                                    egui::pos2(x, y)
                                })
                                .filter(|p| p.x >= rect.min.x && p.x <= rect.max.x)
                                .collect();
                            if pts.len() > 1 {
                                ui.painter().add(egui::Shape::line(
                                    pts,
                                    egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 255, 255)),
                                ));
                            }
                        }
                    }
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

            // --- Interaction: handled per-channel (each has its own rect /
            // response), so pan/zoom works on whichever graph is under the
            // cursor. They all share the same time viewport / freq range. ---
            {
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

                // Double-click: pick a ridge seed at the cursor (time + freq).
                if response.double_clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let rx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
                        let ry = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
                        let t_samp = new_viewport.t_start
                            + rx * (new_viewport.t_end - new_viewport.t_start);
                        let log_lo = f_min.ln();
                        let log_hi = f_max.ln();
                        let freq   = (log_hi - ry * (log_hi - log_lo)).exp();
                        self.ridge_seed = Some(RidgeSeed {
                            t_sec: t_samp / sr,
                            f_hz:  freq,
                            ch:    if cross_active { 0 } else { ch },
                        });
                        self.update_ridge();
                    }
                }

                // Scroll wheel:
                //   plain        – zoom time axis around cursor
                //   Ctrl (pinch) – zoom freq axis (egui-winit reports it as
                //                  zoom_delta and zeros raw_scroll_delta)
                //   Shift        – pan freq axis
                //   Alt / horizontal wheel – pan time
                if response.hovered() {
                    let (scroll, modifiers, zoomf) =
                        ctx.input(|i| (i.raw_scroll_delta, i.modifiers, i.zoom_delta()));

                    let ry = response
                        .hover_pos()
                        .map(|p| ((p.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64)
                        .unwrap_or(0.5);

                    if (zoomf - 1.0).abs() > 1e-3 {
                        // Ctrl+scroll / pinch: zoom frequency around cursor.
                        let (lo, hi) = zoom_freq(ry, 1.0 / zoomf as f64, f_min, f_max);
                        new_f_min = lo;
                        new_f_max = hi;
                        viewport_changed = true;
                    }

                    let pan_pts = if modifiers.alt { scroll.y } else { 0.0 } + scroll.x;
                    if modifiers.shift {
                        // Shift+wheel: pan frequency (like Alt+wheel pans time).
                        // Many platforms deliver Shift+wheel as a horizontal
                        // scroll, so take whichever axis carries the delta.
                        let wheel = if scroll.y.abs() >= scroll.x.abs() {
                            scroll.y
                        } else {
                            scroll.x
                        };
                        if wheel.abs() > 0.5 {
                            let (lo, hi) = pan_freq(wheel as f64, f_min, f_max);
                            new_f_min = lo;
                            new_f_max = hi;
                            viewport_changed = true;
                        }
                    } else if pan_pts.abs() > 0.5 {
                        // Pan: one wheel notch (~50 pt) ≈ 10 % of the window.
                        let len = new_viewport.t_end - new_viewport.t_start;
                        let dt  = -(pan_pts as f64) * 0.002 * len;
                        let mut ts = (new_viewport.t_start + dt).max(0.0);
                        let mut te = ts + len;
                        if te > total_samp as f64 {
                            te = total_samp as f64;
                            ts = (te - len).max(0.0);
                        }
                        new_viewport.t_start = ts;
                        new_viewport.t_end   = te;
                        viewport_changed = true;
                    } else if scroll.y.abs() > 0.5 && !modifiers.ctrl && !modifiers.command {
                        // Zoom time axis around cursor. Skip when Ctrl is held:
                        // Ctrl+wheel already zooms frequency via zoom_delta
                        // above, and some platforms also report a raw scroll
                        // delta that would otherwise zoom time.
                        let zoom = if scroll.y > 0.0 { 0.85f64 } else { 1.0 / 0.85 };
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
                        if new_viewport.t_end - new_viewport.t_start < 8.0 {
                            new_viewport.t_end = new_viewport.t_start + 8.0;
                        }
                        viewport_changed = true;
                    }
                }

                // Drag: pan time axis.
                // FIX: apply delta to self.viewport each frame so that the
                // accumulated pan is not lost between frames.
                if response.dragged_by(egui::PointerButton::Primary) {
                    self.panning = true;
                    let delta = response.drag_delta();
                    // Use current self.viewport (already committed) as base
                    let len = self.viewport.t_end - self.viewport.t_start;
                    let dt  = -(delta.x as f64 / rect.width() as f64) * len;
                    let mut ts = (self.viewport.t_start + dt).max(0.0);
                    let mut te = ts + len;
                    if te > total_samp as f64 {
                        te = total_samp as f64;
                        ts = (te - len).max(0.0);
                    }
                    self.viewport.t_start = ts;
                    self.viewport.t_end   = te;
                    new_viewport = self.viewport.clone();
                    // Repaint so labels follow cursor; recompute on release
                }

                // Drag released → trigger recompute with final panned view
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

        // Time scrollbar (position + zoom of the visible window over the file)
        if self.show_time_scrollbar(ui, total_samp as f64) {
            new_viewport     = self.viewport.clone();
            viewport_changed = true;
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

    // -----------------------------------------------------------------------
    // Horizontal time scrollbar.
    // The track spans the whole file; the thumb's position and width show the
    // visible window's position and zoom. Drag the body to pan, drag an edge
    // to zoom. Returns true on release → caller triggers a recompute.
    // -----------------------------------------------------------------------
    fn show_time_scrollbar(&mut self, ui: &mut egui::Ui, total_samp: f64) -> bool {
        let total = total_samp.max(1.0);
        ui.add_space(4.0);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 16.0),
            egui::Sense::click_and_drag(),
        );

        // Thumb geometry from the current viewport
        let frac0 = (self.viewport.t_start / total).clamp(0.0, 1.0) as f32;
        let frac1 = (self.viewport.t_end   / total).clamp(0.0, 1.0) as f32;
        let x0 = rect.min.x + frac0 * rect.width();
        let x1 = (rect.min.x + frac1 * rect.width()).max(x0 + 6.0);
        let thumb = egui::Rect::from_min_max(
            egui::pos2(x0, rect.min.y),
            egui::pos2(x1, rect.max.y),
        );
        let edge = 6.0_f32;

        // Draw track + thumb
        let active = self.scrollbar_drag != ScrollbarDrag::None;
        let thumb_color = if active || response.hovered() {
            egui::Color32::from_gray(160)
        } else {
            egui::Color32::from_gray(110)
        };
        let p = ui.painter();
        p.rect_filled(rect, 4.0, egui::Color32::from_gray(40));
        p.rect_filled(thumb, 4.0, thumb_color);
        for gx in [thumb.min.x + 3.0, thumb.max.x - 3.0] {
            p.line_segment(
                [egui::pos2(gx, thumb.min.y + 3.0), egui::pos2(gx, thumb.max.y - 3.0)],
                egui::Stroke::new(1.5, egui::Color32::from_gray(210)),
            );
        }

        // Decide drag mode on press (and jump-center when clicking the track)
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                if (pos.x - thumb.min.x).abs() <= edge {
                    self.scrollbar_drag = ScrollbarDrag::ResizeLeft;
                } else if (pos.x - thumb.max.x).abs() <= edge {
                    self.scrollbar_drag = ScrollbarDrag::ResizeRight;
                } else if pos.x >= thumb.min.x && pos.x <= thumb.max.x {
                    self.scrollbar_drag = ScrollbarDrag::Pan;
                } else {
                    // Click on the track outside the thumb → center the window here
                    self.scrollbar_drag = ScrollbarDrag::Pan;
                    let len    = self.viewport.t_end - self.viewport.t_start;
                    let center = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64 * total;
                    let mut ts = (center - len / 2.0).max(0.0);
                    let mut te = ts + len;
                    if te > total { te = total; ts = (te - len).max(0.0); }
                    self.viewport.t_start = ts;
                    self.viewport.t_end   = te;
                    self.panning = true;
                }
            }
        }

        // Live drag
        if response.dragged() && self.scrollbar_drag != ScrollbarDrag::None {
            let dt      = response.drag_delta().x as f64 / rect.width() as f64 * total;
            let min_len = 8.0;
            match self.scrollbar_drag {
                ScrollbarDrag::Pan => {
                    let len    = self.viewport.t_end - self.viewport.t_start;
                    let mut ts = (self.viewport.t_start + dt).max(0.0);
                    let mut te = ts + len;
                    if te > total { te = total; ts = (te - len).max(0.0); }
                    self.viewport.t_start = ts;
                    self.viewport.t_end   = te;
                    self.panning = true;
                }
                ScrollbarDrag::ResizeLeft => {
                    let mut ts = (self.viewport.t_start + dt).max(0.0);
                    if ts > self.viewport.t_end - min_len { ts = self.viewport.t_end - min_len; }
                    self.viewport.t_start = ts;
                }
                ScrollbarDrag::ResizeRight => {
                    let mut te = (self.viewport.t_end + dt).min(total);
                    if te < self.viewport.t_start + min_len { te = self.viewport.t_start + min_len; }
                    self.viewport.t_end = te;
                }
                ScrollbarDrag::None => {}
            }
        }

        // Cursor feedback
        match self.scrollbar_drag {
            ScrollbarDrag::Pan =>
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing),
            ScrollbarDrag::ResizeLeft | ScrollbarDrag::ResizeRight =>
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal),
            ScrollbarDrag::None => {
                if let Some(pos) = response.hover_pos() {
                    if (pos.x - thumb.min.x).abs() <= edge || (pos.x - thumb.max.x).abs() <= edge {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    } else if pos.x >= thumb.min.x && pos.x <= thumb.max.x {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                }
            }
        }

        // Release → recompute for the final window
        if response.drag_stopped() {
            self.scrollbar_drag = ScrollbarDrag::None;
            self.panning        = false;
            return true;
        }
        false
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
                    self.tex_viewport = self.viewport.clone();
                    self.panning      = false;
                    self.params.f_min = 20.0;
                    self.params.f_max = (audio.sample_rate as f32 * 0.49).min(20_000.0);
                    self.textures = vec![None; audio.channels.len()];
                    self.scalograms.clear();
                    self.cross         = None;
                    self.ridge_seed    = None;
                    self.ridge_rows    = None;
                    self.norm_captured = false;   // recapture brightness reference for new file
                    if audio.channels.len() < 2 && self.display_mode == DisplayMode::CrossPhase {
                        self.display_mode = DisplayMode::Amplitude;
                    }
                    self.audio = Some(audio);
                    // Trigger initial compute
                    self.computing = false;
                    self.trigger_compute(self.last_width_px);
                }
                WorkerResult::Computed { channels: results, cross } => {
                    self.computing       = false;
                    // The new texture matches the window it was computed for;
                    // drop the pan offset so it snaps cleanly into place.
                    self.tex_viewport    = self.pending_compute_viewport.clone();
                    self.panning         = false;
                    let mut scals = Vec::with_capacity(results.len());
                    let mut phs   = Vec::with_capacity(results.len());
                    let mut cohs  = Vec::with_capacity(results.len());
                    let mut idevs = Vec::with_capacity(results.len());
                    for (a, p, c, d) in results {
                        scals.push(a);
                        phs.push(p);
                        cohs.push(c);
                        idevs.push(d);
                    }
                    self.scalograms      = scals;
                    self.phases          = phs;
                    self.coherences      = cohs;
                    self.inst_devs       = idevs;
                    self.cross           = cross;
                    self.scalogram_width = self.last_width_px;
                    // Freeze the normalisation reference on the first compute
                    // after load; keep it stable across zoom / freq changes.
                    if !self.norm_captured {
                        self.capture_norm();
                        self.norm_captured = true;
                    }
                    self.rebuild_textures(ctx);
                    self.update_ridge();
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
) {
    let gpu = match GpuContext::new() {
        Ok(g)  => g,
        Err(e) => {
            let _ = tx.send(WorkerResult::Error(format!("GPU init: {}", e)));
            return;
        }
    };
    let mut engine = CwtEngine::new(gpu);

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
                // wgpu turns device errors (e.g. out-of-memory) into panics by
                // default; catch them so the UI reports instead of hanging in
                // "Computing…" forever with a dead worker.
                let computed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.compute_all_superlet(
                        &req.channels,
                        req.sample_rate,
                        req.t_start,
                        req.t_end,
                        req.width_pixels,
                        &req.params,
                        req.antialias,
                        req.unweighted,
                        req.superlet_order,
                    )
                }));
                let result = match computed {
                    Ok(Ok((channels, cross))) => WorkerResult::Computed { channels, cross },
                    Ok(Err(e)) => WorkerResult::Error(e.to_string()),
                    Err(_) => WorkerResult::Error(
                        "GPU compute failed — try fewer scales or a smaller window".into(),
                    ),
                };
                let _ = tx.send(result);
            }
        }
    }
}
