/// Available colour maps for the scalogram.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMap {
    Plasma,
    Viridis,
    Magma,
    Inferno,
    Hot,
}

/// What the scalogram image shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DisplayMode {
    /// Amplitude only, via the selected `ColorMap` (default).
    Amplitude,
    /// Phase only, via a cyclic hue colour wheel (minimal phase view).
    Phase,
    /// Phase as hue, amplitude as brightness (combined domain-colouring view).
    Combined,
    /// Instantaneous-frequency deviation as a diverging blue↔red map, amplitude
    /// as brightness — exposes phase "wobble" (pulling / slips) along a ridge.
    InstFreq,
}

impl DisplayMode {
    pub fn name(self) -> &'static str {
        match self {
            DisplayMode::Amplitude => "Amplitude",
            DisplayMode::Phase     => "Phase",
            DisplayMode::Combined  => "Phase+Amplitude",
            DisplayMode::InstFreq  => "Inst. frequency",
        }
    }
}

/// Reference baseline subtracted from the instantaneous frequency in the
/// `InstFreq` view (see [`instfreq_to_rgba`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InstFreqBaseline {
    /// `f_inst − f_i`: deviation from the row's nominal centre frequency. Clean
    /// on the ridge; off-ridge rows carry a static (vertical) colour gradient.
    Nominal,
    /// `f_inst − ⟨f_inst⟩_local`: deviation from the row's local time-average.
    /// Removes the off-ridge bias and slow drift, leaving only fast wobble.
    Detrend,
}

impl InstFreqBaseline {
    pub fn name(self) -> &'static str {
        match self {
            InstFreqBaseline::Nominal => "vs nominal",
            InstFreqBaseline::Detrend => "detrend (time)",
        }
    }
}

impl ColorMap {
    pub fn name(self) -> &'static str {
        match self {
            ColorMap::Plasma  => "Plasma",
            ColorMap::Viridis => "Viridis",
            ColorMap::Magma   => "Magma",
            ColorMap::Inferno => "Inferno",
            ColorMap::Hot     => "Hot",
        }
    }

    /// Map a value `t ∈ [0, 1]` to an RGB triple (each 0–255).
    pub fn apply(self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let (r, g, b) = match self {
            ColorMap::Plasma  => lerp_colormap(t, PLASMA),
            ColorMap::Viridis => lerp_colormap(t, VIRIDIS),
            ColorMap::Magma   => lerp_colormap(t, MAGMA),
            ColorMap::Inferno => lerp_colormap(t, INFERNO),
            ColorMap::Hot     => lerp_colormap(t, HOT),
        };
        [
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }
}

// --- Colour-map tables (breakpoints: t, r, g, b) ---

/// Piecewise-linear interpolation over a list of (t, r, g, b) stops.
fn lerp_colormap(t: f32, stops: &[(f32, f32, f32, f32)]) -> (f32, f32, f32) {
    if stops.is_empty() { return (t, t, t); }
    if t <= stops[0].0 { let s = stops[0]; return (s.1, s.2, s.3); }
    let last = stops[stops.len() - 1];
    if t >= last.0 { return (last.1, last.2, last.3); }

    for i in 1..stops.len() {
        let (t1, r1, g1, b1) = stops[i];
        let (t0, r0, g0, b0) = stops[i - 1];
        if t <= t1 {
            let frac = (t - t0) / (t1 - t0);
            return (
                r0 + frac * (r1 - r0),
                g0 + frac * (g1 - g0),
                b0 + frac * (b1 - b0),
            );
        }
    }
    let s = stops[stops.len() - 1];
    (s.1, s.2, s.3)
}

// Approximate breakpoints for standard matplotlib colour maps
const PLASMA: &[(f32, f32, f32, f32)] = &[
    (0.00, 0.050, 0.030, 0.527),
    (0.25, 0.432, 0.003, 0.681),
    (0.50, 0.796, 0.180, 0.451),
    (0.75, 0.974, 0.484, 0.136),
    (1.00, 0.940, 0.975, 0.131),
];

const VIRIDIS: &[(f32, f32, f32, f32)] = &[
    (0.00, 0.267, 0.005, 0.329),
    (0.25, 0.229, 0.322, 0.545),
    (0.50, 0.127, 0.566, 0.550),
    (0.75, 0.369, 0.789, 0.382),
    (1.00, 0.993, 0.906, 0.144),
];

const MAGMA: &[(f32, f32, f32, f32)] = &[
    (0.00, 0.001, 0.000, 0.014),
    (0.25, 0.316, 0.071, 0.484),
    (0.50, 0.737, 0.242, 0.491),
    (0.75, 0.977, 0.656, 0.636),
    (1.00, 0.988, 0.991, 0.750),
];

const INFERNO: &[(f32, f32, f32, f32)] = &[
    (0.00, 0.001, 0.000, 0.014),
    (0.25, 0.321, 0.067, 0.426),
    (0.50, 0.776, 0.214, 0.277),
    (0.75, 0.987, 0.628, 0.159),
    (1.00, 0.988, 0.998, 0.645),
];

const HOT: &[(f32, f32, f32, f32)] = &[
    (0.000, 0.000, 0.000, 0.000),
    (0.333, 1.000, 0.000, 0.000),
    (0.667, 1.000, 1.000, 0.000),
    (1.000, 1.000, 1.000, 1.000),
];

/// Logarithmic amplitude compression (monotonic, maps 0 → 0).
#[inline]
fn log_compress(v: f32) -> f32 {
    if v > 0.0 { (v * 1e6 + 1.0).ln() } else { 0.0 }
}

/// Map an amplitude `v` to a brightness in `[0, 1]`, given a fixed `[vmin, vmax]`
/// normalisation reference and a `log_amount ∈ [0, 1]` that blends smoothly
/// between linear (0.0) and logarithmic (1.0) scaling.
#[inline]
fn brightness(v: f32, vmin: f32, vmax: f32, log_amount: f32) -> f32 {
    let lin_range = (vmax - vmin).max(1e-30);
    let t_lin = ((v - vmin) / lin_range).clamp(0.0, 1.0);

    let lv    = log_compress(v);
    let lvmin = log_compress(vmin);
    let lvmax = log_compress(vmax);
    let log_range = (lvmax - lvmin).max(1e-30);
    let t_log = ((lv - lvmin) / log_range).clamp(0.0, 1.0);

    t_lin + (t_log - t_lin) * log_amount.clamp(0.0, 1.0)
}

/// Convert a float scalogram to a raw RGBA image.
/// `scalogram[s * width + col]` where s=0 is the lowest frequency.
/// Image row 0 corresponds to the highest frequency (top of display).
/// `vmin`/`vmax` are the fixed normalisation reference; `log_amount` blends
/// between linear (0.0) and logarithmic (1.0) brightness scaling.
pub fn scalogram_to_rgba(
    scalogram: &[f32],
    width: usize,
    num_scales: usize,
    colormap: ColorMap,
    vmin: f32,
    vmax: f32,
    log_amount: f32,
) -> Vec<u8> {
    let mut rgba = vec![0u8; num_scales * width * 4];
    for row in 0..num_scales {
        // Flip vertically: image row 0 = top = highest frequency
        let sc_row = num_scales - 1 - row;
        for col in 0..width {
            let v = scalogram[sc_row * width + col];
            let t = brightness(v, vmin, vmax, log_amount);
            let [r, g, b] = colormap.apply(t);
            let idx = (row * width + col) * 4;
            rgba[idx]     = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = 255;
        }
    }
    rgba
}

/// Convert HSV (all in 0..1, hue wraps) to an RGB triple in 0..1.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h - h.floor()) * 6.0;
    let i  = h6.floor() as i32;
    let f  = h6 - i as f32;
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Render phase (radians, in (-π, π]) as a cyclic hue wheel.
/// Layout matches `scalogram_to_rgba`: `phase[s*width+col]`, s=0 lowest freq,
/// image row 0 = top = highest frequency.
/// `coherence ∈ [0,1]` (phase-locking value) drives saturation: where the phase
/// is unresolvable within the pixel (rotating/aliased) it fades to white, so no
/// false rainbow colour appears. `gamma` shapes that fade (higher ⇒ stricter).
pub fn phase_to_rgba(
    phase: &[f32],
    coherence: &[f32],
    width: usize,
    num_scales: usize,
    gamma: f32,
) -> Vec<u8> {
    use std::f32::consts::PI;
    let mut rgba = vec![0u8; num_scales * width * 4];
    for row in 0..num_scales {
        let sc_row = num_scales - 1 - row;
        for col in 0..width {
            let ph  = phase[sc_row * width + col];
            let sat = coherence[sc_row * width + col].clamp(0.0, 1.0).powf(gamma);
            let hue = (ph + PI) / (2.0 * PI);   // map (-π, π] → [0, 1)
            let (r, g, b) = hsv_to_rgb(hue, sat, 1.0);
            let idx = (row * width + col) * 4;
            rgba[idx]     = (r.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 3] = 255;
        }
    }
    rgba
}

/// Render phase as hue and amplitude as brightness ("domain colouring").
/// Amplitude is normalised the same way as `scalogram_to_rgba`.
#[allow(clippy::too_many_arguments)]
pub fn combined_to_rgba(
    amplitude:  &[f32],
    phase:      &[f32],
    coherence:  &[f32],
    width:      usize,
    num_scales: usize,
    vmin:       f32,
    vmax:       f32,
    log_amount: f32,
    gamma:      f32,
) -> Vec<u8> {
    use std::f32::consts::PI;

    let mut rgba = vec![0u8; num_scales * width * 4];
    for row in 0..num_scales {
        let sc_row = num_scales - 1 - row;
        for col in 0..width {
            let val = brightness(amplitude[sc_row * width + col], vmin, vmax, log_amount);
            let ph  = phase[sc_row * width + col];
            let sat = coherence[sc_row * width + col].clamp(0.0, 1.0).powf(gamma);
            let hue = (ph + PI) / (2.0 * PI);
            let (r, g, b) = hsv_to_rgb(hue, sat, val);
            let idx = (row * width + col) * 4;
            rgba[idx]     = (r.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 3] = 255;
        }
    }
    rgba
}

/// Diverging blue↔neutral↔red ("coolwarm") stops for the instantaneous-frequency
/// view: blue below the baseline, light grey on it, red above.
const COOLWARM: &[(f32, f32, f32, f32)] = &[
    (0.00, 0.23, 0.45, 0.85),
    (0.50, 0.86, 0.86, 0.86),
    (1.00, 0.85, 0.28, 0.21),
];

/// Render the instantaneous-frequency deviation as a diverging blue↔red map with
/// amplitude as brightness. `inst_dev[s*width+col]` is the relative deviation
/// (f_inst − f_i)/f_i (0 = exactly on the row's centre frequency). `baseline`
/// selects what is subtracted; `range` is the ± full-scale (relative) mapped to
/// the colour ends; `detrend_win` is the moving-average window (pixels) used by
/// `Detrend`. Amplitude is normalised exactly like `scalogram_to_rgba`, so silent
/// regions fade to black and only energetic ridges reveal their phase wobble.
#[allow(clippy::too_many_arguments)]
pub fn instfreq_to_rgba(
    inst_dev:    &[f32],
    amplitude:   &[f32],
    width:       usize,
    num_scales:  usize,
    baseline:    InstFreqBaseline,
    range:       f32,
    detrend_win: usize,
    vmin:        f32,
    vmax:        f32,
    log_amount:  f32,
) -> Vec<u8> {
    let mut rgba = vec![0u8; num_scales * width * 4];
    let full = range.max(1e-6);
    let half = (detrend_win.max(1) / 2) as isize;
    let mut psum = vec![0.0f32; width + 1];   // per-row prefix sum (reused)

    for row in 0..num_scales {
        let sc_row = num_scales - 1 - row;
        let off = sc_row * width;

        // Per-row prefix sum → moving-average baseline for the Detrend mode.
        if baseline == InstFreqBaseline::Detrend {
            for c in 0..width {
                psum[c + 1] = psum[c] + inst_dev[off + c];
            }
        }

        for col in 0..width {
            let r = inst_dev[off + col];
            let v = match baseline {
                InstFreqBaseline::Nominal => r,
                InstFreqBaseline::Detrend => {
                    let lo = (col as isize - half).max(0) as usize;
                    let hi = ((col as isize + half + 1).max(0) as usize).min(width);
                    let mean = (psum[hi] - psum[lo]) / (hi - lo).max(1) as f32;
                    r - mean
                }
            };
            let t = (0.5 + v / (2.0 * full)).clamp(0.0, 1.0);
            let (mut cr, mut cg, mut cb) = lerp_colormap(t, COOLWARM);
            let gate = brightness(amplitude[off + col], vmin, vmax, log_amount);
            cr *= gate;
            cg *= gate;
            cb *= gate;
            let idx = (row * width + col) * 4;
            rgba[idx]     = (cr.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (cg.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (cb.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 3] = 255;
        }
    }
    rgba
}
