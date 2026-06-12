use std::borrow::Cow;
use std::f64::consts::PI;

use crate::gpu::{ExtractParams, GpuContext, MulParams, WavParams};
use crate::wavelet::CwtParams;

// ---------------------------------------------------------------------------
// Anti-aliasing decimation: cascade of zero-phase half-band FIR stages
// ---------------------------------------------------------------------------
//
// Decimation factors are restricted to powers of two; each ÷2 stage applies a
// 63-tap Blackman-windowed half-band FIR (linear phase, centre-aligned ⇒ zero
// phase at the decimated grid). Passband is flat to ~0.41 of the output
// Nyquist... in stage units: flat to 0.206·fs_in (−74 dB stopband from
// 0.294·fs_in), so with the d ≤ fs/(2.5·f_max) margin the displayed band
// [0, f_max] ⊂ [0, 0.4·f_ds] sits entirely in the flat passband — no
// amplitude droop and no phase distortion, unlike the former IIR cascade.

const HALFBAND_TAPS: usize = 63;

fn halfband_taps() -> Vec<f32> {
    let c = (HALFBAND_TAPS / 2) as isize; // centre tap index (31)
    let mut h = vec![0.0f64; HALFBAND_TAPS];
    for (n, tap) in h.iter_mut().enumerate() {
        let k = n as isize - c;
        let ideal = if k == 0 {
            0.5
        } else {
            let x = 0.5 * k as f64;
            0.5 * (PI * x).sin() / (PI * x)
        };
        let t = n as f64 / (HALFBAND_TAPS - 1) as f64;
        let w = 0.42 - 0.5 * (2.0 * PI * t).cos() + 0.08 * (4.0 * PI * t).cos();
        *tap = ideal * w;
    }
    let sum: f64 = h.iter().sum();
    h.iter().map(|v| (v / sum) as f32).collect()
}

/// Filter + decimate by 2. Output sample `m` is the lowpassed input at `2m`
/// (centre-aligned symmetric FIR ⇒ no time shift). Edges are zero-extended;
/// the CWT padding absorbs the resulting transients.
fn halfband_decim2(x: &[f32], h: &[f32]) -> Vec<f32> {
    let c = h.len() / 2;
    let n = x.len();
    let m = n / 2;
    let mut y = vec![0.0f32; m];
    for (i, out) in y.iter_mut().enumerate() {
        let t = 2 * i;
        let mut acc = h[c] * x[t];
        // Half-band: only odd offsets from the centre are non-zero.
        let mut k = 1usize;
        while k <= c {
            let hk = h[c + k];
            let xa = if t >= k { x[t - k] } else { 0.0 };
            let xb = if t + k < n { x[t + k] } else { 0.0 };
            acc += hk * (xa + xb);
            k += 2;
        }
        *out = acc;
    }
    y
}

/// Downsample by power-of-two `d`. With `antialias` the half-band cascade is
/// used; otherwise a plain boxcar average over each block of `d` samples.
fn downsample(seg: &[f32], d: usize, antialias: bool, taps: &[f32]) -> Vec<f32> {
    if antialias {
        let mut cur = seg.to_vec();
        let mut dd = d;
        while dd > 1 {
            cur = halfband_decim2(&cur, taps);
            dd /= 2;
        }
        cur
    } else {
        seg.chunks(d)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    }
}

fn floor_pow2(x: usize) -> usize {
    if x <= 1 { 1 } else { 1usize << (usize::BITS - 1 - x.leading_zeros()) }
}

// ---------------------------------------------------------------------------
// CWT engine
// ---------------------------------------------------------------------------

/// CWT outputs for one channel, each laid out `[scale * width + col]`:
/// (amplitude, phase, coherence, instantaneous-frequency deviation).
pub type CwtOutput = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Cross-channel wavelet spectrum (first two channels), `[scale * width + col]`.
pub struct CrossOutput {
    /// arg(Σ W₀·W̄₁): phase difference φ₀ − φ₁ per pixel.
    pub phase: Vec<f32>,
    /// |Σ W₀·W̄₁| / Σ|W₀||W₁| ∈ [0, 1]: cross phase-locking within the pixel.
    pub coherence: Vec<f32>,
    /// Geometric-mean amplitude √(|W₀||W₁|), pixel-averaged.
    pub amplitude: Vec<f32>,
}

/// FFT length cap per chunk; longer views are split into time chunks.
const N_FFT_CAP: usize = 1 << 20;
/// Per-buffer VRAM budget for the big (rows × n_fft complex) buffers; the
/// scale axis is processed in batches that fit this (4 such buffers live at
/// once in stereo). Further capped by the device's max binding size.
const BIG_BUF_BUDGET: u64 = 192 << 20;

/// All GPU buffers for one compute configuration, cached between recomputes
/// so consecutive pans/zooms with unchanged sizes reallocate nothing.
struct Bufs {
    n_fft: usize,
    batch: usize,
    width: usize,
    num_scales: usize,
    nch: usize,
    d_real: Vec<wgpu::Buffer>,    // per channel, n_fft f32
    d_complex: wgpu::Buffer,      // n_fft complex (shared temp)
    d_fft_scratch: wgpu::Buffer,  // n_fft complex (forward-FFT ping-pong)
    d_sig_fft: Vec<wgpu::Buffer>, // per channel, n_fft complex
    d_a: wgpu::Buffer,            // batch big: wavelet freqs / IFFT scratch
    d_b: wgpu::Buffer,            // batch big: products
    d_cwt: Vec<wgpu::Buffer>,     // per channel, batch big
    d_scales: wgpu::Buffer,
    d_eta: wgpu::Buffer,
    d_bounds: wgpu::Buffer,
    d_out: Vec<wgpu::Buffer>,     // 4 per channel (+3 cross when stereo)
    s_out: Vec<wgpu::Buffer>,     // staging mirrors of d_out
}

impl Bufs {
    fn create(
        gpu: &GpuContext,
        n_fft: usize,
        batch: usize,
        width: usize,
        num_scales: usize,
        nch: usize,
    ) -> Self {
        let bytes_real = (n_fft * 4) as u64;
        let bytes_c    = (n_fft * 8) as u64;
        let bytes_big  = (batch * n_fft * 8) as u64;
        let bytes_out  = (num_scales * width * 4) as u64;
        let bytes_sc   = (num_scales * 4) as u64;
        let bytes_bnd  = ((width + 1) * 4) as u64;
        let n_out = 4 * nch + if nch == 2 { 3 } else { 0 };
        Bufs {
            n_fft,
            batch,
            width,
            num_scales,
            nch,
            d_real:    (0..nch).map(|_| gpu.storage(bytes_real)).collect(),
            d_complex: gpu.storage(bytes_c),
            d_fft_scratch: gpu.storage(bytes_c),
            d_sig_fft: (0..nch).map(|_| gpu.storage(bytes_c)).collect(),
            d_a:       gpu.storage(bytes_big),
            d_b:       gpu.storage(bytes_big),
            d_cwt:     (0..nch).map(|_| gpu.storage(bytes_big)).collect(),
            d_scales:  gpu.storage(bytes_sc),
            d_eta:     gpu.storage(bytes_sc),
            d_bounds:  gpu.storage(bytes_bnd),
            d_out:     (0..n_out).map(|_| gpu.storage(bytes_out)).collect(),
            s_out:     (0..n_out).map(|_| gpu.staging(bytes_out)).collect(),
        }
    }
}

pub struct CwtEngine {
    gpu: GpuContext,
    taps: Vec<f32>,
    bufs: Option<Bufs>,
}

impl CwtEngine {
    pub fn new(gpu: GpuContext) -> Self {
        CwtEngine { gpu, taps: halfband_taps(), bufs: None }
    }

    /// Compute the CWT scalogram for all channels of one file. The first two
    /// channels are processed jointly so their cross spectrum (relative phase
    /// + coherence) comes along for free; further channels run individually.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_all(
        &mut self,
        channels:     &[Vec<f32>],
        sample_rate:  u32,
        t_start:      usize,
        t_end:        usize,
        width_pixels: usize,
        params:       &CwtParams,
        antialias:    bool,
        unweighted:   bool,
    ) -> anyhow::Result<(Vec<CwtOutput>, Option<CrossOutput>)> {
        match channels {
            [] => Ok((Vec::new(), None)),
            [one] => self.run(
                &[one.as_slice()],
                sample_rate, t_start, t_end, width_pixels, params, antialias, unweighted,
            ),
            [a, b, rest @ ..] => {
                let (mut outs, cross) = self.run(
                    &[a.as_slice(), b.as_slice()],
                    sample_rate, t_start, t_end, width_pixels, params, antialias, unweighted,
                )?;
                for ch in rest {
                    let (mut single, _) = self.run(
                        &[ch.as_slice()],
                        sample_rate, t_start, t_end, width_pixels, params, antialias, unweighted,
                    )?;
                    outs.append(&mut single);
                }
                Ok((outs, cross))
            }
        }
    }

    /// Multiplicative superlets (Moca et al. 2021): the amplitude is replaced
    /// by the geometric mean over `order` passes whose frequency-sharpness
    /// parameter is scaled ×1, ×2, … ×order. Phase, coherence, inst-freq and
    /// the cross spectrum come from the base (×1) pass. `order ≤ 1` is a
    /// plain [`Self::compute_all`].
    #[allow(clippy::too_many_arguments)]
    pub fn compute_all_superlet(
        &mut self,
        channels:     &[Vec<f32>],
        sample_rate:  u32,
        t_start:      usize,
        t_end:        usize,
        width_pixels: usize,
        params:       &CwtParams,
        antialias:    bool,
        unweighted:   bool,
        order:        u32,
    ) -> anyhow::Result<(Vec<CwtOutput>, Option<CrossOutput>)> {
        let (mut outs, cross) = self.compute_all(
            channels, sample_rate, t_start, t_end, width_pixels,
            params, antialias, unweighted,
        )?;
        if order <= 1 {
            return Ok((outs, cross));
        }
        // f64 accumulators: amplitudes span many decades and an f32 product
        // of `order` of them underflows.
        let mut prods: Vec<Vec<f64>> = outs
            .iter()
            .map(|(a, ..)| a.iter().map(|&v| v as f64).collect())
            .collect();
        for k in 2..=order {
            let pk = params.superlet_pass(k);
            let (outs_k, _) = self.compute_all(
                channels, sample_rate, t_start, t_end, width_pixels,
                &pk, antialias, unweighted,
            )?;
            for (prod, (a, ..)) in prods.iter_mut().zip(&outs_k) {
                for (p, &v) in prod.iter_mut().zip(a) {
                    *p *= v as f64;
                }
            }
        }
        let inv = 1.0 / order as f64;
        for ((a, ..), prod) in outs.iter_mut().zip(prods) {
            for (dst, p) in a.iter_mut().zip(prod) {
                *dst = p.powf(inv) as f32;
            }
        }
        Ok((outs, cross))
    }

    /// Core pipeline for one or two signals (two ⇒ cross spectrum included).
    ///
    /// The visible window is processed in time chunks (each FFT capped at
    /// `N_FFT_CAP`) and the scale axis in batches sized to the VRAM budget,
    /// so arbitrarily long views compute fully instead of being truncated.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        sigs:         &[&[f32]],
        sample_rate:  u32,
        t_start:      usize,
        t_end:        usize,
        width_pixels: usize,
        params:       &CwtParams,
        antialias:    bool,
        unweighted:   bool,
    ) -> anyhow::Result<(Vec<CwtOutput>, Option<CrossOutput>)> {
        let nch        = sigs.len();
        let num_scales = params.num_scales;
        let width      = width_pixels;

        let zeros = || -> (Vec<CwtOutput>, Option<CrossOutput>) {
            let z = || vec![0.0f32; num_scales * width];
            let outs = (0..nch).map(|_| (z(), z(), z(), z())).collect();
            let cross = (nch == 2).then(|| CrossOutput {
                phase: z(), coherence: z(), amplitude: z(),
            });
            (outs, cross)
        };

        let total   = sigs.iter().map(|s| s.len()).min().unwrap_or(0);
        let t_start = t_start.min(total);
        let t_end   = t_end.min(total);
        if t_start >= t_end || width == 0 || num_scales == 0 {
            return Ok(zeros());
        }
        let visible_samples = t_end - t_start;

        // ---- decimation factor (power of two) -----------------------------
        // One pixel should represent at most `visible/width` original samples,
        // but keep f_ds ≥ 2.5·f_max so the displayed band stays inside the
        // decimator's flat passband and the per-sample phase increment at
        // f_max (2π·f_max/f_ds ≤ 0.8π) has wrap headroom for the inst-freq
        // estimator.
        let max_d = ((sample_rate as f64) / (2.5 * params.f_max as f64))
            .floor()
            .max(1.0) as usize;
        let d    = floor_pow2((visible_samples / width).max(1).min(max_d));
        let f_ds = sample_rate as f64 / d as f64;

        // ---- padding (wavelet support at the lowest frequency) ------------
        let f_min     = (params.f_min as f64).max(0.1);
        let f_max_eff = (params.f_max as f64).min(f_ds / 2.0 * 0.99).max(f_min * 1.01);
        let eta_low    = params.peak_eta(0.0);
        let s_max_orig = eta_low * sample_rate as f64 / (2.0 * PI * f_min);
        let pad_cycles = params.support_cycles() * sample_rate as f64 / f_min;
        let padding = ((5.0 * s_max_orig).max(pad_cycles) as usize)
            .max(64 * d)              // room for the FIR decimator tail
            .min(total / 2)
            .max(16);

        // ---- downsampled tape ----------------------------------------------
        let seg_start = t_start.saturating_sub(padding);
        let seg_end   = (t_end + padding).min(total);
        let tapes: Vec<Cow<[f32]>> = sigs
            .iter()
            .map(|s| {
                if d == 1 {
                    Cow::Borrowed(&s[seg_start..seg_end])
                } else {
                    Cow::Owned(downsample(&s[seg_start..seg_end], d, antialias, &self.taps))
                }
            })
            .collect();
        let tape_len = tapes.iter().map(|t| t.len()).min().unwrap_or(0);

        let pre_pad = t_start - seg_start;
        let g_vs    = (pre_pad / d).min(tape_len);          // first valid tape sample
        let g_ve    = (g_vs + visible_samples / d + 1).min(tape_len);
        if tape_len < 4 || g_vs >= g_ve {
            return Ok(zeros());
        }
        let valid_len = g_ve - g_vs;

        // Exact column → tape-sample bounds (integer math; uploaded to the GPU
        // so the kernels avoid f32 index arithmetic on long windows).
        let mut bounds = vec![0u32; width + 1];
        for (c, b) in bounds.iter_mut().enumerate() {
            *b = (g_vs + c * valid_len / width) as u32;
        }

        // ---- time-chunk plan ------------------------------------------------
        // Each chunk holds the tape samples of a run of whole columns plus the
        // wavelet-support padding on both sides; `slack` keeps the circular
        // convolution wrap away from the valid region even when a chunk fills
        // its FFT exactly.
        let pad_ds = (padding / d).clamp(1, N_FFT_CAP / 4);
        let slack  = pad_ds.clamp(64, 4096);
        let (n_fft, chunks): (usize, Vec<(usize, usize, usize, usize)>) =
            if tape_len + slack <= N_FFT_CAP {
                let n = (tape_len + slack).next_power_of_two().clamp(64, N_FFT_CAP);
                (n, vec![(0, width, 0, tape_len)])
            } else {
                let usable = N_FFT_CAP - 2 * pad_ds - slack;
                let mut list = Vec::new();
                let mut c0 = 0usize;
                while c0 < width {
                    let mut c1 = c0 + 1;
                    while c1 < width
                        && (bounds[c1 + 1] - bounds[c0]) as usize <= usable
                    {
                        c1 += 1;
                    }
                    let lo = (bounds[c0] as usize).saturating_sub(pad_ds);
                    let hi = ((bounds[c1] as usize) + pad_ds).min(tape_len);
                    list.push((c0, c1, lo, hi - lo));
                    c0 = c1;
                }
                (N_FFT_CAP, list)
            };

        // ---- scales + per-scale η (log-spaced from f_min to f_max_eff) ---
        let mut scales = vec![0.0f32; num_scales];
        let mut etas   = vec![0.0f32; num_scales];
        for i in 0..num_scales {
            let frac  = i as f64 / (num_scales - 1).max(1) as f64;
            let f_i   = f_min * (f_max_eff / f_min).powf(frac);
            let eta_i = params.peak_eta(frac);
            scales[i] = (eta_i * f_ds / (2.0 * PI * f_i)) as f32;
            etas[i]   = eta_i as f32;
        }

        // ---- scale batch size from the VRAM budget ------------------------
        let max_bind = self.gpu.device.limits().max_storage_buffer_binding_size as u64;
        let budget   = BIG_BUF_BUDGET.min(max_bind);
        let row_bytes = (n_fft * 8) as u64;
        let batch = ((budget / row_bytes).max(1) as usize).min(num_scales);

        // ---- GPU buffers (cached between identically-sized computes) ------
        let stale = match &self.bufs {
            Some(b) => b.n_fft != n_fft || b.batch != batch || b.width != width
                || b.num_scales != num_scales || b.nch != nch,
            None => true,
        };
        if stale {
            self.bufs = Some(Bufs::create(&self.gpu, n_fft, batch, width, num_scales, nch));
        }
        let bufs = self.bufs.as_ref().unwrap();
        let gpu  = &self.gpu;

        gpu.upload_f32(&bufs.d_scales, &scales);
        gpu.upload_f32(&bufs.d_eta, &etas);
        gpu.queue.write_buffer(&bufs.d_bounds, 0, bytemuck::cast_slice(&bounds));

        let n32        = n_fft as u32;
        let (p1, p2)   = params.kernel_params();
        let kind32     = params.kind.code() as u32;

        // ---- encode & submit per chunk -------------------------------------
        let mut padded = vec![0.0f32; n_fft];
        for &(c0, c1, lo, len) in &chunks {
            let copy = len.min(n_fft);
            for (chi, tape) in tapes.iter().enumerate() {
                padded.fill(0.0);
                padded[..copy].copy_from_slice(&tape[lo..lo + copy]);
                gpu.upload_f32(&bufs.d_real[chi], &padded);
            }

            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // Forward FFT of each channel's chunk.
            for chi in 0..nch {
                gpu.real_to_complex(&mut enc, &bufs.d_real[chi], &bufs.d_complex, n32);
                gpu.fft(
                    &mut enc,
                    &bufs.d_complex,
                    &bufs.d_sig_fft[chi],
                    &bufs.d_fft_scratch,
                    n32, 1, false,
                );
            }

            let valid_end_local = (g_ve - lo).min(copy) as i32;
            let mut off = 0usize;
            while off < num_scales {
                let rows = batch.min(num_scales - off);
                let ep = ExtractParams {
                    n: n32,
                    rows: rows as u32,
                    scale_offset: off as u32,
                    width: width as u32,
                    col_start: c0 as u32,
                    col_count: (c1 - c0) as u32,
                    chunk_lo: lo as i32,
                    chunk_len: copy as i32,
                    valid_end: valid_end_local,
                    unit_weight: unweighted as u32,
                    _pad: [0; 2],
                };
                for chi in 0..nch {
                    // The wavelet bank is regenerated per channel because the
                    // IFFT consumes d_a as its ping-pong scratch; the kernel
                    // is far cheaper than one FFT stage.
                    gpu.wavelet(
                        &mut enc, &bufs.d_a, &bufs.d_scales, &bufs.d_eta,
                        WavParams {
                            n: n32, num_scales: rows as u32, scale_offset: off as u32,
                            kind: kind32, p1, p2, _pad: [0; 2],
                        },
                    );
                    gpu.multiply(
                        &mut enc, &bufs.d_sig_fft[chi], &bufs.d_a, &bufs.d_b,
                        MulParams { n: n32, num_scales: rows as u32, _pad: [0; 2] },
                    );
                    gpu.fft(&mut enc, &bufs.d_b, &bufs.d_cwt[chi], &bufs.d_a, n32, rows as u32, true);
                    let o = chi * 4;
                    gpu.extract(
                        &mut enc, &bufs.d_cwt[chi],
                        &bufs.d_out[o], &bufs.d_out[o + 1], &bufs.d_out[o + 2], &bufs.d_out[o + 3],
                        &bufs.d_scales, &bufs.d_eta, &bufs.d_bounds, ep,
                    );
                }
                if nch == 2 {
                    gpu.cross_extract(
                        &mut enc, &bufs.d_cwt[0], &bufs.d_cwt[1],
                        &bufs.d_out[8], &bufs.d_out[9], &bufs.d_out[10],
                        &bufs.d_bounds, ep,
                    );
                }
                off += rows;
            }
            gpu.queue.submit(Some(enc.finish()));
        }

        // ---- download -------------------------------------------------------
        let bytes_out = (num_scales * width * 4) as u64;
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (d_buf, s_buf) in bufs.d_out.iter().zip(&bufs.s_out) {
            enc.copy_buffer_to_buffer(d_buf, 0, s_buf, 0, bytes_out);
        }
        gpu.queue.submit(Some(enc.finish()));

        let n_out = num_scales * width;
        let read  = |i: usize| gpu.download_f32(&bufs.s_out[i], n_out);
        let mut outs = Vec::with_capacity(nch);
        for chi in 0..nch {
            let o = chi * 4;
            outs.push((read(o), read(o + 1), read(o + 2), read(o + 3)));
        }
        let cross = (nch == 2).then(|| CrossOutput {
            phase: read(8),
            coherence: read(9),
            amplitude: read(10),
        });

        Ok((outs, cross))
    }
}

// ---------------------------------------------------------------------------
// Display-side synchrosqueezing
// ---------------------------------------------------------------------------

/// Pixel-level synchrosqueezing: reassign each pixel's amplitude along the
/// frequency axis to the row matching its mean instantaneous frequency.
/// Rows are log-spaced over `[f_min, f_max]`, so the shift depends only on
/// the relative deviation: Δrow = (ns−1)·ln(1+dev)/`log_ratio`, where
/// `log_ratio` = ln(f_max/f_min). The amplitude is split linearly between the
/// two nearest rows; pixels whose target lies outside the grid are dropped
/// (their energy belongs off-screen). Mass-preserving: ridges concentrate,
/// so peak values grow — re-normalise brightness if they saturate.
pub fn synchrosqueeze(
    amp:        &[f32],
    dev:        &[f32],
    width:      usize,
    num_scales: usize,
    log_ratio:  f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; amp.len()];
    let top = (num_scales - 1) as f32;
    let rows_per_ln = top / log_ratio.max(1e-6);
    for s in 0..num_scales {
        for c in 0..width {
            let i = s * width + c;
            let a = amp[i];
            if a <= 0.0 {
                continue;
            }
            let t = s as f32 + (1.0 + dev[i]).max(1e-6).ln() * rows_per_ln;
            // Also rejects NaN deviations.
            if !(t >= 0.0 && t <= top) {
                continue;
            }
            let r0 = t.floor() as usize;
            let frac = t - r0 as f32;
            out[r0 * width + c] += a * (1.0 - frac);
            if r0 + 1 < num_scales {
                out[(r0 + 1) * width + c] += a * frac;
            }
        }
    }
    out
}

/// Phase-aware variant of [`synchrosqueeze`]: reassigns amplitude identically
/// while also accumulating the complex sum Σ a·e^{iφ} per target pixel.
/// Returns (amplitude, phase, coherence) where phase = arg(Σ a·e^{iφ}) and
/// coherence = |Σ a·e^{iφ}| / Σa ∈ [0, 1] — the amplitude-weighted phase
/// agreement of the mass that landed in the bin, so bins fed by pixels with
/// conflicting phases desaturate in the combined view.
pub fn synchrosqueeze_with_phase(
    amp:        &[f32],
    phase:      &[f32],
    dev:        &[f32],
    width:      usize,
    num_scales: usize,
    log_ratio:  f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out_a  = vec![0.0f32; amp.len()];
    let mut out_re = vec![0.0f32; amp.len()];
    let mut out_im = vec![0.0f32; amp.len()];
    let top = (num_scales - 1) as f32;
    let rows_per_ln = top / log_ratio.max(1e-6);
    for s in 0..num_scales {
        for c in 0..width {
            let i = s * width + c;
            let a = amp[i];
            if a <= 0.0 {
                continue;
            }
            let t = s as f32 + (1.0 + dev[i]).max(1e-6).ln() * rows_per_ln;
            // Also rejects NaN deviations.
            if !(t >= 0.0 && t <= top) {
                continue;
            }
            let (sin, cos) = phase[i].sin_cos();
            let r0 = t.floor() as usize;
            let frac = t - r0 as f32;
            let j  = r0 * width + c;
            let w0 = a * (1.0 - frac);
            out_a[j]  += w0;
            out_re[j] += w0 * cos;
            out_im[j] += w0 * sin;
            if r0 + 1 < num_scales {
                let w1 = a * frac;
                out_a[j + width]  += w1;
                out_re[j + width] += w1 * cos;
                out_im[j + width] += w1 * sin;
            }
        }
    }
    // Collapse the complex sums into phase / coherence in place
    // (out_re becomes phase, out_im becomes coherence).
    for i in 0..amp.len() {
        let (re, im) = (out_re[i], out_im[i]);
        if out_a[i] > 0.0 {
            out_re[i] = im.atan2(re);
            out_im[i] = (re.hypot(im) / out_a[i]).min(1.0);
        }
    }
    (out_a, out_re, out_im)
}

// ---------------------------------------------------------------------------
// Tests. GPU tests self-skip when no adapter is available.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet::{CwtParams, WaveletKind};

    fn tone(f_rel: f64, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * f_rel * i as f64).sin() as f32).collect()
    }

    fn rms_mid(x: &[f32]) -> f64 {
        let a = &x[x.len() / 4..3 * x.len() / 4];
        (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// CPU-only: the half-band decimator must pass the band of interest
    /// untouched and crush everything that would fold back onto it.
    #[test]
    fn halfband_passband_and_stopband() {
        let taps = halfband_taps();
        let n = 8192;
        let r0 = rms_mid(&tone(0.10, n));
        let pass = halfband_decim2(&tone(0.10, n), &taps); // 0.10·fs: passband
        let stop = halfband_decim2(&tone(0.40, n), &taps); // 0.40·fs: stopband
        let g_pass = rms_mid(&pass) / r0;
        let g_stop = rms_mid(&stop) / r0;
        assert!((g_pass - 1.0).abs() < 0.01, "passband gain {g_pass}");
        assert!(g_stop < 0.01, "stopband gain {g_stop}");
    }

    /// CPU-only: a pixel whose instantaneous frequency sits exactly one
    /// log-grid row above its own row must hand its amplitude to that row;
    /// one whose target lies above the grid must be dropped.
    #[test]
    fn synchrosqueeze_reassigns_to_inst_freq_row() {
        let ns = 4usize;
        let width = 1usize;
        let log_ratio = 0.3f32;                 // grid step δ = 0.1 per row
        let delta = log_ratio / (ns - 1) as f32;

        let mut amp = vec![0.0f32; ns];
        let mut dev = vec![0.0f32; ns];
        amp[1] = 1.0;
        dev[1] = delta.exp() - 1.0;             // f_inst exactly one row up
        amp[3] = 1.0;
        dev[3] = (2.0 * delta).exp() - 1.0;     // target above the grid

        let out = synchrosqueeze(&amp, &dev, width, ns, log_ratio);
        assert!((out[2] - 1.0).abs() < 1e-4, "row 2 got {out:?}");
        assert!(out[1].abs() < 1e-4 && out[3].abs() < 1e-4, "{out:?}");
    }

    /// CPU-only: the phase-aware variant must carry the source phase to the
    /// target row with full coherence when contributors agree, and cancel
    /// the coherence (amplitude intact) when opposite phases share a bin.
    #[test]
    fn synchrosqueeze_with_phase_carries_and_cancels() {
        let ns = 4usize;
        let width = 1usize;
        let log_ratio = 0.3f32;
        let delta = log_ratio / (ns - 1) as f32;

        // Rows 1 and 3 both target row 2 (one row up / one row down).
        let mut amp = vec![0.0f32; ns];
        let mut dev = vec![0.0f32; ns];
        let mut ph  = vec![0.0f32; ns];
        amp[1] = 1.0;
        dev[1] = delta.exp() - 1.0;
        ph[1]  = 1.0;
        amp[3] = 1.0;
        dev[3] = (-delta).exp() - 1.0;
        ph[3]  = 1.0;

        let (a, p, c) = synchrosqueeze_with_phase(&amp, &ph, &dev, width, ns, log_ratio);
        assert!((a[2] - 2.0).abs() < 1e-4, "amp {a:?}");
        assert!((p[2] - 1.0).abs() < 1e-4, "phase {p:?}");
        assert!((c[2] - 1.0).abs() < 1e-4, "coherence {c:?}");

        // Opposite phases: amplitude still adds, complex sum cancels.
        ph[3] = 1.0 - std::f32::consts::PI;
        let (a, _, c) = synchrosqueeze_with_phase(&amp, &ph, &dev, width, ns, log_ratio);
        assert!((a[2] - 2.0).abs() < 1e-4, "amp {a:?}");
        assert!(c[2].abs() < 1e-4, "coherence {c:?}");
    }

    macro_rules! engine_or_skip {
        () => {
            match GpuContext::new() {
                Ok(g) => CwtEngine::new(g),
                Err(e) => {
                    eprintln!("skipping GPU test (no adapter): {e}");
                    return;
                }
            }
        };
    }

    fn peak_row(scalo: &[f32], ns: usize, width: usize) -> usize {
        let (mut best, mut best_v) = (0usize, f32::MIN);
        for s in 0..ns {
            let sum: f32 = (0..width).map(|c| scalo[s * width + c]).sum();
            if sum > best_v {
                best_v = sum;
                best = s;
            }
        }
        best
    }

    fn row_freq(params: &CwtParams, ns: usize, row: usize) -> f64 {
        let frac = row as f64 / (ns - 1) as f64;
        params.f_min as f64 * (params.f_max as f64 / params.f_min as f64).powf(frac)
    }

    /// A pure tone must light up the scalogram row whose centre frequency
    /// matches the tone. Exercises the whole GPU pipeline.
    #[test]
    fn scalogram_peaks_at_tone_frequency() {
        let mut engine = engine_or_skip!();
        let sr = 44_100u32;
        let f0 = 2_000.0f64;
        let n = 16_384usize;
        let signal = tone(f0 / sr as f64, n);

        let params = CwtParams {
            num_scales: 96,
            f_min: 500.0,
            f_max: 8_000.0,
            kind: WaveletKind::Morlet,
            omega0_low: 6.0,
            omega0_high: 6.0,
            ..CwtParams::default()
        };
        let width = 16usize;
        let (outs, _cross) = engine
            .compute_all(&[signal], sr, 0, n, width, &params, false, false)
            .expect("compute failed");
        let scalo = &outs[0].0;

        let ns = params.num_scales;
        let best = peak_row(scalo, ns, width);
        let f_peak = row_freq(&params, ns, best);
        let rel = (f_peak - f0).abs() / f0;
        assert!(rel < 0.12, "peak row freq {f_peak:.1} Hz vs {f0} Hz (rel {rel:.3})");
    }

    /// A long input must be computed in full (time-chunked), not silently
    /// truncated at the FFT cap: every column carries energy and the peak row
    /// still matches the tone.
    #[test]
    fn long_input_not_truncated() {
        let mut engine = engine_or_skip!();
        let sr = 44_100u32;
        let f0 = 2_000.0f64;
        let n = 3_000_000usize; // ≈ 68 s; tape exceeds the 1M FFT cap
        let signal = tone(f0 / sr as f64, n);

        let params = CwtParams {
            num_scales: 64,
            f_min: 500.0,
            f_max: 8_000.0,
            kind: WaveletKind::Morlet,
            ..CwtParams::default()
        };
        let width = 32usize;
        let (outs, _) = engine
            .compute_all(&[signal], sr, 0, n, width, &params, true, false)
            .expect("compute failed");
        let scalo = &outs[0].0;
        let ns = params.num_scales;

        for col in 0..width {
            let e: f32 = (0..ns).map(|s| scalo[s * width + col]).sum();
            assert!(e > 1e-6, "column {col} is empty (truncated view?)");
        }
        let best = peak_row(scalo, ns, width);
        let f_peak = row_freq(&params, ns, best);
        let rel = (f_peak - f0).abs() / f0;
        assert!(rel < 0.12, "peak row freq {f_peak:.1} Hz vs {f0} Hz (rel {rel:.3})");
    }

    /// A superlet must keep the tone's peak row and sharpen the frequency
    /// profile (fewer rows above half-maximum) relative to the base pass.
    #[test]
    fn superlet_sharpens_tone_peak() {
        let mut engine = engine_or_skip!();
        let sr = 44_100u32;
        let f0 = 2_000.0f64;
        let n = 16_384usize;
        let signal = tone(f0 / sr as f64, n);

        let params = CwtParams {
            num_scales: 96,
            f_min: 500.0,
            f_max: 8_000.0,
            kind: WaveletKind::Morlet,
            omega0_low: 6.0,
            omega0_high: 6.0,
            ..CwtParams::default()
        };
        let width = 16usize;
        let ns = params.num_scales;

        let rows_above_half = |scalo: &[f32]| {
            let prof: Vec<f32> = (0..ns)
                .map(|s| (0..width).map(|c| scalo[s * width + c]).sum())
                .collect();
            let max = prof.iter().cloned().fold(f32::MIN, f32::max);
            prof.iter().filter(|&&v| v > max * 0.5).count()
        };

        let (base, _) = engine
            .compute_all(
                std::slice::from_ref(&signal), sr, 0, n, width, &params, false, false,
            )
            .expect("base compute failed");
        let (sl, _) = engine
            .compute_all_superlet(
                std::slice::from_ref(&signal), sr, 0, n, width, &params, false, false, 3,
            )
            .expect("superlet compute failed");

        let best = peak_row(&sl[0].0, ns, width);
        let f_peak = row_freq(&params, ns, best);
        let rel = (f_peak - f0).abs() / f0;
        assert!(rel < 0.12, "peak row freq {f_peak:.1} Hz vs {f0} Hz (rel {rel:.3})");

        let w_base = rows_above_half(&base[0].0);
        let w_sl   = rows_above_half(&sl[0].0);
        assert!(w_sl < w_base, "superlet width {w_sl} rows vs base {w_base}");
    }

    /// The instantaneous-frequency deviation must report exactly the offset
    /// between the tone and the peak row's centre frequency, for both the
    /// amplitude-weighted and unweighted estimators.
    #[test]
    fn instfreq_matches_tone_offset() {
        let mut engine = engine_or_skip!();
        let sr = 44_100u32;
        let f0 = 1_234.0f64;
        let n = 32_768usize;
        let signal = tone(f0 / sr as f64, n);

        let params = CwtParams {
            num_scales: 64,
            f_min: 500.0,
            f_max: 2_000.0,
            kind: WaveletKind::Morlet,
            ..CwtParams::default()
        };
        let width = 8usize;

        for unweighted in [false, true] {
            let (outs, _) = engine
                .compute_all(
                    std::slice::from_ref(&signal),
                    sr, 0, n, width, &params, true, unweighted,
                )
                .expect("compute failed");
            let (scalo, _ph, _co, dev) = &outs[0];
            let ns = params.num_scales;
            let best = peak_row(scalo, ns, width);
            let f_row = row_freq(&params, ns, best);
            let expected = f0 / f_row - 1.0;
            let measured = dev[best * width + width / 2] as f64;
            assert!(
                (measured - expected).abs() < 0.005,
                "unweighted={unweighted}: inst dev {measured:.4} vs expected {expected:.4} \
                 (row {best}, f_row {f_row:.1} Hz)"
            );
        }
    }

    /// Identical signals on both channels ⇒ cross phase ≈ 0 with high
    /// coherence on the tone's row.
    #[test]
    fn cross_phase_of_identical_channels_is_zero() {
        let mut engine = engine_or_skip!();
        let sr = 44_100u32;
        let f0 = 2_000.0f64;
        let n = 16_384usize;
        let signal = tone(f0 / sr as f64, n);

        let params = CwtParams {
            num_scales: 64,
            f_min: 500.0,
            f_max: 8_000.0,
            kind: WaveletKind::Morlet,
            ..CwtParams::default()
        };
        let width = 16usize;
        let (outs, cross) = engine
            .compute_all(&[signal.clone(), signal], sr, 0, n, width, &params, false, false)
            .expect("compute failed");
        let cross = cross.expect("stereo input must yield a cross spectrum");
        let ns = params.num_scales;
        let best = peak_row(&outs[0].0, ns, width);

        let idx = best * width + width / 2;
        assert!(cross.phase[idx].abs() < 1e-3, "cross phase {} ≠ 0", cross.phase[idx]);
        assert!(cross.coherence[idx] > 0.99, "cross coherence {}", cross.coherence[idx]);
    }
}
