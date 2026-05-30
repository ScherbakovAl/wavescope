use std::f64::consts::PI;

use crate::gpu::{ExtractParams, GpuContext, MulParams, WavParams};
use crate::wavelet::CwtParams;

// ---------------------------------------------------------------------------
// Zero-phase IIR anti-aliasing low-pass filter
// ---------------------------------------------------------------------------

/// Applies `order` forward + `order` backward passes of a single-pole IIR
/// low-pass filter, giving zero phase distortion and ~6*2*order dB/oct rolloff.
/// `cutoff_hz` is the -3 dB point; `sample_rate` is the signal's sample rate.
fn lowpass_zero_phase(signal: &[f32], cutoff_hz: f64, sample_rate: f64, order: usize) -> Vec<f32> {
    let cn    = cutoff_hz / sample_rate;          // normalized cutoff ∈ (0, 0.5)
    let alpha = (-2.0 * PI * cn).exp() as f32;   // pole location
    let k     = 1.0 - alpha;                      // DC gain correction

    let mut buf = signal.to_vec();

    for _ in 0..order {
        // Forward pass
        let mut s = buf[0];
        for x in buf.iter_mut() {
            s   = alpha * s + k * *x;
            *x  = s;
        }
        // Backward pass (cancels phase shift)
        let mut s = *buf.last().unwrap();
        for x in buf.iter_mut().rev() {
            s   = alpha * s + k * *x;
            *x  = s;
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// CWT engine
// ---------------------------------------------------------------------------

/// CWT outputs for one channel, each laid out `[scale * width + col]`:
/// (amplitude, phase, coherence, instantaneous-frequency deviation).
pub type CwtOutput = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub struct CwtEngine {
    gpu: GpuContext,
}

impl CwtEngine {
    pub fn new(gpu: GpuContext) -> Self {
        CwtEngine { gpu }
    }

    /// Compute the CWT scalogram for one audio channel.
    ///
    /// Returns `Vec<f32>` of length `params.num_scales * width_pixels`.
    /// Index: `scalogram[scale * width_pixels + col]`
    /// `scale = 0` → `f_min` (lowest freq), `scale = num_scales-1` → `f_max`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &self,
        signal:      &[f32],
        sample_rate: u32,
        t_start:     usize,
        t_end:       usize,
        width_pixels: usize,
        params:      &CwtParams,
        antialias:   bool,
    ) -> anyhow::Result<CwtOutput> {
        let total     = signal.len();
        let t_start   = t_start.min(total);
        let t_end     = t_end.min(total);
        let num_scales = params.num_scales;

        if t_start >= t_end || width_pixels == 0 || num_scales == 0 {
            return Ok((vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels]));
        }

        let visible_samples = t_end - t_start;

        // ---- downsampling factor -----------------------------------------
        // One pixel should represent at most this many original samples,
        // but we must preserve enough bandwidth to resolve f_max (Nyquist).
        let max_d_for_fmax = ((sample_rate as f64) / (2.0 * params.f_max as f64))
            .floor()
            .max(1.0) as usize;
        let d = (visible_samples / width_pixels).max(1).min(max_d_for_fmax);
        let f_ds = sample_rate as f64 / d as f64;   // effective sample rate

        // ---- padding (wavelet support at the lowest frequency) ------------
        // Largest scale occurs at f_min, using the wavelet's peak η there:
        //   s_max = η·fs / (2π·f_min).
        let f_min     = (params.f_min as f64).max(0.1);
        let f_max_eff = (params.f_max as f64).min(f_ds / 2.0 * 0.99).max(f_min * 1.01);
        let eta_low    = params.peak_eta(0.0);
        let s_max_orig = eta_low * sample_rate as f64 / (2.0 * PI * f_min);
        // Broad families (Morse/Paul/Bump) ring longer in time than the Morlet
        // baseline; cover their support expressed as cycles at f_min as well.
        let pad_cycles = params.support_cycles() * sample_rate as f64 / f_min;
        let padding = ((5.0 * s_max_orig).max(pad_cycles) as usize)
            .min(total / 2)
            .max(16);

        // ---- extract + downsample segment ---------------------------------
        let seg_start = t_start.saturating_sub(padding);
        let seg_end   = (t_end + padding).min(total);
        let pre_pad   = t_start - seg_start;   // samples before t_start in segment

        // Anti-aliasing: low-pass filter at 90 % of the downsampled Nyquist
        // before decimation to prevent frequencies above f_ds/2 from folding back.
        let segment_ds: Vec<f32> = if antialias && d > 1 {
            let cutoff = sample_rate as f64 / (2.0 * d as f64) * 0.9;
            let filtered = lowpass_zero_phase(
                &signal[seg_start..seg_end], cutoff, sample_rate as f64, 4,
            );
            filtered.chunks(d)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        } else {
            signal[seg_start..seg_end]
                .chunks(d)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        };

        let seg_ds_len   = segment_ds.len();
        let valid_start  = (pre_pad / d).min(seg_ds_len);
        let valid_end    = (valid_start + visible_samples / d + 1).min(seg_ds_len);

        if seg_ds_len < 4 || valid_start >= valid_end {
            return Ok((vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels],
                       vec![0.0f32; num_scales * width_pixels]));
        }

        // Next power of two ≥ segment length, capped at 1 M to stay in VRAM
        let n_fft = seg_ds_len.next_power_of_two().clamp(64, 1 << 20);

        // ---- scales + per-scale η (log-spaced from f_min to f_max_eff) ---
        // η is the wavelet's dimensionless peak frequency (= ω₀ for Morlet,
        // possibly log-interpolated low→high; constant for the other families).
        // Scale is kept consistent with η so the peak lands exactly on f_i:
        //   s_i = η_i · f_ds / (2π·f_i)
        let mut scales = vec![0.0f32; num_scales];
        let mut etas   = vec![0.0f32; num_scales];
        for i in 0..num_scales {
            let frac  = i as f64 / (num_scales - 1).max(1) as f64;
            let f_i   = f_min * (f_max_eff / f_min).powf(frac);
            let eta_i = params.peak_eta(frac);
            scales[i] = (eta_i * f_ds / (2.0 * PI * f_i)) as f32;
            etas[i]   = eta_i as f32;
        }

        // ---- GPU memory --------------------------------------------------
        // complex = 2×f32 = 8 bytes
        let bytes_f   = (n_fft * 4) as u64;
        let bytes_c   = (n_fft * 8) as u64;
        let bytes_sc  = (num_scales * 4) as u64;
        let bytes_all = (num_scales * n_fft * 8) as u64;
        let bytes_out = (num_scales * width_pixels * 4) as u64;

        let gpu = &self.gpu;
        let d_real    = gpu.storage(bytes_f);
        let d_complex = gpu.storage(bytes_c);
        let d_fft     = gpu.storage(bytes_c);
        let d_scratch = gpu.storage(bytes_c);   // forward-FFT ping-pong partner
        let d_scales  = gpu.storage(bytes_sc);
        let d_eta     = gpu.storage(bytes_sc);
        let d_wfreqs  = gpu.storage(bytes_all);
        let d_prod    = gpu.storage(bytes_all);
        let d_cwt     = gpu.storage(bytes_all);
        let d_scalo   = gpu.storage(bytes_out);
        let d_phase   = gpu.storage(bytes_out);
        let d_coher   = gpu.storage(bytes_out);
        let d_instdev = gpu.storage(bytes_out);

        // ---- upload -------------------------------------------------------
        let mut padded = vec![0.0f32; n_fft];
        let copy_len = seg_ds_len.min(n_fft);
        padded[..copy_len].copy_from_slice(&segment_ds[..copy_len]);
        gpu.upload_f32(&d_real, &padded);
        gpu.upload_f32(&d_scales, &scales);
        gpu.upload_f32(&d_eta, &etas);

        let n32        = n_fft as u32;
        let ns32       = num_scales as u32;
        let (p1, p2)   = params.kernel_params();
        let kind32     = params.kind.code() as u32;

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // ---- 1. real → complex -------------------------------------------
        gpu.real_to_complex(&mut enc, &d_real, &d_complex, n32);

        // ---- 2. forward FFT (d_complex → d_fft) --------------------------
        gpu.fft(&mut enc, &d_complex, &d_fft, &d_scratch, n32, 1, false);

        // ---- 3. wavelet in frequency domain for all scales ---------------
        gpu.wavelet(
            &mut enc,
            &d_wfreqs,
            &d_scales,
            &d_eta,
            WavParams { n: n32, num_scales: ns32, kind: kind32, p1, p2, _pad: [0; 3] },
        );

        // ---- 4. multiply signal_fft × wavelet_freqs ----------------------
        gpu.multiply(
            &mut enc,
            &d_fft,
            &d_wfreqs,
            &d_prod,
            MulParams { n: n32, num_scales: ns32, _pad: [0; 2] },
        );

        // ---- 5. batch IFFT (one per scale). d_wfreqs is free now and
        //         doubles as the ping-pong scratch. -----------------------
        gpu.fft(&mut enc, &d_prod, &d_cwt, &d_wfreqs, n32, ns32, true);

        // ---- 6. extract scalogram ----------------------------------------
        gpu.extract(
            &mut enc,
            &d_cwt,
            &d_scalo,
            &d_phase,
            &d_coher,
            &d_instdev,
            &d_scales,
            &d_eta,
            ExtractParams {
                n: n32,
                num_scales: ns32,
                valid_start: valid_start as i32,
                valid_end: valid_end as i32,
                width: width_pixels as u32,
                _pad: [0; 3],
            },
        );

        // ---- copy outputs to mappable staging buffers --------------------
        let s_scalo   = gpu.staging(bytes_out);
        let s_phase   = gpu.staging(bytes_out);
        let s_coher   = gpu.staging(bytes_out);
        let s_instdev = gpu.staging(bytes_out);
        enc.copy_buffer_to_buffer(&d_scalo,   0, &s_scalo,   0, bytes_out);
        enc.copy_buffer_to_buffer(&d_phase,   0, &s_phase,   0, bytes_out);
        enc.copy_buffer_to_buffer(&d_coher,   0, &s_coher,   0, bytes_out);
        enc.copy_buffer_to_buffer(&d_instdev, 0, &s_instdev, 0, bytes_out);

        gpu.queue.submit(Some(enc.finish()));

        // ---- download ----------------------------------------------------
        let n_out = num_scales * width_pixels;
        let scalogram = gpu.download_f32(&s_scalo,   n_out);
        let phase     = gpu.download_f32(&s_phase,   n_out);
        let coherence = gpu.download_f32(&s_coher,   n_out);
        let inst_dev  = gpu.download_f32(&s_instdev, n_out);

        Ok((scalogram, phase, coherence, inst_dev))
    }
}

// ---------------------------------------------------------------------------
// End-to-end test: a pure tone must light up the scalogram row whose centre
// frequency matches the tone. Exercises the whole GPU pipeline (real→complex,
// FFT, wavelet, multiply, batch IFFT, extract). Skips if no GPU adapter.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet::{CwtParams, WaveletKind};

    #[test]
    fn scalogram_peaks_at_tone_frequency() {
        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test (no adapter): {e}");
                return;
            }
        };
        let engine = CwtEngine::new(gpu);

        let sr = 44_100u32;
        let f0 = 2_000.0f64;
        let n = 16_384usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / sr as f64).sin() as f32)
            .collect();

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
        let (scalo, _p, _c, _d) = engine
            .compute(&signal, sr, 0, n, width, &params, false)
            .expect("compute failed");

        // Row with the most energy summed across columns.
        let ns = params.num_scales;
        let (mut best, mut best_v) = (0usize, f32::MIN);
        for s in 0..ns {
            let sum: f32 = (0..width).map(|c| scalo[s * width + c]).sum();
            if sum > best_v {
                best_v = sum;
                best = s;
            }
        }
        let frac = best as f64 / (ns - 1) as f64;
        let f_peak = params.f_min as f64 * (params.f_max as f64 / params.f_min as f64).powf(frac);
        let rel = (f_peak - f0).abs() / f0;
        assert!(rel < 0.12, "peak row freq {f_peak:.1} Hz vs {f0} Hz (rel {rel:.3})");
    }
}
