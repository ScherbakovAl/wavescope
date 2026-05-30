use std::f64::consts::PI;
use std::sync::Arc;
use std::ffi::c_void;

use crate::cuda::{
    CudaContext, CudaModule, CudaFunction, CudaBuffer,
    CufftPlan, CUFFT_FORWARD, CUFFT_INVERSE,
};
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

pub struct CwtEngine {
    context:              Arc<CudaContext>,
    fn_real_to_complex:   CudaFunction,
    fn_morlet_all_scales: CudaFunction,
    fn_multiply:          CudaFunction,
    fn_extract:           CudaFunction,
}

impl CwtEngine {
    pub fn new(
        context: Arc<CudaContext>,
        module:  &CudaModule,
    ) -> anyhow::Result<Self> {
        Ok(CwtEngine {
            context,
            fn_real_to_complex:   module.get_function("real_to_complex_kernel")?,
            fn_morlet_all_scales: module.get_function("morlet_freq_all_scales_kernel")?,
            fn_multiply:          module.get_function("multiply_signal_wavelets_kernel")?,
            fn_extract:           module.get_function("extract_scalogram_kernel")?,
        })
    }

    /// Compute the CWT scalogram for one audio channel.
    ///
    /// Returns `Vec<f32>` of length `params.num_scales * width_pixels`.
    /// Index: `scalogram[scale * width_pixels + col]`
    /// `scale = 0` → `f_min` (lowest freq), `scale = num_scales-1` → `f_max`.
    pub fn compute(
        &self,
        signal:      &[f32],
        sample_rate: u32,
        t_start:     usize,
        t_end:       usize,
        width_pixels: usize,
        params:      &CwtParams,
        antialias:   bool,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let total     = signal.len();
        let t_start   = t_start.min(total);
        let t_end     = t_end.min(total);
        let num_scales = params.num_scales;

        if t_start >= t_end || width_pixels == 0 || num_scales == 0 {
            return Ok((vec![0.0f32; num_scales * width_pixels],
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

        // ---- padding (half wavelet support at the lowest frequency) -------
        // Scale for f_min in original samples: s_max = ω₀·fs / (2π·f_min)
        let f_min     = (params.f_min as f64).max(0.1);
        let f_max_eff = (params.f_max as f64).min(f_ds / 2.0 * 0.99).max(f_min * 1.01);
        // Largest scale occurs at f_min, which uses ω₀ = omega0_low.
        let s_max_orig = params.omega0_low as f64 * sample_rate as f64
            / (2.0 * PI * f_min);
        let padding = ((5.0 * s_max_orig) as usize).min(total / 2).max(16);

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
                       vec![0.0f32; num_scales * width_pixels]));
        }

        // Next power of two ≥ segment length, capped at 1 M to stay in VRAM
        let n_fft = seg_ds_len.next_power_of_two().min(1 << 20).max(64);

        // ---- scales + per-scale ω₀ (log-spaced from f_min to f_max_eff) --
        // ω₀ is log-interpolated in frequency between omega0_low (at f_min)
        // and omega0_high (at f_max_eff). Scale is kept consistent with the
        // row's ω₀ so the wavelet peak still lands exactly on f_i:
        //   s_i = ω₀_i · f_ds / (2π·f_i)
        let omega0_low  = params.omega0_low  as f64;
        let omega0_high = params.omega0_high as f64;
        let mut scales  = vec![0.0f32; num_scales];
        let mut omega0s = vec![0.0f32; num_scales];
        for i in 0..num_scales {
            let frac    = i as f64 / (num_scales - 1).max(1) as f64;
            let f_i     = f_min * (f_max_eff / f_min).powf(frac);
            let omega0_i = omega0_low * (omega0_high / omega0_low).powf(frac);
            scales[i]  = (omega0_i * f_ds / (2.0 * PI * f_i)) as f32;
            omega0s[i] = omega0_i as f32;
        }

        // ---- GPU memory --------------------------------------------------
        // cuFloatComplex = 2×f32 = 8 bytes
        let bytes_f   = n_fft * 4;
        let bytes_c   = n_fft * 8;
        let bytes_sc  = num_scales * 4;
        let bytes_all = num_scales * n_fft * 8;
        let bytes_out = num_scales * width_pixels * 4;

        let d_real    = CudaBuffer::alloc(bytes_f)?;
        let d_complex = CudaBuffer::alloc(bytes_c)?;
        let d_fft     = CudaBuffer::alloc(bytes_c)?;
        let d_scales  = CudaBuffer::alloc(bytes_sc)?;
        let d_omega0  = CudaBuffer::alloc(bytes_sc)?;
        let d_wfreqs  = CudaBuffer::alloc(bytes_all)?;
        let d_prod    = CudaBuffer::alloc(bytes_all)?;
        let d_cwt     = CudaBuffer::alloc(bytes_all)?;
        let d_scalo   = CudaBuffer::alloc(bytes_out)?;
        let d_phase   = CudaBuffer::alloc(bytes_out)?;
        let d_coher   = CudaBuffer::alloc(bytes_out)?;

        // ---- upload -------------------------------------------------------
        let mut padded = vec![0.0f32; n_fft];
        let copy_len = seg_ds_len.min(n_fft);
        padded[..copy_len].copy_from_slice(&segment_ds[..copy_len]);
        d_real.upload_f32(&padded)?;
        d_scales.upload_f32(&scales)?;
        d_omega0.upload_f32(&omega0s)?;

        // ---- kernel helpers ----------------------------------------------
        let n32      = n_fft as i32;
        let ns32     = num_scales as i32;
        let wp32     = width_pixels as i32;
        let vs32     = valid_start as i32;
        let ve32     = valid_end   as i32;

        let bx: u32 = 256;
        let bxy_x: u32 = 32;
        let bxy_y: u32 = 8;
        let gn  = ((n_fft      as u32) + bx    - 1) / bx;
        let gkx = ((n_fft      as u32) + bxy_x - 1) / bxy_x;
        let gky = ((num_scales as u32) + bxy_y - 1) / bxy_y;
        let gex = ((width_pixels as u32) + bxy_x - 1) / bxy_x;

        // ---- 1. real → complex -------------------------------------------
        unsafe {
            let mut a0 = d_real.ptr();
            let mut a1 = d_complex.ptr();
            let mut a2 = n32;
            let mut p: [*mut c_void; 3] = [
                &mut a0 as *mut _ as *mut c_void,
                &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void,
            ];
            self.fn_real_to_complex.launch((gn, 1, 1), (bx, 1, 1), &mut p)?;
        }

        // ---- 2. forward FFT ----------------------------------------------
        let plan_fwd = CufftPlan::plan_single_c2c(n_fft)?;
        plan_fwd.exec_c2c(d_complex.ptr(), d_fft.ptr(), CUFFT_FORWARD)?;

        // ---- 3. Morlet in frequency domain for all scales ----------------
        unsafe {
            let mut a0 = d_wfreqs.ptr();
            let mut a1 = d_scales.ptr();
            let mut a2 = n32;
            let mut a3 = ns32;
            let mut a4 = d_omega0.ptr();
            let mut p: [*mut c_void; 5] = [
                &mut a0 as *mut _ as *mut c_void,
                &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void,
                &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void,
            ];
            self.fn_morlet_all_scales
                .launch((gkx, gky, 1), (bxy_x, bxy_y, 1), &mut p)?;
        }

        // ---- 4. multiply signal_fft × wavelet_freqs ---------------------
        unsafe {
            let mut a0 = d_fft.ptr();
            let mut a1 = d_wfreqs.ptr();
            let mut a2 = d_prod.ptr();
            let mut a3 = n32;
            let mut a4 = ns32;
            let mut p: [*mut c_void; 5] = [
                &mut a0 as *mut _ as *mut c_void,
                &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void,
                &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void,
            ];
            self.fn_multiply
                .launch((gkx, gky, 1), (bxy_x, bxy_y, 1), &mut p)?;
        }

        // ---- 5. batch IFFT (one per scale) -------------------------------
        let plan_inv = CufftPlan::plan_batch_c2c(n_fft, num_scales)?;
        plan_inv.exec_c2c(d_prod.ptr(), d_cwt.ptr(), CUFFT_INVERSE)?;

        // ---- 6. extract scalogram ----------------------------------------
        unsafe {
            let mut a0  = d_cwt.ptr();
            let mut a1  = d_scalo.ptr();
            let mut a1p = d_phase.ptr();
            let mut a1c = d_coher.ptr();
            let mut a2  = n32;
            let mut a3  = ns32;
            let mut a4  = vs32;
            let mut a5  = ve32;
            let mut a6  = wp32;
            let mut p: [*mut c_void; 9] = [
                &mut a0  as *mut _ as *mut c_void,
                &mut a1  as *mut _ as *mut c_void,
                &mut a1p as *mut _ as *mut c_void,
                &mut a1c as *mut _ as *mut c_void,
                &mut a2  as *mut _ as *mut c_void,
                &mut a3  as *mut _ as *mut c_void,
                &mut a4  as *mut _ as *mut c_void,
                &mut a5  as *mut _ as *mut c_void,
                &mut a6  as *mut _ as *mut c_void,
            ];
            self.fn_extract
                .launch((gex, gky, 1), (bxy_x, bxy_y, 1), &mut p)?;
        }

        self.context.synchronize()?;

        // ---- download ----------------------------------------------------
        let mut scalogram = vec![0.0f32; num_scales * width_pixels];
        d_scalo.download_f32(&mut scalogram)?;
        let mut phase = vec![0.0f32; num_scales * width_pixels];
        d_phase.download_f32(&mut phase)?;
        let mut coherence = vec![0.0f32; num_scales * width_pixels];
        d_coher.download_f32(&mut coherence)?;

        Ok((scalogram, phase, coherence))
    }
}
