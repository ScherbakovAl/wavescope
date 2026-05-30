#include <cuComplex.h>
#include <math.h>

// Convert real float array to complex (zero imaginary part)
extern "C" __global__ void real_to_complex_kernel(
    const float* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = make_cuFloatComplex(in[i], 0.0f);
}

// Compute Morlet wavelet in frequency domain for all scales simultaneously.
// For scale s, the Morlet in freq domain (normalized angular freq omega_k):
//   psi_hat_s(k) = pi^(-1/4) * sqrt(2*pi) * sqrt(s) * exp(-0.5*(s*omega_k - omega0)^2)
//                  for s*omega_k > 0, else 0
// omega_k = 2*pi*k/N  for k in [0, N/2]
//          = 2*pi*(k-N)/N for k in (N/2, N)
// Output: wavelet_freqs[scale_idx * N + k]
// `omega0` is per-scale: omega0[s] is the central frequency for scale s,
// kept consistent with scales[s] so the peak still lands at the target freq.
extern "C" __global__ void morlet_freq_all_scales_kernel(
    cuFloatComplex* __restrict__ wavelet_freqs,
    const float* __restrict__   scales,
    int N, int num_scales,
    const float* __restrict__   omega0)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= N || s >= num_scales) return;

    float scale  = scales[s];
    float omega0s = omega0[s];

    float omega;
    if (k <= N / 2)
        omega = 2.0f * (float)M_PI * (float)k / (float)N;
    else
        omega = 2.0f * (float)M_PI * (float)(k - N) / (float)N;

    float s_omega = scale * omega;
    float val = 0.0f;
    if (s_omega > 0.0f) {
        float diff = s_omega - omega0s;
        val = powf((float)M_PI, -0.25f)
            * sqrtf(2.0f * (float)M_PI)
            * sqrtf(scale)
            * expf(-0.5f * diff * diff);
    }

    wavelet_freqs[(long long)s * N + k] = make_cuFloatComplex(val, 0.0f);
}

// Element-wise multiply signal_fft (broadcast over scales) with wavelet_freqs.
// products[s*N + k] = signal_fft[k] * wavelet_freqs[s*N + k]
extern "C" __global__ void multiply_signal_wavelets_kernel(
    const cuFloatComplex* __restrict__ signal_fft,
    const cuFloatComplex* __restrict__ wavelet_freqs,
    cuFloatComplex* __restrict__       products,
    int N, int num_scales)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= N || s >= num_scales) return;

    long long idx = (long long)s * N + k;
    products[idx] = cuCmulf(signal_fft[k], wavelet_freqs[idx]);
}

// Extract scalogram from complex CWT rows (output of batch IFFT).
// Maps valid range [valid_start, valid_end) → width_pixels output columns.
// Row 0 in scalogram = scale 0 (lowest freq), row num_scales-1 = highest freq.
// scalogram[s * width_pixels + col] = mean |cwt_rows[s*N + t]| / N
//   for t in [samp_start, samp_end) within [valid_start, valid_end)
// `coherence[s*width+col]` = |Σz| / Σ|z| ∈ [0,1] over the same samples: the
// phase-locking value (vector strength). ≈1 when the phase is constant within
// the pixel (resolvable), →0 when it rotates / averages out (aliased), so it
// can drive saturation and suppress false phase colour where it is meaningless.
//
// `inst_dev[s*width+col]` = relative instantaneous-frequency deviation from the
// row's nominal frequency, (f_inst − f_i)/f_i. The mean per-sample phase advance
// is dφ = arg(Σ W[t+1]·conj(W[t])) (wrap-free, amplitude-weighted). Since the
// scale satisfies s_i = ω₀_i·f_ds/(2π·f_i), we have 2π·f_i/f_ds = ω₀_i/s_i, so
//   f_inst/f_i = dφ·s_i/ω₀_i  ⇒  inst_dev = dφ·scales[s]/omega0[s] − 1,
// needing neither f_ds nor f_i. 0 ⇒ exactly on the row's centre frequency,
// >0 above it, <0 below; its time-wobble exposes phase pulling / slips.
extern "C" __global__ void extract_scalogram_kernel(
    const cuFloatComplex* __restrict__ cwt_rows,
    float* __restrict__                scalogram,
    float* __restrict__                phase,
    float* __restrict__                coherence,
    float* __restrict__                inst_dev,
    const float* __restrict__          scales,
    const float* __restrict__          omega0,
    int N, int num_scales,
    int valid_start, int valid_end,
    int width_pixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int s   = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width_pixels || s >= num_scales) return;

    int valid_len  = valid_end - valid_start;
    int samp_start = valid_start + (int)((float)col       * valid_len / width_pixels);
    int samp_end   = valid_start + (int)((float)(col + 1) * valid_len / width_pixels);

    if (samp_end   > valid_end) samp_end   = valid_end;
    if (samp_end   > N)         samp_end   = N;
    if (samp_start >= N) {
        scalogram[s * width_pixels + col] = 0.0f;
        phase[s * width_pixels + col]     = 0.0f;
        coherence[s * width_pixels + col] = 0.0f;
        inst_dev[s * width_pixels + col]  = 0.0f;
        return;
    }
    if (samp_end <= samp_start) samp_end = samp_start + 1;

    float sum   = 0.0f;
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    int   count = samp_end - samp_start;
    long long row_off = (long long)s * N;
    // Lag-1 autocorrelation Σ W[t+1]·conj(W[t]) for the instantaneous frequency.
    cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
    for (int t = samp_start; t < samp_end; t++) {
        cuFloatComplex c = cwt_rows[row_off + t];
        sum    += cuCabsf(c);
        sum_re += cuCrealf(c);
        sum_im += cuCimagf(c);
        if (t + 1 < N) {
            cuFloatComplex cn = cwt_rows[row_off + t + 1];
            acc = cuCaddf(acc, cuCmulf(cn, cuConjf(c)));
        }
    }

    // Divide by N for IFFT normalisation, then by count to average
    scalogram[s * width_pixels + col] = sum / ((float)N * (float)count);
    // Phase of the (complex) mean. The N*count scaling is positive and does
    // not affect atan2, so we take the phase of the raw complex sum.
    phase[s * width_pixels + col] = atan2f(sum_im, sum_re);
    // Phase-locking value: |Σz| / Σ|z|. 1 ⇒ phase constant within the pixel,
    // 0 ⇒ rotating/averaged-out (aliased) ⇒ no meaningful phase to colour.
    float mag = sqrtf(sum_re * sum_re + sum_im * sum_im);
    coherence[s * width_pixels + col] = (sum > 0.0f) ? (mag / sum) : 0.0f;
    // Relative instantaneous-frequency deviation (see header note).
    float dphi  = atan2f(cuCimagf(acc), cuCrealf(acc));
    float omega = omega0[s];
    inst_dev[s * width_pixels + col] =
        (omega > 0.0f) ? (dphi * scales[s] / omega - 1.0f) : 0.0f;
}
