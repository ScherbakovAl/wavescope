// Post-processing of a tracked ridge: oscillator-stability and synchronization
// metrics borrowed from frequency metrology, timing astronomy, and the physics
// of coupled oscillators. All functions are pure; the caller builds the input
// series from the ridge (fractional frequency y, L−R phase difference Δφ).
//
//   * Allan deviation  σ_y(τ)  — frequency-standard metrology (atomic clocks).
//   * O−C residual            — Observed minus Calculated timing (pulsars,
//                               eclipsing binaries, pulsating stars).
//   * Kuramoto order param r  — synchronization of coupled oscillators.

use std::f64::consts::TAU;

/// Overlapping Allan deviation σ_y(τ) from a uniformly-sampled fractional
/// frequency series `y` (sampling interval `tau0`, seconds).
///
/// Computed via the phase (time-error) samples x_k = Σ_{i<k} y_i·τ₀ and the
/// standard second-difference estimator (NIST SP 1065, eq. for overlapping
/// ADEV):
///
///     σ_y²(mτ₀) = 1 / (2 (mτ₀)² (N − 2m)) · Σ (x_{i+2m} − 2x_{i+m} + x_i)²
///
/// Returned as `[τ, σ_y(τ)]` pairs at octave-spaced averaging factors
/// m = 1, 2, 4, …, the usual log-log presentation. Slopes identify the noise
/// type (−1 white/flicker PM, −½ white FM, 0 flicker FM, +½ random-walk FM).
pub fn allan_deviation(y: &[f64], tau0: f64) -> Vec<[f64; 2]> {
    let n = y.len();
    if n < 2 || tau0 <= 0.0 {
        return Vec::new();
    }
    // Phase (time-error) samples: x has n+1 entries, x[0] = 0.
    let mut x = vec![0.0f64; n + 1];
    for i in 0..n {
        x[i + 1] = x[i] + y[i] * tau0;
    }
    let nx = x.len();

    let mut out = Vec::new();
    let mut m = 1usize;
    // Need at least one term: i runs 0..=(nx-1-2m), so require nx - 2m >= 2.
    while nx >= 2 * m + 2 {
        let terms = nx - 2 * m; // number of i values (0..nx-2m)
        let mut acc = 0.0;
        for i in 0..terms {
            let d = x[i + 2 * m] - 2.0 * x[i + m] + x[i];
            acc += d * d;
        }
        let tau = m as f64 * tau0;
        let sigma2 = acc / (2.0 * tau * tau * terms as f64);
        out.push([tau, sigma2.sqrt()]);
        m *= 2;
    }
    out
}

/// O−C (Observed minus Calculated) timing residual along the ridge, expressed
/// in carrier cycles. The "calculated" model is a constant-frequency carrier
/// f₀; with `y_i = (f_inst_i − f₀)/f₀` the accumulated phase error in cycles is
///
///     (O−C)_k = f₀ · ∫₀^{t_k} y dt = f₀·τ₀ · Σ_{i<k} y_i
///
/// One entry per ridge column (same length as `y`), starting at 0. Curvature
/// reveals period changes; a constant `y` (pure offset) integrates to a
/// straight line. When `y` is built relative to the window-mean f₀ its mean is
/// ~0, so the linear trend is already removed — only the wobble remains.
pub fn oc_cycles(y: &[f64], f0: f64, tau0: f64) -> Vec<f64> {
    let mut oc = Vec::with_capacity(y.len());
    let mut acc = 0.0;
    for &yi in y {
        oc.push(acc * tau0 * f0);
        acc += yi;
    }
    oc
}

/// Instantaneous Kuramoto order parameter r(t) for the two-oscillator ensemble
/// (left/right channel) from the per-column phase difference Δφ = φ_L − φ_R.
///
/// For N = 2, r = |½(e^{iφ_L} + e^{iφ_R})| = |cos(Δφ/2)|: r = 1 in phase, 0 in
/// antiphase. One entry per column.
pub fn kuramoto_r(dphi: &[f64]) -> Vec<f64> {
    dphi.iter().map(|d| (d * 0.5).cos().abs()).collect()
}

/// Phase-locking value: |⟨e^{iΔφ}⟩| over the window ∈ [0, 1]. A scalar "lock
/// quality" — 1 means a perfectly steady phase relationship (any fixed offset),
/// 0 means the relative phase is uniformly smeared.
pub fn mean_plv(dphi: &[f64]) -> f64 {
    if dphi.is_empty() {
        return 0.0;
    }
    let (mut sre, mut sim) = (0.0f64, 0.0f64);
    for &d in dphi {
        sre += d.cos();
        sim += d.sin();
    }
    let n = dphi.len() as f64;
    ((sre / n).powi(2) + (sim / n).powi(2)).sqrt()
}

/// Wrap an angle to (−π, π].
pub fn wrap_pi(a: f64) -> f64 {
    let mut x = a % TAU;
    if x > std::f64::consts::PI {
        x -= TAU;
    } else if x <= -std::f64::consts::PI {
        x += TAU;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allan_constant_frequency_is_zero() {
        // A pure frequency offset → linear phase → zero second difference.
        let y = vec![0.003; 64];
        let adev = allan_deviation(&y, 1.0);
        assert!(!adev.is_empty());
        for [_, s] in adev {
            assert!(s.abs() < 1e-12, "sigma {s} should vanish");
        }
    }

    #[test]
    fn allan_octave_taus() {
        let y: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let adev = allan_deviation(&y, 0.5);
        let taus: Vec<f64> = adev.iter().map(|p| p[0]).collect();
        // n=32 ⇒ nx=33; m=1,2,4,8 pass (m=16 needs nx≥34).
        assert_eq!(taus, vec![0.5, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn oc_constant_drift_is_linear() {
        let y = vec![0.01; 5];
        let oc = oc_cycles(&y, 100.0, 0.1); // f0=100, tau0=0.1 → step = 0.01*0.1*100 = 0.1
        assert_eq!(oc.len(), 5);
        for (k, v) in oc.iter().enumerate() {
            assert!((v - k as f64 * 0.1).abs() < 1e-12, "oc[{k}] = {v}");
        }
    }

    #[test]
    fn kuramoto_in_and_antiphase() {
        assert!((kuramoto_r(&[0.0])[0] - 1.0).abs() < 1e-12);
        assert!(kuramoto_r(&[std::f64::consts::PI])[0].abs() < 1e-12);
    }

    #[test]
    fn plv_locked_vs_smeared() {
        let locked = vec![0.7; 100];
        assert!((mean_plv(&locked) - 1.0).abs() < 1e-12);
        let smeared: Vec<f64> = (0..360).map(|d| (d as f64).to_radians()).collect();
        assert!(mean_plv(&smeared) < 1e-2);
    }

    #[test]
    fn wrap_pi_range() {
        assert!((wrap_pi(TAU + 0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_pi(-TAU - 0.5) + 0.5).abs() < 1e-12);
        assert!((wrap_pi(std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-12);
    }
}
