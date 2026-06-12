/// Analysing wavelet family. All are analytic (ψ̂(ω)=0 for ω<0), so the phase,
/// phase-locking and instantaneous-frequency outputs stay meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveletKind {
    /// Complex Morlet — Gaussian in frequency, classic balanced time/freq.
    Morlet,
    /// Generalized Morse ψ̂(w)=wᵝ·e^(−wᵞ): two knobs (β, γ) tune symmetry and
    /// the time/frequency trade-off independently. Generalises Morlet/Paul.
    Morse,
    /// Bump ψ̂(w)=e^(1−1/(1−((w−μ)/σ)²)): compact in frequency ⇒ very sharp
    /// frequency lines, poorer time localisation. Good for close stationary tones.
    Bump,
    /// Paul ψ̂(w)=wᵐ·e^(−w): excellent time localisation ⇒ good for transients
    /// and onsets, poorer frequency resolution.
    Paul,
}

/// Bump centre μ (dimensionless). Fixed; bandwidth is tuned through σ.
pub const BUMP_MU: f64 = 5.0;

impl WaveletKind {
    pub fn name(self) -> &'static str {
        match self {
            WaveletKind::Morlet => "Morlet",
            WaveletKind::Morse  => "Generalized Morse",
            WaveletKind::Bump   => "Bump",
            WaveletKind::Paul   => "Paul",
        }
    }

    /// Integer code passed to the wavelet-generation compute kernel.
    pub fn code(self) -> i32 {
        match self {
            WaveletKind::Morlet => 0,
            WaveletKind::Morse  => 1,
            WaveletKind::Bump   => 2,
            WaveletKind::Paul   => 3,
        }
    }
}

/// Parameters for the CWT computation.
#[derive(Clone, Debug, PartialEq)]
pub struct CwtParams {
    /// Number of frequency bins (scales).
    pub num_scales: usize,
    /// Lowest visible frequency in Hz.
    pub f_min: f32,
    /// Highest visible frequency in Hz.
    pub f_max: f32,
    /// Which wavelet family to analyse with.
    pub kind: WaveletKind,
    /// Morlet central frequency ω₀ at `f_min` (low-frequency end).
    pub omega0_low: f32,
    /// Morlet central frequency ω₀ at `f_max` (high-frequency end).
    /// Equal to `omega0_low` ⇒ classic constant-ω₀ (constant-Q) behaviour;
    /// `omega0_high > omega0_low` ⇒ sharper frequency lines on the highs,
    /// better time localisation on the lows. Log-interpolated in between.
    pub omega0_high: f32,
    /// Generalized Morse β (>0): raises the time-bandwidth (number of
    /// oscillations); larger ⇒ sharper in frequency.
    pub morse_beta: f32,
    /// Generalized Morse γ (>0): symmetry of ψ̂; γ=3 is the symmetric default.
    pub morse_gamma: f32,
    /// Bump σ (0<σ<μ): half-bandwidth around μ; smaller ⇒ sharper frequency.
    pub bump_sigma: f32,
    /// Paul order m (≥1): larger ⇒ more oscillations, sharper in frequency.
    pub paul_order: f32,
}

impl Default for CwtParams {
    fn default() -> Self {
        Self {
            num_scales: 128,
            f_min: 20.0,
            f_max: 20_000.0,
            kind: WaveletKind::Morlet,
            omega0_low: 6.0,
            omega0_high: 6.0,
            morse_beta: 20.0,
            morse_gamma: 3.0,
            bump_sigma: 0.6,
            paul_order: 4.0,
        }
    }
}

impl CwtParams {
    /// Dimensionless peak frequency η of the wavelet at fractional position
    /// `frac` ∈ [0,1] along the log-frequency axis (0 ⇒ f_min, 1 ⇒ f_max).
    /// The scale is set so the peak lands on the row's frequency:
    ///   s = η · f_ds / (2π · f).
    /// Only Morlet varies η across the band (adaptive ω₀); the others are
    /// constant-Q and use a single peak value.
    pub fn peak_eta(&self, frac: f64) -> f64 {
        match self.kind {
            WaveletKind::Morlet => {
                let lo = self.omega0_low as f64;
                let hi = self.omega0_high as f64;
                lo * (hi / lo).powf(frac)
            }
            WaveletKind::Morse => {
                let b = self.morse_beta as f64;
                let g = self.morse_gamma as f64;
                (b / g).powf(1.0 / g)
            }
            WaveletKind::Bump => BUMP_MU,
            WaveletKind::Paul => self.paul_order as f64,
        }
    }

    /// Shape parameters (p1, p2) passed to the generation kernel.
    /// Morlet: unused. Morse: (β, γ). Bump: (σ, _). Paul: (m, _).
    pub fn kernel_params(&self) -> (f32, f32) {
        match self.kind {
            WaveletKind::Morlet => (0.0, 0.0),
            WaveletKind::Morse  => (self.morse_beta, self.morse_gamma),
            WaveletKind::Bump   => (self.bump_sigma, 0.0),
            WaveletKind::Paul   => (self.paul_order, 0.0),
        }
    }

    /// Parameters of the k-th superlet pass (k ≥ 1): the family's
    /// frequency-sharpness knob is scaled by k, mirroring multiplicative
    /// superlets (Moca et al. 2021) where the k-th wavelet carries k·c₁
    /// cycles. The generation kernel evaluates Morse/Paul in log space and
    /// peaks at 1, so large scaled parameters stay numerically safe.
    pub fn superlet_pass(&self, k: u32) -> CwtParams {
        let k = k as f32;
        let mut p = self.clone();
        match self.kind {
            WaveletKind::Morlet => {
                p.omega0_low  *= k;
                p.omega0_high *= k;
            }
            WaveletKind::Morse => p.morse_beta *= k,
            WaveletKind::Bump  => p.bump_sigma /= k,
            WaveletKind::Paul  => p.paul_order *= k,
        }
        p
    }

    /// One-sided time support of the family in oscillation cycles, used to size
    /// the edge padding. Morlet returns 0 (its support is set by the existing
    /// scale-based rule); the broader families need extra room at f_min.
    pub fn support_cycles(&self) -> f64 {
        match self.kind {
            WaveletKind::Morlet => 0.0,
            WaveletKind::Morse  =>
                1.5 * (self.morse_beta as f64 * self.morse_gamma as f64).sqrt(),
            WaveletKind::Bump   => 1.5 * BUMP_MU / self.bump_sigma as f64,
            WaveletKind::Paul   => 1.5 * self.paul_order as f64,
        }
    }
}
