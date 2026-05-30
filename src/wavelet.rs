/// Trait for wavelet families (extensibility point for future wavelets).
pub trait WaveletKernel: Send + Sync {
    fn name(&self) -> &str;
    fn omega0(&self) -> f32;
}

pub struct MorletWavelet {
    pub omega0: f32,
}

impl WaveletKernel for MorletWavelet {
    fn name(&self) -> &str { "Morlet" }
    fn omega0(&self) -> f32 { self.omega0 }
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
    /// Morlet central frequency ω₀ at `f_min` (low-frequency end).
    pub omega0_low: f32,
    /// Morlet central frequency ω₀ at `f_max` (high-frequency end).
    /// Equal to `omega0_low` ⇒ classic constant-ω₀ (constant-Q) behaviour;
    /// `omega0_high > omega0_low` ⇒ sharper frequency lines on the highs,
    /// better time localisation on the lows. Log-interpolated in between.
    pub omega0_high: f32,
}

impl Default for CwtParams {
    fn default() -> Self {
        Self {
            num_scales: 128,
            f_min: 20.0,
            f_max: 20_000.0,
            omega0_low: 6.0,
            omega0_high: 6.0,
        }
    }
}
