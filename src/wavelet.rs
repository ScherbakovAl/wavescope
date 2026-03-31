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
    /// Morlet central frequency parameter ω₀ (typically 5–8).
    pub omega0: f32,
}

impl Default for CwtParams {
    fn default() -> Self {
        Self {
            num_scales: 128,
            f_min: 20.0,
            f_max: 20_000.0,
            omega0: 6.0,
        }
    }
}
