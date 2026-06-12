use std::path::Path;
use std::sync::Arc;
use anyhow::Context;

/// Loaded audio file with samples normalised to [-1, 1].
pub struct AudioFile {
    pub sample_rate: u32,
    /// One Vec<f32> per channel, interleaved samples split into channels.
    /// Shared via `Arc` so compute requests don't copy the audio data.
    pub channels: Arc<Vec<Vec<f32>>>,
    pub duration_secs: f32,
    pub path: String,
}

impl AudioFile {
    /// Load a WAV or FLAC file (auto-detected by extension).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "wav" | "wave" => Self::load_wav(path),
            "flac"         => Self::load_flac(path),
            other          => anyhow::bail!("Unsupported audio format: .{}", other),
        }
    }

    fn load_wav(path: &Path) -> anyhow::Result<Self> {
        let mut reader =
            hound::WavReader::open(path).context("Cannot open WAV file")?;
        let spec = reader.spec();

        let sample_rate  = spec.sample_rate;
        let num_channels = spec.channels as usize;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .context("Reading WAV float samples")?,
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .collect::<Result<Vec<_>, _>>()
                    .context("Reading WAV int samples")?
                    .into_iter()
                    .map(|s| s as f32 / max)
                    .collect()
            }
        };

        Self::from_interleaved(samples, num_channels, sample_rate, path)
    }

    fn load_flac(path: &Path) -> anyhow::Result<Self> {
        let mut reader =
            claxon::FlacReader::open(path).context("Cannot open FLAC file")?;
        let info = reader.streaminfo();

        let sample_rate  = info.sample_rate;
        let num_channels = info.channels as usize;
        let max          = (1i64 << (info.bits_per_sample - 1)) as f32;

        // `samples()` yields interleaved i32 values
        let samples: Vec<f32> = reader
            .samples()
            .collect::<Result<Vec<i32>, _>>()
            .context("Reading FLAC samples")?
            .into_iter()
            .map(|s| s as f32 / max)
            .collect();

        Self::from_interleaved(samples, num_channels, sample_rate, path)
    }

    fn from_interleaved(
        samples: Vec<f32>,
        num_channels: usize,
        sample_rate: u32,
        path: &Path,
    ) -> anyhow::Result<Self> {
        if num_channels == 0 {
            anyhow::bail!("Audio file has zero channels");
        }
        let num_frames = samples.len() / num_channels;
        let mut channels = vec![Vec::with_capacity(num_frames); num_channels];
        for (i, s) in samples.iter().enumerate() {
            channels[i % num_channels].push(*s);
        }
        let duration_secs = num_frames as f32 / sample_rate as f32;
        Ok(AudioFile {
            sample_rate,
            channels: Arc::new(channels),
            duration_secs,
            path: path.to_string_lossy().into_owned(),
        })
    }

    pub fn num_samples(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }
}
