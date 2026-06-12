# Wavescope

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20573387.svg)](https://doi.org/10.5281/zenodo.20573387)

*A GPU wavelet scalogram viewer for audio.*

GPU-accelerated continuous-wavelet-transform (CWT) viewer for WAV/FLAC audio.
Interactive scalogram with zoom/pan and amplitude / phase / instantaneous-frequency
/ cross-channel-phase views, built for hunting subtle phase dynamics
(oscillator synchronisation, frequency pulling, phase slips).

Four analytic wavelet families to analyse with:

- **Morlet** — Gaussian in frequency, classic balanced time/frequency.
- **Generalized Morse** — two knobs (β, γ) tune symmetry and the time/frequency
  trade-off independently; generalises Morlet and Paul.
- **Bump** — compact in frequency ⇒ very sharp frequency lines, good for close
  stationary tones.
- **Paul** — excellent time localisation ⇒ good for transients and onsets.

## Phase analysis

- **Inst. frequency view** — per-pixel relative deviation (f_inst − f_row)/f_row
  on a diverging blue↔red map, with *nominal* or *detrended* baseline. The
  estimator is arg Σ W(t+1)·W̄(t); an optional **unweighted** mode averages
  unit-magnitude phase increments instead, so phase slips that coincide with
  amplitude dips are not underweighted.
- **Cross phase (L−R)** — for stereo files: hue = arg(W_L·W̄_R) (relative phase
  between channels), saturation = cross-coherence, brightness = geometric-mean
  amplitude. The canonical observable for synchronisation analysis.
- **Ridge tracking & CSV export** — double-click a ridge to track it across the
  view; export the trace (time, f_inst, amplitude, phase, coherence) as CSV.

Processing is measurement-friendly by construction: L1-normalised wavelets
(equal-amplitude tones are equally bright at any frequency and zoom),
zero-phase half-band FIR decimation (flat passband, no phase distortion), and
time-chunked compute so arbitrarily long files render in full within a bounded
VRAM budget.

## Controls

- **Scroll** — zoom time axis · **Ctrl/Shift+Scroll** — zoom frequency axis
- **Alt+Scroll** — pan time · **Drag** — pan
- **Double-click** — pick a ridge · **Right-click** — reset view

Cross-platform via [`wgpu`](https://github.com/gfx-rs/wgpu): compute runs on
Vulkan (Linux/Windows), Metal (macOS) or DX12 (Windows), so it works on NVIDIA,
AMD, Intel and Apple GPUs.

## Download

Prebuilt binaries for Linux, Windows and macOS are attached to each
[GitHub Release](../../releases):

- **Linux** — `…-linux-x86_64.tar.gz` (extract and run `wavescope`)
- **Windows** — `…-windows-x86_64.zip` (extract and run `wavescope.exe`)
- **macOS** — `…-macos-universal.zip` (universal Intel + Apple Silicon `.app`)

> **macOS:** the app is **not code-signed**. On first launch macOS Gatekeeper
> will block it — right-click the app → **Open** → **Open**, or run
> `xattr -dr com.apple.quarantine "Wavescope.app"`.

A GPU with a Vulkan / Metal / DX12 driver is required.

## Build from source

Requires a stable Rust toolchain.

```sh
cargo build --release
# binary at target/release/wavescope
```

### Linux build dependencies

```sh
sudo apt-get install -y \
  libxkbcommon-dev libwayland-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  libgl1-mesa-dev
```

### Tests

The FFT and end-to-end scalogram tests need a real GPU adapter; they
**self-skip** when none is available.

```sh
cargo test
```

## Releasing

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds all
three platforms, packages them (universal `.app` on macOS) and uploads the
archives to the matching GitHub Release.

```sh
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/ci.yml` compiles and lints (`clippy -D warnings`) on Linux,
Windows and macOS for every push / pull request.

## Citation

If you use Wavescope in your research, please cite it via its archived
[Zenodo record](https://doi.org/10.5281/zenodo.20573387):

```bibtex
@software{wavescope,
  author    = {Scherbakov, Alexey},
  title     = {Wavescope: A GPU Wavelet Scalogram Viewer for Audio},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20573387},
  url       = {https://doi.org/10.5281/zenodo.20573387}
}
```

The DOI `10.5281/zenodo.20573387` always resolves to the latest release.

## License

Released under the [MIT License](LICENSE).
