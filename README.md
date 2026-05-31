# Wavescope

*A GPU wavelet scalogram viewer for audio.*

GPU-accelerated continuous-wavelet-transform (CWT) viewer for WAV/FLAC audio.
Interactive scalogram with zoom/pan and amplitude / phase / instantaneous-frequency
views.

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
