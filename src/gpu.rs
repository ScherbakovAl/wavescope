//! Portable GPU compute backend built on wgpu.
//!
//! Replaces the former CUDA driver-API + cuFFT layer. wgpu picks a native
//! backend per platform (Vulkan on Linux/Windows, Metal on macOS, DX12 on
//! Windows), so the same code runs on NVIDIA, AMD, Intel and Apple GPUs.
//!
//! The five compute kernels live as WGSL at the bottom of this file. The FFT
//! (cuFFT's replacement) is a batched radix-2 Stockham autosort transform:
//! `log2(N)` ping-pong passes, no bit-reversal, power-of-two `N` only — which
//! is what `cwt.rs` always provides (`n_fft = next_power_of_two(...)`).

use std::borrow::Cow;
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Uniform parameter blocks (must match the WGSL `struct P` layouts exactly:
// scalars are 4-byte aligned and packed; only trailing padding rounds the
// struct up to a 16-byte multiple as required by the uniform address space).
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RcParams {
    pub n: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WavParams {
    pub n: u32,
    /// Rows in this scale batch.
    pub num_scales: u32,
    /// First global scale row of the batch (indexes `scales` / `eta`).
    pub scale_offset: u32,
    pub kind: u32,
    pub p1: f32,
    pub p2: f32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MulParams {
    pub n: u32,
    pub num_scales: u32,
    pub _pad: [u32; 2],
}

/// Shared by the per-channel `extract` and the `cross_extract` kernels.
/// Columns are addressed globally: column `col` covers tape (downsampled
/// signal) samples `[bounds[col], bounds[col+1])`; the chunk currently on the
/// GPU holds tape samples `[chunk_lo, chunk_lo + chunk_len)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ExtractParams {
    pub n: u32,
    /// Rows in this scale batch.
    pub rows: u32,
    /// First global scale row of the batch.
    pub scale_offset: u32,
    /// Full output width in pixels (output stride).
    pub width: u32,
    /// First global column handled by this dispatch.
    pub col_start: u32,
    /// Number of columns handled by this dispatch.
    pub col_count: u32,
    /// Tape sample index of the chunk's first sample.
    pub chunk_lo: i32,
    /// Number of real samples uploaded for this chunk (≤ n).
    pub chunk_len: i32,
    /// Chunk-local end of the valid (non-padding) region.
    pub valid_end: i32,
    /// 1 ⇒ unweighted Δφ estimator (each phase increment normalised to unit
    /// magnitude before averaging); 0 ⇒ amplitude²-weighted (default).
    pub unit_weight: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n: u32,
    batch: u32,
    ns: u32,
    dir: f32,
}

// ---------------------------------------------------------------------------
// Bind-group / pipeline plumbing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Bind {
    StorageRead,
    StorageWrite,
    Uniform,
}

pub struct Pipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub p_real_to_complex: Pipeline,
    pub p_wavelet: Pipeline,
    pub p_multiply: Pipeline,
    pub p_extract: Pipeline,
    pub p_cross: Pipeline,
    pub p_fft: Pipeline,
}

const WG: u32 = 64;

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

impl GpuContext {
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan / Metal / DX12
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .context("no suitable GPU adapter found")?;

        let info = adapter.get_info();
        log::info!(
            "GPU adapter: {} ({:?}, {:?})",
            info.name, info.device_type, info.backend
        );

        // Request the adapter's full limits so the large complex buffers
        // (num_scales × n_fft) can exceed the conservative defaults.
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wavelet-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .context("failed to create GPU device")?;

        let p_real_to_complex = build_pipeline(
            &device,
            "real_to_complex",
            SRC_REAL_TO_COMPLEX,
            &[Bind::StorageRead, Bind::StorageWrite, Bind::Uniform],
        );
        let p_wavelet = build_pipeline(
            &device,
            "wavelet",
            SRC_WAVELET,
            &[
                Bind::StorageWrite, // wfreqs
                Bind::StorageRead,  // scales
                Bind::StorageRead,  // eta
                Bind::Uniform,
            ],
        );
        let p_multiply = build_pipeline(
            &device,
            "multiply",
            SRC_MULTIPLY,
            &[
                Bind::StorageRead,  // signal_fft
                Bind::StorageRead,  // wfreqs
                Bind::StorageWrite, // products
                Bind::Uniform,
            ],
        );
        let p_extract = build_pipeline(
            &device,
            "extract",
            SRC_EXTRACT,
            &[
                Bind::StorageRead,  // cwt rows
                Bind::StorageWrite, // scalogram
                Bind::StorageWrite, // phase
                Bind::StorageWrite, // coherence
                Bind::StorageWrite, // inst_dev
                Bind::StorageRead,  // scales
                Bind::StorageRead,  // eta
                Bind::StorageRead,  // column bounds
                Bind::Uniform,
            ],
        );
        let p_cross = build_pipeline(
            &device,
            "cross_extract",
            SRC_CROSS_EXTRACT,
            &[
                Bind::StorageRead,  // cwt channel 0
                Bind::StorageRead,  // cwt channel 1
                Bind::StorageWrite, // phase difference
                Bind::StorageWrite, // cross coherence
                Bind::StorageWrite, // geometric-mean amplitude
                Bind::StorageRead,  // column bounds
                Bind::Uniform,
            ],
        );
        let p_fft = build_pipeline(
            &device,
            "fft",
            SRC_FFT,
            &[Bind::StorageRead, Bind::StorageWrite, Bind::Uniform],
        );

        Ok(GpuContext {
            device,
            queue,
            p_real_to_complex,
            p_wavelet,
            p_multiply,
            p_extract,
            p_cross,
            p_fft,
        })
    }

    // -- buffer helpers ----------------------------------------------------

    /// Storage buffer usable as kernel in/out and as a copy source.
    pub fn storage(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// CPU-mappable buffer used to read results back.
    pub fn staging(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload_f32(&self, buf: &wgpu::Buffer, data: &[f32]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    fn uniform(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn bind(&self, layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &Pipeline,
        bind: &wgpu::BindGroup,
        wg: (u32, u32, u32),
    ) {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(&pipeline.pipeline);
        cp.set_bind_group(0, bind, &[]);
        cp.dispatch_workgroups(wg.0, wg.1, wg.2);
    }

    // -- kernels -----------------------------------------------------------

    pub fn real_to_complex(
        &self,
        enc: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) {
        let ub = self.uniform(bytemuck::bytes_of(&RcParams { n, _pad: [0; 3] }));
        let bg = self.bind(&self.p_real_to_complex.layout, &[input, output, &ub]);
        self.pass(enc, &self.p_real_to_complex, &bg, (ceil_div(n, WG), 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wavelet(
        &self,
        enc: &mut wgpu::CommandEncoder,
        wfreqs: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        eta: &wgpu::Buffer,
        params: WavParams,
    ) {
        let ub = self.uniform(bytemuck::bytes_of(&params));
        let bg = self.bind(&self.p_wavelet.layout, &[wfreqs, scales, eta, &ub]);
        self.pass(
            enc,
            &self.p_wavelet,
            &bg,
            (ceil_div(params.n, WG), params.num_scales, 1),
        );
    }

    pub fn multiply(
        &self,
        enc: &mut wgpu::CommandEncoder,
        signal_fft: &wgpu::Buffer,
        wfreqs: &wgpu::Buffer,
        products: &wgpu::Buffer,
        params: MulParams,
    ) {
        let ub = self.uniform(bytemuck::bytes_of(&params));
        let bg = self.bind(
            &self.p_multiply.layout,
            &[signal_fft, wfreqs, products, &ub],
        );
        self.pass(
            enc,
            &self.p_multiply,
            &bg,
            (ceil_div(params.n, WG), params.num_scales, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &self,
        enc: &mut wgpu::CommandEncoder,
        cwt: &wgpu::Buffer,
        scalo: &wgpu::Buffer,
        phase: &wgpu::Buffer,
        coher: &wgpu::Buffer,
        instdev: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        eta: &wgpu::Buffer,
        bounds: &wgpu::Buffer,
        params: ExtractParams,
    ) {
        let ub = self.uniform(bytemuck::bytes_of(&params));
        let bg = self.bind(
            &self.p_extract.layout,
            &[cwt, scalo, phase, coher, instdev, scales, eta, bounds, &ub],
        );
        self.pass(
            enc,
            &self.p_extract,
            &bg,
            (ceil_div(params.col_count, WG), params.rows, 1),
        );
    }

    /// Cross-channel wavelet spectrum per pixel: phase difference
    /// arg(Σ W₀·W̄₁), cross-coherence |Σ W₀·W̄₁| / Σ|W₀||W₁| and the
    /// geometric-mean amplitude — exposes the relative phase between the two
    /// channels (the canonical synchronisation observable).
    #[allow(clippy::too_many_arguments)]
    pub fn cross_extract(
        &self,
        enc: &mut wgpu::CommandEncoder,
        cwt0: &wgpu::Buffer,
        cwt1: &wgpu::Buffer,
        phase: &wgpu::Buffer,
        coher: &wgpu::Buffer,
        amp: &wgpu::Buffer,
        bounds: &wgpu::Buffer,
        params: ExtractParams,
    ) {
        let ub = self.uniform(bytemuck::bytes_of(&params));
        let bg = self.bind(
            &self.p_cross.layout,
            &[cwt0, cwt1, phase, coher, amp, bounds, &ub],
        );
        self.pass(
            enc,
            &self.p_cross,
            &bg,
            (ceil_div(params.col_count, WG), params.rows, 1),
        );
    }

    /// Batched radix-2 Stockham FFT. `n` must be a power of two. The result is
    /// guaranteed to land in `output`; `scratch` is used as the ping-pong
    /// partner (must hold `batch * n` complex elements). Forward is unnormalised;
    /// inverse is also unnormalised (the 1/N scaling is applied in `extract`),
    /// matching cuFFT's convention.
    #[allow(clippy::too_many_arguments)]
    pub fn fft(
        &self,
        enc: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        n: u32,
        batch: u32,
        inverse: bool,
    ) {
        let stages = n.trailing_zeros();
        let dir = if inverse { 1.0 } else { -1.0 };
        let wg_x = ceil_div(n / 2, WG);

        // Pick the first stage's destination so the final stage writes `output`.
        let mut dst_is_output = (stages % 2) == 1;
        let mut src = input;
        for stage in 0..stages {
            let dst = if dst_is_output { output } else { scratch };
            let ub = self.uniform(bytemuck::bytes_of(&FftParams {
                n,
                batch,
                ns: 1u32 << stage,
                dir,
            }));
            let bg = self.bind(&self.p_fft.layout, &[src, dst, &ub]);
            self.pass(enc, &self.p_fft, &bg, (wg_x, batch, 1));
            src = dst;
            dst_is_output = !dst_is_output;
        }
    }

    /// Map a staging buffer and copy out `len` f32 values. Blocks until the GPU
    /// work feeding this buffer has completed.
    pub fn download_f32(&self, staging: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async channel").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..len].to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

// ---------------------------------------------------------------------------
// Pipeline builder
// ---------------------------------------------------------------------------

fn build_pipeline(device: &wgpu::Device, label: &str, src: &str, binds: &[Bind]) -> Pipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });

    let entries: Vec<wgpu::BindGroupLayoutEntry> = binds
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match b {
                    Bind::StorageRead => wgpu::BufferBindingType::Storage { read_only: true },
                    Bind::StorageWrite => wgpu::BufferBindingType::Storage { read_only: false },
                    Bind::Uniform => wgpu::BufferBindingType::Uniform,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    Pipeline { pipeline, layout }
}

// ---------------------------------------------------------------------------
// WGSL kernels (complex = vec2<f32>: x = real, y = imag)
// ---------------------------------------------------------------------------

const SRC_REAL_TO_COMPLEX: &str = r#"
struct P { n: u32 };
@group(0) @binding(0) var<storage, read>       rin:  array<f32>;
@group(0) @binding(1) var<storage, read_write> rout: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             p:    P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    rout[i] = vec2<f32>(rin[i], 0.0);
}
"#;

const SRC_WAVELET: &str = r#"
struct P { n: u32, num_scales: u32, scale_offset: u32, kind: u32, p1: f32, p2: f32 };
@group(0) @binding(0) var<storage, read_write> wfreqs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       eta:    array<f32>;
@group(0) @binding(3) var<uniform>             p:      P;

const PI: f32 = 3.141592653589793;

// All families are L1-normalised: ψ̂ peaks at exactly 1, with no √s factor.
// A unit-amplitude sinusoid then lights its ridge with |W| = 0.5 regardless
// of frequency, zoom or decimation — amplitude-faithful across the band.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let s = gid.y;
    if (k >= p.n || s >= p.num_scales) { return; }

    let srow  = p.scale_offset + s;
    let scale = scales[srow];
    let N = f32(p.n);
    var omega: f32;
    if (k <= p.n / 2u) {
        omega = 2.0 * PI * f32(k) / N;
    } else {
        omega = 2.0 * PI * (f32(k) - N) / N;
    }

    let w  = scale * omega;
    var val = 0.0;

    if (w > 0.0) {
        switch p.kind {
            case 0u: {   // Morlet
                let diff = w - eta[srow];
                val = exp(-0.5 * diff * diff);
            }
            case 1u: {   // Generalized Morse
                let beta  = p.p1;
                let gamma = p.p2;
                let wpeak = pow(beta / gamma, 1.0 / gamma);
                let lg = beta * log(w / wpeak) - pow(w, gamma) + beta / gamma;
                val = exp(lg);
            }
            case 2u: {   // Bump
                let mu    = eta[srow];
                let sigma = p.p1;
                let x = (w - mu) / sigma;
                if (x > -1.0 && x < 1.0) {
                    val = exp(1.0 - 1.0 / (1.0 - x * x));
                }
            }
            case 3u: {   // Paul
                let m = p.p1;
                let lg = m * log(w / m) - w + m;
                val = exp(lg);
            }
            default: { }
        }
    }

    wfreqs[s * p.n + k] = vec2<f32>(val, 0.0);
}
"#;

const SRC_MULTIPLY: &str = r#"
struct P { n: u32, num_scales: u32 };
@group(0) @binding(0) var<storage, read>       sig:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       wf:   array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> prod: array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             p:    P;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let s = gid.y;
    if (k >= p.n || s >= p.num_scales) { return; }
    let idx = s * p.n + k;
    prod[idx] = cmul(sig[k], wf[idx]);
}
"#;

const SRC_FFT: &str = r#"
struct P { n: u32, batch: u32, ns: u32, dir: f32 };
@group(0) @binding(0) var<storage, read>       fin:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fout: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             p:    P;

const PI: f32 = 3.141592653589793;

// One radix-2 Stockham butterfly per (j in [0,N/2), batch b).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let b = gid.y;
    let half = p.n / 2u;
    if (j >= half || b >= p.batch) { return; }

    let k    = j & (p.ns - 1u);
    let base = (j - k) << 1u;
    let off  = b * p.n;

    let a  = fin[off + j];
    let bb = fin[off + j + half];

    let angle = p.dir * PI * f32(k) / f32(p.ns);
    let tw = vec2<f32>(cos(angle), sin(angle));
    let t  = vec2<f32>(bb.x * tw.x - bb.y * tw.y, bb.x * tw.y + bb.y * tw.x);

    fout[off + base + k]        = a + t;
    fout[off + base + k + p.ns] = a - t;
}
"#;

const SRC_EXTRACT: &str = r#"
struct P {
    n: u32, rows: u32, scale_offset: u32, width: u32,
    col_start: u32, col_count: u32,
    chunk_lo: i32, chunk_len: i32, valid_end: i32,
    unit_weight: u32,
};
@group(0) @binding(0) var<storage, read>       cwt:     array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> scalo:   array<f32>;
@group(0) @binding(2) var<storage, read_write> phase:   array<f32>;
@group(0) @binding(3) var<storage, read_write> coher:   array<f32>;
@group(0) @binding(4) var<storage, read_write> instdev: array<f32>;
@group(0) @binding(5) var<storage, read>       scales:  array<f32>;
@group(0) @binding(6) var<storage, read>       eta:     array<f32>;
@group(0) @binding(7) var<storage, read>       bounds:  array<u32>;
@group(0) @binding(8) var<uniform>             p:       P;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cconj(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let s  = gid.y;
    if (ci >= p.col_count || s >= p.rows) { return; }
    let col  = p.col_start + ci;
    let srow = p.scale_offset + s;

    let outidx = srow * p.width + col;
    let Ni = i32(p.n);

    // Exact column → tape-sample bounds (integer, computed on the CPU),
    // translated to chunk-local indices.
    var samp_start = i32(bounds[col])      - p.chunk_lo;
    var samp_end   = i32(bounds[col + 1u]) - p.chunk_lo;
    if (samp_end > p.valid_end) { samp_end = p.valid_end; }
    if (samp_end > Ni)          { samp_end = Ni; }
    if (samp_start < 0)         { samp_start = 0; }
    if (samp_end <= samp_start) { samp_end = samp_start + 1; }
    if (samp_start >= Ni || samp_start >= p.valid_end) {
        scalo[outidx]   = 0.0;
        phase[outidx]   = 0.0;
        coher[outidx]   = 0.0;
        instdev[outidx] = 0.0;
        return;
    }
    let count = samp_end - samp_start;

    var sum = 0.0;
    var sre = 0.0;
    var sim = 0.0;
    var acc = vec2<f32>(0.0, 0.0);
    let row_off = s * p.n;

    for (var t = samp_start; t < samp_end; t = t + 1) {
        let c = cwt[row_off + u32(t)];
        sum += length(c);
        sre += c.x;
        sim += c.y;
        // Phase increment to the next sample; t+1 stays inside the chunk's
        // real data (padding region holds real signal, so pairs may cross
        // the column / chunk-valid boundary).
        if (t + 1 < p.chunk_len) {
            let cn = cwt[row_off + u32(t + 1)];
            var z = cmul(cn, cconj(c));
            if (p.unit_weight == 1u) {
                let l = length(z);
                if (l > 0.0) { z = z / l; } else { z = vec2<f32>(0.0, 0.0); }
            }
            acc += z;
        }
    }

    scalo[outidx] = sum / (f32(p.n) * f32(count));
    phase[outidx] = atan2(sim, sre);
    let mag = sqrt(sre * sre + sim * sim);
    coher[outidx] = select(0.0, mag / sum, sum > 0.0);
    let dphi  = atan2(acc.y, acc.x);
    let eta_s = eta[srow];
    instdev[outidx] = select(0.0, dphi * scales[srow] / eta_s - 1.0, eta_s > 0.0);
}
"#;

const SRC_CROSS_EXTRACT: &str = r#"
struct P {
    n: u32, rows: u32, scale_offset: u32, width: u32,
    col_start: u32, col_count: u32,
    chunk_lo: i32, chunk_len: i32, valid_end: i32,
    unit_weight: u32,
};
@group(0) @binding(0) var<storage, read>       cwt0:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       cwt1:  array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> phase: array<f32>;
@group(0) @binding(3) var<storage, read_write> coher: array<f32>;
@group(0) @binding(4) var<storage, read_write> amp:   array<f32>;
@group(0) @binding(5) var<storage, read>       bounds: array<u32>;
@group(0) @binding(6) var<uniform>             p:     P;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cconj(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let s  = gid.y;
    if (ci >= p.col_count || s >= p.rows) { return; }
    let col  = p.col_start + ci;
    let srow = p.scale_offset + s;

    let outidx = srow * p.width + col;
    let Ni = i32(p.n);

    var samp_start = i32(bounds[col])      - p.chunk_lo;
    var samp_end   = i32(bounds[col + 1u]) - p.chunk_lo;
    if (samp_end > p.valid_end) { samp_end = p.valid_end; }
    if (samp_end > Ni)          { samp_end = Ni; }
    if (samp_start < 0)         { samp_start = 0; }
    if (samp_end <= samp_start) { samp_end = samp_start + 1; }
    if (samp_start >= Ni || samp_start >= p.valid_end) {
        phase[outidx] = 0.0;
        coher[outidx] = 0.0;
        amp[outidx]   = 0.0;
        return;
    }
    let count = samp_end - samp_start;

    var acc  = vec2<f32>(0.0, 0.0);   // Σ W0·conj(W1)
    var wsum = 0.0;                   // Σ |W0|·|W1|
    var asum = 0.0;                   // Σ sqrt(|W0|·|W1|)
    let row_off = s * p.n;

    for (var t = samp_start; t < samp_end; t = t + 1) {
        let c0 = cwt0[row_off + u32(t)];
        let c1 = cwt1[row_off + u32(t)];
        acc += cmul(c0, cconj(c1));
        let m = length(c0) * length(c1);
        wsum += m;
        asum += sqrt(m);
    }

    phase[outidx] = atan2(acc.y, acc.x);
    coher[outidx] = select(0.0, length(acc) / wsum, wsum > 0.0);
    amp[outidx]   = asum / (f32(p.n) * f32(count));
}
"#;

// ---------------------------------------------------------------------------
// Tests — validate the Stockham FFT against a naive DFT. These require a GPU
// adapter; if none is available the test prints a skip notice and passes.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run_fft(gpu: &GpuContext, input: &[f32], n: u32, batch: u32, inverse: bool) -> Vec<f32> {
        let bytes = (input.len() * 4) as u64;
        let d_in = gpu.storage(bytes);
        let d_out = gpu.storage(bytes);
        let d_scratch = gpu.storage(bytes);
        gpu.upload_f32(&d_in, input);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gpu.fft(&mut enc, &d_in, &d_out, &d_scratch, n, batch, inverse);
        let staging = gpu.staging(bytes);
        enc.copy_buffer_to_buffer(&d_out, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(enc.finish()));
        gpu.download_f32(&staging, input.len())
    }

    fn naive_dft(x: &[(f64, f64)], inverse: bool) -> Vec<(f64, f64)> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, &(xr, xi)) in x.iter().enumerate() {
                    let ang = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    let (s, c) = ang.sin_cos();
                    re += xr * c - xi * s;
                    im += xr * s + xi * c;
                }
                (re, im)
            })
            .collect()
    }

    macro_rules! gpu_or_skip {
        () => {
            match GpuContext::new() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping GPU test (no adapter): {e}");
                    return;
                }
            }
        };
    }

    #[test]
    fn fft_matches_dft() {
        let gpu = gpu_or_skip!();
        let n = 16usize;
        let mut input = Vec::new();
        let mut x = Vec::new();
        for i in 0..n {
            let re = ((i * 7 + 3) % 13) as f64 - 6.0;
            let im = ((i * 5 + 1) % 11) as f64 - 5.0;
            input.push(re as f32);
            input.push(im as f32);
            x.push((re, im));
        }
        let out = run_fft(&gpu, &input, n as u32, 1, false);
        let expect = naive_dft(&x, false);
        for k in 0..n {
            let (gr, gi) = (out[2 * k] as f64, out[2 * k + 1] as f64);
            let (er, ei) = expect[k];
            assert!(
                (gr - er).abs() < 1e-3 && (gi - ei).abs() < 1e-3,
                "k={k}: gpu=({gr:.4},{gi:.4}) dft=({er:.4},{ei:.4})"
            );
        }
    }

    #[test]
    fn fft_roundtrip_is_identity_times_n() {
        let gpu = gpu_or_skip!();
        let n = 64usize;
        let mut input = Vec::new();
        for i in 0..n {
            input.push((i as f32 * 0.1).sin());
            input.push(0.0);
        }
        let fwd = run_fft(&gpu, &input, n as u32, 1, false);
        let inv = run_fft(&gpu, &fwd, n as u32, 1, true);
        for i in 0..n {
            let r = inv[2 * i] / n as f32;
            let im = inv[2 * i + 1] / n as f32;
            assert!((r - input[2 * i]).abs() < 1e-3, "i={i}: {r} vs {}", input[2 * i]);
            assert!(im.abs() < 1e-3, "i={i}: imag {im}");
        }
    }

    #[test]
    fn fft_batched_independent_rows() {
        let gpu = gpu_or_skip!();
        let n = 8usize;
        let batch = 3usize;
        let mut input = Vec::new();
        let mut rows = Vec::new();
        for b in 0..batch {
            let mut row = Vec::new();
            for i in 0..n {
                let re = ((i * 3 + b * 2) % 7) as f64 - 3.0;
                input.push(re as f32);
                input.push(0.0);
                row.push((re, 0.0));
            }
            rows.push(row);
        }
        let out = run_fft(&gpu, &input, n as u32, batch as u32, false);
        for (b, row) in rows.iter().enumerate() {
            let expect = naive_dft(row, false);
            for (k, &(er, ei)) in expect.iter().enumerate() {
                let idx = b * n + k;
                let (gr, gi) = (out[2 * idx] as f64, out[2 * idx + 1] as f64);
                assert!(
                    (gr - er).abs() < 1e-3 && (gi - ei).abs() < 1e-3,
                    "b={b} k={k}: gpu=({gr:.4},{gi:.4}) dft=({er:.4},{ei:.4})"
                );
            }
        }
    }
}
