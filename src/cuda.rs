use std::ffi::{c_void, CStr, CString};
use std::ptr;
use anyhow::Context;

// ---------------------------------------------------------------------------
// Raw FFI types
// ---------------------------------------------------------------------------

pub type CUresult    = i32;
pub type CUdevice    = i32;
pub type CUcontext   = *mut c_void;
pub type CUmodule    = *mut c_void;
pub type CUfunction  = *mut c_void;
pub type CUdeviceptr = u64;
pub type CufftHandle = u32;
pub type CufftResult = i32;

pub const CUDA_SUCCESS:   CUresult    = 0;
pub const CUFFT_SUCCESS:  CufftResult = 0;
pub const CUFFT_C2C:      i32         = 0x29;
pub const CUFFT_FORWARD:  i32         = -1;
pub const CUFFT_INVERSE:  i32         = 1;

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

mod sys {
    use super::*;

    #[link(name = "cuda")]
    extern "C" {
        pub fn cuInit(flags: u32) -> CUresult;
        pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
        pub fn cuCtxCreate_v2(
            pctx: *mut CUcontext, flags: u32, dev: CUdevice,
        ) -> CUresult;
        pub fn cuModuleLoadData(
            module: *mut CUmodule, image: *const c_void,
        ) -> CUresult;
        pub fn cuModuleGetFunction(
            hfunc: *mut CUfunction,
            hmod: CUmodule,
            name: *const std::os::raw::c_char,
        ) -> CUresult;
        pub fn cuLaunchKernel(
            f: CUfunction,
            gridDimX: u32, gridDimY: u32, gridDimZ: u32,
            blockDimX: u32, blockDimY: u32, blockDimZ: u32,
            sharedMemBytes: u32,
            hStream: *mut c_void,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
        pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        pub fn cuMemcpyHtoD_v2(
            dst: CUdeviceptr, src: *const c_void, byte_count: usize,
        ) -> CUresult;
        pub fn cuMemcpyDtoH_v2(
            dst: *mut c_void, src: CUdeviceptr, byte_count: usize,
        ) -> CUresult;
        pub fn cuCtxSynchronize() -> CUresult;
        pub fn cuGetErrorString(
            error: CUresult, pStr: *mut *const std::os::raw::c_char,
        ) -> CUresult;
    }

    #[link(name = "cufft")]
    extern "C" {
        pub fn cufftPlan1d(
            plan: *mut CufftHandle, nx: i32, type_: i32, batch: i32,
        ) -> CufftResult;
        pub fn cufftPlanMany(
            plan: *mut CufftHandle,
            rank: i32,
            n: *mut i32,
            inembed: *mut i32, istride: i32, idist: i32,
            onembed: *mut i32, ostride: i32, odist: i32,
            type_: i32, batch: i32,
        ) -> CufftResult;
        pub fn cufftExecC2C(
            plan: CufftHandle,
            idata: CUdeviceptr, odata: CUdeviceptr,
            direction: i32,
        ) -> CufftResult;
        pub fn cufftDestroy(plan: CufftHandle) -> CufftResult;
    }
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn cuda_error_string(result: CUresult) -> String {
    unsafe {
        let mut ptr: *const std::os::raw::c_char = ptr::null();
        sys::cuGetErrorString(result, &mut ptr);
        if ptr.is_null() {
            format!("CUDA error code {}", result)
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub fn cuda_check(result: CUresult, ctx: &str) -> anyhow::Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        anyhow::bail!("{}: {}", ctx, cuda_error_string(result))
    }
}

pub fn cufft_check(result: CufftResult, ctx: &str) -> anyhow::Result<()> {
    if result == CUFFT_SUCCESS {
        Ok(())
    } else {
        anyhow::bail!("{}: cuFFT error {}", ctx, result)
    }
}

// ---------------------------------------------------------------------------
// CudaContext
// ---------------------------------------------------------------------------

pub struct CudaContext {
    _ctx: CUcontext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn init() -> anyhow::Result<Self> {
        unsafe {
            cuda_check(sys::cuInit(0), "cuInit")?;
            let mut device: CUdevice = 0;
            cuda_check(sys::cuDeviceGet(&mut device, 0), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            cuda_check(
                sys::cuCtxCreate_v2(&mut ctx, 0, device),
                "cuCtxCreate",
            )?;
            Ok(CudaContext { _ctx: ctx })
        }
    }

    pub fn load_ptx(&self, ptx: &str) -> anyhow::Result<CudaModule> {
        let cstr = CString::new(ptx).context("PTX contains null byte")?;
        unsafe {
            let mut module: CUmodule = ptr::null_mut();
            cuda_check(
                sys::cuModuleLoadData(
                    &mut module,
                    cstr.as_ptr() as *const c_void,
                ),
                "cuModuleLoadData",
            )?;
            Ok(CudaModule { module })
        }
    }

    pub fn synchronize(&self) -> anyhow::Result<()> {
        unsafe { cuda_check(sys::cuCtxSynchronize(), "cuCtxSynchronize") }
    }
}

// ---------------------------------------------------------------------------
// CudaModule
// ---------------------------------------------------------------------------

pub struct CudaModule {
    pub module: CUmodule,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    pub fn get_function(&self, name: &str) -> anyhow::Result<CudaFunction> {
        let cname = CString::new(name).unwrap();
        unsafe {
            let mut func: CUfunction = ptr::null_mut();
            cuda_check(
                sys::cuModuleGetFunction(&mut func, self.module, cname.as_ptr()),
                &format!("cuModuleGetFunction({})", name),
            )?;
            Ok(CudaFunction { func })
        }
    }
}

// ---------------------------------------------------------------------------
// CudaFunction
// ---------------------------------------------------------------------------

pub struct CudaFunction {
    pub func: CUfunction,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    /// Launch with a flat array of `*mut c_void` pointers to kernel arguments.
    pub unsafe fn launch(
        &self,
        grid:  (u32, u32, u32),
        block: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> anyhow::Result<()> {
        cuda_check(
            sys::cuLaunchKernel(
                self.func,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )
    }
}

// ---------------------------------------------------------------------------
// CudaBuffer
// ---------------------------------------------------------------------------

/// Owns a GPU memory allocation (device pointer + byte size).
pub struct CudaBuffer {
    ptr:  CUdeviceptr,
    pub size: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    pub fn alloc(size_bytes: usize) -> anyhow::Result<Self> {
        if size_bytes == 0 {
            return Ok(CudaBuffer { ptr: 0, size: 0 });
        }
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            cuda_check(
                sys::cuMemAlloc_v2(&mut ptr, size_bytes),
                "cuMemAlloc",
            )?;
            Ok(CudaBuffer { ptr, size: size_bytes })
        }
    }

    pub fn ptr(&self) -> CUdeviceptr { self.ptr }

    pub fn upload_raw(&self, data: &[u8]) -> anyhow::Result<()> {
        assert!(data.len() <= self.size, "upload_raw: data exceeds buffer");
        if data.is_empty() { return Ok(()); }
        unsafe {
            cuda_check(
                sys::cuMemcpyHtoD_v2(
                    self.ptr,
                    data.as_ptr() as *const c_void,
                    data.len(),
                ),
                "cuMemcpyHtoD",
            )
        }
    }

    pub fn upload_f32(&self, data: &[f32]) -> anyhow::Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.upload_raw(bytes)
    }

    pub fn download_f32(&self, data: &mut [f32]) -> anyhow::Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
        };
        assert!(bytes.len() <= self.size, "download_f32: output exceeds buffer");
        if bytes.is_empty() { return Ok(()); }
        unsafe {
            cuda_check(
                sys::cuMemcpyDtoH_v2(
                    bytes.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes.len(),
                ),
                "cuMemcpyDtoH",
            )
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { sys::cuMemFree_v2(self.ptr); }
        }
    }
}

// ---------------------------------------------------------------------------
// CufftPlan
// ---------------------------------------------------------------------------

pub struct CufftPlan {
    handle: CufftHandle,
}

unsafe impl Send for CufftPlan {}

impl CufftPlan {
    /// Single C2C plan (one transform of length `n_fft`).
    pub fn plan_single_c2c(n_fft: usize) -> anyhow::Result<Self> {
        let mut handle: CufftHandle = 0;
        unsafe {
            cufft_check(
                sys::cufftPlan1d(&mut handle, n_fft as i32, CUFFT_C2C, 1),
                "cufftPlan1d",
            )?;
        }
        Ok(CufftPlan { handle })
    }

    /// Batched C2C plan: `batch` transforms of length `n_fft`.
    pub fn plan_batch_c2c(n_fft: usize, batch: usize) -> anyhow::Result<Self> {
        let mut handle: CufftHandle = 0;
        let mut n = n_fft as i32;
        unsafe {
            cufft_check(
                sys::cufftPlanMany(
                    &mut handle,
                    1, &mut n,
                    ptr::null_mut(), 1, n_fft as i32,
                    ptr::null_mut(), 1, n_fft as i32,
                    CUFFT_C2C, batch as i32,
                ),
                "cufftPlanMany",
            )?;
        }
        Ok(CufftPlan { handle })
    }

    pub fn exec_c2c(
        &self,
        idata: CUdeviceptr,
        odata: CUdeviceptr,
        direction: i32,
    ) -> anyhow::Result<()> {
        unsafe {
            cufft_check(
                sys::cufftExecC2C(self.handle, idata, odata, direction),
                "cufftExecC2C",
            )
        }
    }
}

impl Drop for CufftPlan {
    fn drop(&mut self) {
        unsafe { sys::cufftDestroy(self.handle); }
    }
}
