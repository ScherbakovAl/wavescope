fn main() {
    let cuda_root = std::env::var("CUDA_ROOT")
        .unwrap_or_else(|_| "/usr/local/cuda-13.1".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
}
