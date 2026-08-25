//! Compiles the CUDA device code (`src/cuda/kernels.cu`) to PTX with `nvcc` and
//! exposes the resulting path to the crate via the `KERNELS_PTX` env var, so the
//! host code can embed it with `include_str!(env!("KERNELS_PTX"))`.
//!
//! Target arch is `sm_86` (the RTX 3050 is Ampere GA107). PTX is forward
//! compatible, so the driver JITs it for the actual device at load time.

use std::path::PathBuf;
use std::process::Command;

const CUDA_SRC: &str = "src/cuda/kernels.cu";
const ARCH: &str = "sm_86";

fn main() {
    println!("cargo:rerun-if-changed={CUDA_SRC}");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_path = out_dir.join("kernels.ptx");

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .args(["--ptx", "-arch", ARCH, "-o"])
        .arg(&ptx_path)
        .arg(CUDA_SRC)
        // -lineinfo keeps PTX debuggable without disabling optimization.
        .arg("-lineinfo")
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{nvcc}`: {e}. Is the CUDA toolkit installed?"));

    if !status.success() {
        panic!("`{nvcc} --ptx {CUDA_SRC}` failed (see errors above)");
    }

    println!("cargo:rustc-env=KERNELS_PTX={}", ptx_path.display());
}
