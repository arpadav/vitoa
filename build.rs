// build script intentionally panics on unsupported arches — that IS the contract
#![allow(clippy::panic)]
//! Build script: validates target architecture and warns on missing AVX-512 features
//!
//! Runs at compile time to:
//! * Abort unsupported target architectures with a clear panic message
//! * Emit a cargo warning when building on non-x86_64 hosts (scalar fallback)
//! * Emit a cargo warning when required AVX-512 target features are absent on
//!   x86_64, so callers know why the SIMD fast path was not selected
//! * Register rerun triggers so cargo re-executes the script only when relevant
//!   inputs change
//!
//! Author: aav
fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| String::from("unknown"));
    const SUPPORTED: &[&str] = &["x86_64", "aarch64", "x86", "riscv64", "wasm32"];
    if !SUPPORTED.contains(&arch.as_str()) {
        panic!(
            "vitoa: unsupported target architecture `{arch}`.\n\
             Supported: {SUPPORTED:?}. \
             x86_64 with AVX-512 IFMA gets the SIMD fast path; others use scalar fallback.\n\
             Open an issue if you need this target."
        );
    }
    if arch != "x86_64" {
        println!(
            "cargo:warning=vitoa: building on `{arch}` — using scalar fallback (~3-4x slower than AVX-512 IFMA on x86_64)."
        );
    } else {
        // Dispatch is purely compile-time (no CPUID), so missing features here
        // mean the scalar path is used even on capable CPUs.
        let active_features: std::collections::HashSet<String> =
            std::env::var("CARGO_CFG_TARGET_FEATURE")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.to_string())
                .collect();
        const REQUIRED: &[&str] = &["avx512f", "avx512ifma", "avx512vbmi", "avx512bw"];
        let missing: Vec<&str> = REQUIRED
            .iter()
            .copied()
            .filter(|f| !active_features.contains(*f))
            .collect();
        if !missing.is_empty() {
            println!(
                "cargo:warning=vitoa: x86_64 build missing target features {missing:?} — scalar fallback in use. \
                 Set RUSTFLAGS='-C target-cpu=native' or \
                 RUSTFLAGS='-C target-feature=+avx512f,+avx512ifma,+avx512vbmi,+avx512bw' to enable the AVX-512 IFMA fast path."
            );
        }
    }
    // Register kani cfgs so #[cfg(kani)] / #[cfg(kani_slow)] don't trip
    // unexpected_cfgs warnings under normal cargo (kani's runner sets them).
    println!("cargo:rustc-check-cfg=cfg(kani)");
    println!("cargo:rustc-check-cfg=cfg(kani_slow)");
    // `cfg(simd_ifma)` alias condenses the 5-condition gate into one flag.
    println!("cargo:rustc-check-cfg=cfg(simd_ifma)");
    if arch == "x86_64" {
        let active: std::collections::HashSet<String> = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.to_string())
            .collect();
        let required = ["avx512f", "avx512ifma", "avx512vbmi", "avx512bw"];
        if required.iter().all(|f| active.contains(*f)) {
            println!("cargo:rustc-cfg=simd_ifma");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
}
