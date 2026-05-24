// Restriction lints are lifted for test code so `unwrap()` etc. remain ergonomic.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
    )
)]
#![doc = include_str!("../README.md")]
//!
//! ---
//!
//! # Implementation notes
//!
//! Rust port of Champagne Gareau & Lemire, "Converting an Integer to a Decimal
//! String in Under Two Nanoseconds" (SPE 2026, doi:10.1002/spe.70079)
//!
//! Implements the heterogeneous AVX-512 IFMA variant (paper §5.4, Figure 6),
//! the homogeneous variant for 17–20 digit values (§5.5, Figure 7), a dynamic
//! batch selector (§5.6, Algorithm 1), and a scalar fallback for non-AVX-512 hosts
//!
//! Author: aav
// --------------------------------------------------
// mods
// --------------------------------------------------
pub mod batch;
mod digits;
pub mod error;
#[cfg(kani)]
mod kani_proofs;
#[cfg(feature = "macros")]
mod macros;
mod scalar;
pub mod selector;
#[cfg(simd_ifma)]
pub(crate) mod simd;
#[cfg(feature = "macros")]
pub mod support;
mod u128_div;
// --------------------------------------------------
// re-exports
// --------------------------------------------------
pub use batch::{fmt_batch, fmt_batch_joined, fmt_batch_with};
pub use digits::count_digits;
pub use error::Error;
pub use selector::BatchConfig;
#[cfg(feature = "macros")]
pub use support::{FastIntArg, FastIntWrite};

#[inline(always)]
/// Decimal digit count of `n` (1 for n == 0)
///
/// Thin alias over [`digits::count_digits`] exported so existing callers
/// that reference `digit_count` by name continue to compile without change
///
/// # Arguments
///
/// * `n` - The value whose decimal digit count is requested
///
/// # Example
///
/// ```rust
/// assert_eq!(vitoa::digit_count(0), 1);
/// assert_eq!(vitoa::digit_count(999), 3);
/// assert_eq!(vitoa::digit_count(u64::MAX), 20);
/// ```
pub fn digit_count(n: u64) -> u32 {
    digits::count_digits(n)
}

#[cfg(simd_ifma)]
/// Compile-time flag: AVX-512 IFMA fast path is available
///
/// True when all four required target features (`avx512f`, `avx512ifma`,
/// `avx512vbmi`, `avx512bw`) are enabled at compile time on x86_64. The
/// compiler dead-code-eliminates the opposite branch
pub const HAS_AVX512_IFMA: bool = true;

#[cfg(not(simd_ifma))]
/// Compile-time flag: AVX-512 IFMA fast path is unavailable, scalar in use
pub const HAS_AVX512_IFMA: bool = false;

#[cfg(simd_ifma)]
#[inline(always)]
/// Dispatch to the AVX-512 SIMD or scalar implementation, chosen at compile time
///
/// When [`HAS_AVX512_IFMA`] is true, delegates to [`simd::heterogeneous`]
/// Otherwise falls back to [`scalar::scalar_to_chars`] via a reconstructed
/// slice. No runtime branch — the unused arm is removed by `#[cfg]`
///
/// # Arguments
///
/// * `value` - The `u64` value to convert
/// * `n` - The pre-computed decimal digit count (must equal `digit_count(value)`)
/// * `ptr` - Pointer to the first byte of the output region
///
/// # Safety
///
/// `ptr` must be valid for `n` byte writes
pub(crate) unsafe fn dispatch_simd(value: u64, n: usize, ptr: *mut u8) {
    unsafe {
        simd::heterogeneous(value, n, ptr);
    }
}

#[cfg(not(simd_ifma))]
#[inline(always)]
/// Scalar dispatch when AVX-512 IFMA is not enabled at compile time
///
/// # Safety
///
/// `ptr` must be valid for `n` byte writes
pub(crate) unsafe fn dispatch_simd(value: u64, n: usize, ptr: *mut u8) {
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, n) };
    scalar::scalar_to_chars(value, n, slice);
}

#[inline]
/// `u64` decimal-to-ASCII implementation (internal helper for [`Decimal`])
///
/// All public callers go through the [`fmt`] generic entry point or the
/// trait dispatch table. This is the actual u64 worker: digit count, bounds
/// check, then SIMD or scalar path
///
/// # Arguments
///
/// * `value` - The `u64` value to convert
/// * `out` - Output byte slice; must have length >= `digit_count(value)`
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] when `out.len() < digit_count(value)`
pub(crate) fn fmt_u64_inner(value: u64, out: &mut [u8]) -> Result<usize, Error> {
    let n = digit_count(value) as usize;
    if out.len() < n {
        return Err(Error::BufferTooSmall {
            need: n,
            have: out.len(),
        });
    }
    // Small-value fast path (value < 10^8, i.e. 1–8 digits): single-kernel 8-digit
    // SIMD path avoids the wasted high-half IFMA + permutex of the 16-digit kernel.
    // ~3 ns saving on small u64 values.
    #[cfg(simd_ifma)]
    if value < 100_000_000u64 {
        // SAFETY: value < 10^8 fits in u32; n is its digit count (1..=8).
        unsafe {
            simd::u32_le_1e8(value as u32, n, out.as_mut_ptr());
        }
        return Ok(n);
    }
    unsafe {
        dispatch_simd(value, n, out.as_mut_ptr());
    }
    Ok(n)
}

/// Sealed dispatch trait — every unsigned integer width that [`fmt`] accepts
///
/// The public [`fmt`] function takes any `T: Decimal` and forwards to the
/// per-width implementation via this trait. Implementations are sealed and
/// only provided by this crate for `u8`, `u16`, `u32`, `u64`, and `u128`.
/// Other input types are rejected at compile time rather than silently
/// truncated
pub trait Decimal: __decimal_sealed::Sealed {
    /// Write `self` to `out` as decimal ASCII, returning bytes written
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] when `out` is too small
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error>;
}

#[doc(hidden)]
pub mod __decimal_sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
}

impl Decimal for u8 {
    #[inline]
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error> {
        fmt_bounded_inner::<3>(self as u64, out)
    }
}
impl Decimal for u16 {
    #[inline]
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error> {
        fmt_bounded_inner::<5>(self as u64, out)
    }
}
impl Decimal for u32 {
    #[inline]
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error> {
        fmt_u32_inner(self, out)
    }
}
impl Decimal for u64 {
    #[inline]
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error> {
        fmt_u64_inner(self, out)
    }
}
impl Decimal for u128 {
    #[inline]
    fn fmt_into(self, out: &mut [u8]) -> Result<usize, Error> {
        fmt_u128_inner(self, out)
    }
}

#[inline]
/// Generic decimal-to-ASCII entry point — accepts any unsigned integer width
///
/// Dispatches via the [`Decimal`] trait to the appropriate per-width
/// implementation: u8/u16 via `fmt_bounded_inner` with compile-time digit
/// bound, u32 via a hand-tuned `< 10^8` fast path, u64 via the IFMA
/// heterogeneous kernel, u128 via the chunk-split path. Returns the number
/// of bytes written
///
/// # Arguments
///
/// * `value` - Any value of type `u8`, `u16`, `u32`, `u64`, or `u128`
/// * `out` - Output byte slice; must hold at least `digit_count(value)` bytes
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] when `out` is too small
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut buf = [0u8; 40];
/// assert_eq!(vitoa::fmt(42u32,         &mut buf)?, 2);
/// assert_eq!(vitoa::fmt(1_234u64,      &mut buf)?, 4);
/// assert_eq!(vitoa::fmt(u128::MAX,     &mut buf)?, 39);
/// # Ok(()) }
/// ```
pub fn fmt<T: Decimal>(value: T, out: &mut [u8]) -> Result<usize, Error> {
    value.fmt_into(out)
}

#[inline]
/// Const-generic bounded-width decimal writer (crate-internal)
///
/// Provides the `MAX_DIGITS` hint so LLVM can dead-code-eliminate branches
/// that are impossible for smaller widths. Called by the u8/u16 trait impls;
/// not part of the public surface
pub(crate) fn fmt_bounded_inner<const MAX_DIGITS: u32>(
    value: u64,
    out: &mut [u8],
) -> Result<usize, Error> {
    let n = digit_count(value) as usize;
    if out.len() < n {
        return Err(Error::BufferTooSmall {
            need: n,
            have: out.len(),
        });
    }
    #[cfg(simd_ifma)]
    {
        Ok(unsafe { fmt_bounded_simd_inner::<MAX_DIGITS>(value, n, out.as_mut_ptr()) })
    }
    #[cfg(not(simd_ifma))]
    {
        scalar::scalar_to_chars(value, n, out);
        Ok(n)
    }
}

#[cfg(simd_ifma)]
#[inline]
/// Const-generic SIMD micro-kernel for bounded digit counts on AVX-512 IFMA hosts
///
/// Selects the minimum-work SIMD path at compile time:
/// - `MAX_DIGITS <= 8`: treat value as the low 8 digits of a 16-digit window
/// - `MAX_DIGITS <= 16`: single `to_string_16digits` call, no 17–20 branch
/// - Otherwise: full [`simd::heterogeneous`] path (handles up to 20 digits)
///
/// Dead branches are eliminated by LLVM because the `const { ... }` guards
/// are evaluated at monomorphisation time
///
/// # Arguments
///
/// * `value` - The `u64` value to convert
/// * `n` - Pre-computed digit count (must equal `digit_count(value)`)
/// * `ptr` - Pointer to the first byte of the output region
///
/// # Safety
///
/// Caller must have verified AVX-512 IFMA/VBMI/BW availability. `ptr` must be
/// valid for `n` byte writes. When `MAX_DIGITS <= 8` the masked store may begin
/// at `ptr + n - 16`; the allocation at `ptr` must be at least 16 bytes
unsafe fn fmt_bounded_simd_inner<const MAX_DIGITS: u32>(
    value: u64,
    n: usize,
    ptr: *mut u8,
) -> usize {
    use core::arch::x86_64::*;

    unsafe {
        if const { MAX_DIGITS <= 8 } {
            // Treat as the low half of a 16-digit window; mask drops leading zeros.
            let lo8 = simd::to_string_16digits(0, value);
            let mask: __mmask16 = (0xFFFFu16 << (16 - n)) as __mmask16;
            _mm_mask_storeu_epi8(ptr.wrapping_offset(n as isize - 16) as *mut i8, mask, lo8);
            return n;
        }

        if const { MAX_DIGITS <= 16 } {
            let digits_15_0 = simd::to_string_16digits(value / 100_000_000, value % 100_000_000);
            let mask: __mmask16 = (0xFFFFu16 << (16 - n)) as __mmask16;
            _mm_mask_storeu_epi8(
                ptr.wrapping_offset(n as isize - 16) as *mut i8,
                mask,
                digits_15_0,
            );
            return n;
        }

        simd::heterogeneous(value, n, ptr)
    }
}

/// Convert a `u128` value to decimal ASCII (1–39 digits)
///
/// Splits the value into 16-digit chunks per paper §5.4 generalised to u128:
/// - 1..=16 digits: delegate to [`fmt`] (cast to u64)
/// - 17..=32 digits: `hi = value / 10^16` (1–16 digits, u64) + `lo = value % 10^16` (16 digits, u64). Write `hi` variable-length via the existing 16-digit kernel + masked store, then `lo` as a fixed 16-byte unmasked store
/// - 33..=39 digits: split into three chunks `top` (1–7 digits) + `mid` (16) + `lo` (16). Write `top` via the 8-digit fast path, then `mid` and `lo` as fixed 16-byte unmasked stores
///
/// All `u128 / 10^16` divides go through [`crate::u128_div::div_rem_1e16`]
/// (Granlund-Montgomery multiplicative inverse), ~10 cycles each, vs ~50
/// cycles for the LLVM `__udivti3` libcall
///
/// # Arguments
///
/// * `value` - The `u128` value to convert
/// * `out` - Output byte slice; must have length >= `digit_count(value)`
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] when `out` cannot hold the result
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut buf = [0u8; 40];
/// let n = vitoa::fmt(u128::MAX, &mut buf)?;
/// assert_eq!(&buf[..n], b"340282366920938463463374607431768211455");
/// # Ok(())
/// # }
/// ```
pub(crate) fn fmt_u128_inner(value: u128, out: &mut [u8]) -> Result<usize, Error> {
    const E16: u128 = 10_000_000_000_000_000;
    // u128::ilog10 is ~10x slower than u64::ilog10 — delegate the 1..=16-digit
    // case cheaply via a single u128 < E16 compare, letting the u64 fast path
    // compute its own digit count.
    if value < E16 {
        return fmt(value as u64, out);
    }
    // d >= 17: derive total digit count from the cheap u64/u32 ilog10 of the
    // leading sub-chunk rather than calling the slow u128::ilog10 on the full value.
    let (q, lo) = u128_div::div_rem_1e16(value);

    #[cfg(simd_ifma)]
    unsafe {
        if q < E16 {
            // d=17..32: hi = q (1–16 digits, fits in u64), lo = bottom 16 digits.
            let hi = q as u64;
            let n_hi = digit_count(hi) as usize;
            let n = n_hi + 16;
            if out.len() < n {
                return Err(Error::BufferTooSmall {
                    need: n,
                    have: out.len(),
                });
            }
            let ptr = out.as_mut_ptr();
            let _ = fmt(hi, core::slice::from_raw_parts_mut(ptr, n_hi))?;
            simd::write_16_digits_unmasked(lo, ptr.add(n_hi));
            return Ok(n);
        }
        // d=33..39: top (1–7 digits, fits in u32) + mid (16) + lo (16).
        let (top128, mid) = u128_div::div_rem_1e16(q);
        let top = top128 as u32;
        let n_top = if top == 0 {
            1
        } else {
            (top.ilog10() + 1) as usize
        };
        let n = n_top + 32;
        if out.len() < n {
            return Err(Error::BufferTooSmall {
                need: n,
                have: out.len(),
            });
        }
        let ptr = out.as_mut_ptr();
        // SAFETY: top < 10^7 < 10^8.
        simd::u32_le_1e8(top, n_top, ptr);
        simd::write_16_digits_unmasked(mid, ptr.add(n_top));
        simd::write_16_digits_unmasked(lo, ptr.add(n_top + 16));
        Ok(n)
    }
    #[cfg(not(simd_ifma))]
    {
        // Scalar fallback: derive n without u128::ilog10.
        let n = if q < E16 {
            digit_count(q as u64) as usize + 16
        } else {
            let (top128, _) = u128_div::div_rem_1e16(q);
            let top = top128 as u32;
            let n_top = if top == 0 {
                1
            } else {
                (top.ilog10() + 1) as usize
            };
            n_top + 32
        };
        if out.len() < n {
            return Err(Error::BufferTooSmall {
                need: n,
                have: out.len(),
            });
        }
        let _ = lo; // already computed above; scalar writer takes the full value
        scalar_u128_to_chars(value, n, out);
        Ok(n)
    }
}

#[cfg(not(simd_ifma))]
#[inline]
/// Scalar fallback for u128 — 2-digit-lookup right-to-left into a 40-byte scratch
///
/// Mirrors [`scalar::scalar_to_chars`] but on u128. Only compiled when the
/// AVX-512 fast path is disabled at build time
///
/// # Arguments
///
/// * `value` - The u128 value to convert
/// * `n` - Pre-computed digit count
/// * `out` - Destination slice (length >= n verified by caller)
pub(crate) fn scalar_u128_to_chars(value: u128, n: usize, out: &mut [u8]) {
    use crate::digits::TWO_DIGIT;
    let mut scratch = [0u8; 40];
    let mut v = value;
    let mut i = 40usize;
    while v >= 100 {
        let idx = (v % 100) as usize;
        v /= 100;
        i -= 2;
        // SAFETY: idx < 100, i and i+1 < 40.
        unsafe {
            let pair = *TWO_DIGIT.get_unchecked(idx);
            *scratch.get_unchecked_mut(i) = pair[0];
            *scratch.get_unchecked_mut(i + 1) = pair[1];
        }
    }
    if v >= 10 {
        // SAFETY: v < 100, i and i+1 < 40.
        unsafe {
            let pair = *TWO_DIGIT.get_unchecked(v as usize);
            i -= 2;
            *scratch.get_unchecked_mut(i) = pair[0];
            *scratch.get_unchecked_mut(i + 1) = pair[1];
        }
    } else {
        i -= 1;
        // SAFETY: i < 40.
        unsafe { *scratch.get_unchecked_mut(i) = b'0' + v as u8 };
    }
    // SAFETY: 40 - i == n; out has at least n bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(scratch.as_ptr().add(i), out.as_mut_ptr(), n);
    }
}

#[inline]
/// Convert a `u32` value to decimal ASCII
///
/// Hand-tuned u32 dispatch: branches on `value < 10^8` to use a single
/// 8-digit IFMA kernel + VPMOVQB truncate when possible (1–8 digit values),
/// falling back to the full 16-digit heterogeneous kernel only for 9–10
/// digit values. On non-AVX-512 builds falls through to scalar
///
/// # Arguments
///
/// * `v` - The `u32` value to convert
/// * `out` - Output byte slice; must have length >= `digit_count(v as u64)`
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] when `out` is too small
pub(crate) fn fmt_u32_inner(v: u32, out: &mut [u8]) -> Result<usize, Error> {
    let n = if v == 0 { 1 } else { (v.ilog10() + 1) as usize };
    if out.len() < n {
        return Err(Error::BufferTooSmall {
            need: n,
            have: out.len(),
        });
    }
    #[cfg(simd_ifma)]
    unsafe {
        if v < 100_000_000u32 {
            simd::u32_le_1e8(v, n, out.as_mut_ptr());
        } else {
            simd::heterogeneous(v as u64, n, out.as_mut_ptr());
        }
    }
    #[cfg(not(simd_ifma))]
    {
        scalar::scalar_to_chars(v as u64, n, out);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(n: u64) {
        let mut buf = [0u8; 20];
        let len = fmt(n, &mut buf).unwrap();
        let expected = n.to_string();
        assert_eq!(
            std::str::from_utf8(&buf[..len]).unwrap(),
            expected,
            "wrong for n={n}"
        );
    }

    #[test]
    fn edges() {
        let vals = [
            0u64,
            1,
            9,
            10,
            99,
            100,
            999,
            1_000,
            9_999,
            10_000,
            99_999_999,
            100_000_000,
            9_999_999_999_999_999,
            10_000_000_000_000_000,
            99_999_999_999_999_999,
            100_000_000_000_000_000,
            1_000_000_000_000_000_000,
            9_999_999_999_999_999_999,
            10_000_000_000_000_000_000,
            u64::MAX,
        ];
        for n in vals {
            check(n);
        }
    }

    #[test]
    fn fuzz_lcg() {
        let mut s: u64 = 0xdeadbeef_cafef00d;
        for _ in 0..200_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            check(s);
        }
    }

    #[test]
    fn fuzz_all_lengths() {
        let mut p: u64 = 1;
        for _ in 0..19 {
            check(p);
            check(p.saturating_mul(10).saturating_sub(1));
            p = p.saturating_mul(10);
        }
        check(u64::MAX);
    }

    #[test]
    fn u128_edges_and_fuzz() {
        let edges: &[u128] = &[
            0,
            1,
            9,
            10,
            99,
            100,
            (1u128 << 63) - 1,
            1u128 << 63,
            u64::MAX as u128,
            (u64::MAX as u128) + 1,
            10u128.pow(15),
            10u128.pow(16) - 1,
            10u128.pow(16),
            10u128.pow(16) + 1,
            10u128.pow(31),
            10u128.pow(32) - 1,
            10u128.pow(32),
            10u128.pow(32) + 1,
            10u128.pow(38),
            u128::MAX,
        ];
        let mut buf = [0u8; 40];
        for &v in edges {
            let n = fmt(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..n]).unwrap(),
                expected,
                "fmt({v}) mismatch"
            );
        }
        // Random LCG sweep across the full u128 range.
        let mut s: u64 = 0xdead_beef_cafe_f00d;
        for _ in 0..50_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let s2 = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((s as u128) << 64) | s2 as u128;
            let n = fmt(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..n]).unwrap(),
                expected,
                "fmt({v}) mismatch"
            );
        }
    }

    #[test]
    fn buffer_too_small_returns_err() {
        let mut buf = [0u8; 1];
        let result = fmt(12345u64, &mut buf);
        assert!(matches!(
            result,
            Err(Error::BufferTooSmall { need: 5, have: 1 })
        ));
    }
}

#[cfg(test)]
mod bounded_tests {
    use super::*;

    /// Sweep full domain of u8, sample u16, sample u32
    #[test]
    fn u8_u16_u32() {
        let mut buf = [0u8; 20];

        // u8: full domain
        for v in 0u8..=u8::MAX {
            let len = fmt(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..len]).unwrap(),
                expected,
                "fmt({v})"
            );
        }

        // u16: full domain
        for v in 0u16..=u16::MAX {
            let len = fmt(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..len]).unwrap(),
                expected,
                "fmt({v})"
            );
        }

        // u32: LCG sample
        let mut lcg: u64 = 0xdead_cafe;
        for _ in 0..100_000u32 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = lcg as u32;
            let len = fmt(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..len]).unwrap(),
                expected,
                "fmt({v})"
            );
        }
    }

    /// `fmt_bounded_inner::<8>` for value < 10^8, full random sweep
    #[test]
    fn const_bound_8() {
        let mut buf = [0u8; 20];
        let mut lcg: u64 = 0xabcd_1234_5678_9abc;
        for _ in 0..100_000u32 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = lcg % 100_000_000; // < 10^8
            let len = fmt_bounded_inner::<8>(v, &mut buf).unwrap();
            let expected = v.to_string();
            assert_eq!(
                std::str::from_utf8(&buf[..len]).unwrap(),
                expected,
                "fmt_bounded_inner::<8>({v})"
            );
        }
    }
}
