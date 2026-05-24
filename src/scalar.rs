//! Scalar (non-SIMD) decimal conversion via a two-digit lookup table
//!
//! Used as the universal fallback on non-x86-64 hosts and on x86-64 hosts
//! where AVX-512 IFMA/VBMI/BW is absent at runtime. Correctness is
//! unconditional; throughput is lower than the SIMD paths but competitive
//! with standard-library `to_string`
//!
//! Author: aav
// --------------------------------------------------
// local
// --------------------------------------------------
use crate::digits::TWO_DIGIT;

#[allow(dead_code)] // used only when AVX-512 IFMA target features are NOT enabled
/// Classic two-digit-lookup right-to-left decimal writer
///
/// Iterates from the least-significant digits toward the most-significant,
/// consuming two digits per iteration from the [`TWO_DIGIT`] table. The
/// remaining 1 or 2 most-significant digits are handled after the loop
/// Always writes exactly `n` bytes into `out[0..n]`
///
/// # Arguments
///
/// * `value` - The `u64` value to convert
/// * `n` - The pre-computed decimal digit count; must equal `digit_count(value)`
/// * `out` - Output byte slice; must have length >= `n`
pub(crate) fn scalar_to_chars(value: u64, n: usize, out: &mut [u8]) {
    let mut v = value;
    let mut i = n;
    while v >= 100 {
        let idx = (v % 100) as usize;
        // SAFETY: idx = v % 100 < 100 == TWO_DIGIT.len().
        let pair = unsafe { *TWO_DIGIT.get_unchecked(idx) };
        v /= 100;
        i -= 2;
        // SAFETY: caller guarantees out.len() >= n; i starts at n and decrements
        // by 2 each iteration, so i and i+1 are always < n <= out.len().
        unsafe {
            *out.get_unchecked_mut(i) = pair[0];
            *out.get_unchecked_mut(i + 1) = pair[1];
        }
    }
    if v >= 10 {
        // SAFETY: v < 100 and v >= 10, so v < TWO_DIGIT.len().
        let pair = unsafe { *TWO_DIGIT.get_unchecked(v as usize) };
        // SAFETY: after the loop i == 0 or 1; both < n <= out.len().
        unsafe {
            *out.get_unchecked_mut(0) = pair[0];
            *out.get_unchecked_mut(1) = pair[1];
        }
    } else {
        // SAFETY: out.len() >= n >= 1.
        unsafe { *out.get_unchecked_mut(0) = b'0' + v as u8 };
    }
}
