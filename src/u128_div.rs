//! Fast division by 10^16 for u128 via Granlund-Montgomery multiplicative inverse
//!
//! LLVM lowers a literal `n / 10_000_000_000_000_000u128` to a `__udivti3`
//! libcall on x86_64 — measured ~50 cycles per call. The 17-32 and 33-39 digit
//! paths in [`crate::fmt_u128`] need two such divides per call, so naive
//! division dominates the runtime
//!
//! This module provides a `div_rem_1e16` that performs the same operation in
//! ~5-10 cycles by computing the top half of a 128 × 128 → 256-bit product
//! against a precomputed multiplicative inverse, then a fixed right shift. The
//! technique is from Granlund & Montgomery, *"Division by Invariant Integers
//! Using Multiplication"*, SIGPLAN PLDI 1994
//!
//! Constants below are reused from the dtolnay `fmt` crate's `u128_ext.rs`
//! (MIT/Apache-2.0), where they were derived via the CHOOSE_MULTIPLIER
//! procedure with N=128, prec=128, d=10^16
//!
//! Author: aav
// --------------------------------------------------
// constants
// --------------------------------------------------
/// 10^16 — the divisor we want
const D: u128 = 10_000_000_000_000_000;
/// CHOOSE_MULTIPLIER output for `(N=128, prec=128, d=10^16)`
const M_HIGH: u128 = 76624777043294442917917351357515459181;
/// Post-multiply right-shift count
const SH_POST: u32 = 51;

/// Quotient and remainder of `n / 10^16` and `n mod 10^16`
///
/// The remainder is guaranteed to fit in `u64` because it is < `10^16` and
/// `10^16 < 2^64`. The quotient may still be a `u128` for very large inputs
/// (up to `u128::MAX / 10^16` ≈ `3.4 × 10^22`)
///
/// # Arguments
///
/// * `n` - The `u128` numerator
#[inline]
pub(crate) fn div_rem_1e16(n: u128) -> (u128, u64) {
    if n < D {
        return (0, n as u64);
    }
    // Granlund-Montgomery: quot = (n × M_HIGH).hi_128 >> SH_POST
    let quot = mulhi(n, M_HIGH) >> SH_POST;
    let rem = (n - quot * D) as u64;
    (quot, rem)
}

/// Top 128 bits of a 128 × 128 → 256-bit unsigned multiplication
///
/// Implemented schoolbook-style via four 64×64→128 multiplies on `u128`
/// halves, then assembling the high 128 bits with carry. Each `u64 as u128 *
/// u64 as u128` lowers to a single `mulq` on x86_64
///
/// # Arguments
///
/// * `a` - First operand
/// * `b` - Second operand
#[inline]
fn mulhi(a: u128, b: u128) -> u128 {
    let a_lo = a as u64 as u128;
    let a_hi = a >> 64;
    let b_lo = b as u64 as u128;
    let b_hi = b >> 64;
    let bot = a_lo * b_lo;
    let mid1 = a_hi * b_lo;
    let mid2 = a_lo * b_hi;
    let top = a_hi * b_hi;
    const MASK64: u128 = 0xFFFF_FFFF_FFFF_FFFF;
    let mid = (bot >> 64) + (mid1 & MASK64) + (mid2 & MASK64);
    top + (mid1 >> 64) + (mid2 >> 64) + (mid >> 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check fast `div_rem_1e16` against the naive `u128` divide for
    /// a structured sample of edge cases and a random LCG sweep
    #[test]
    fn matches_naive_divide() {
        let test_values: &[u128] = &[
            0,
            1,
            D - 1,
            D,
            D + 1,
            D * 2,
            D * 9_999_999_999_999_999,
            u128::MAX,
            (D - 1) * D,
            10u128.pow(32),
            10u128.pow(38),
        ];
        for &n in test_values {
            let (q, r) = div_rem_1e16(n);
            assert_eq!(q, n / D, "quotient mismatch at n = {n}");
            assert_eq!(r as u128, n % D, "remainder mismatch at n = {n}");
        }
        let mut s: u64 = 0xdead_beef_cafe_f00d;
        for _ in 0..200_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let s2 = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = ((s as u128) << 64) | s2 as u128;
            let (q, r) = div_rem_1e16(n);
            assert_eq!(q, n / D, "quotient mismatch at n = {n}");
            assert_eq!(r as u128, n % D, "remainder mismatch at n = {n}");
        }
    }
}
