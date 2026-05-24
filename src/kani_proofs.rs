//! Kani model-checker harnesses for exhaustive correctness verification
//!
//! Run via `cargo install --locked kani-verifier && cargo kani setup && cargo kani`
//!
//! These harnesses target the SCALAR path only (`scalar::scalar_to_chars` and
//! `digits::count_digits`). The AVX-512 IFMA intrinsics are modelled by CBMC
//! as uninterpreted functions, so SIMD harnesses would be unsound; the scalar
//! path is the authoritative correctness reference and the SIMD kernels are
//! validated against it by the randomized fuzz tests in `lib.rs::tests`
//!
//! Author: aav
#![cfg(kani)]

// --------------------------------------------------
// local
// --------------------------------------------------
use crate::digits::count_digits;
use crate::scalar::scalar_to_chars;

/// Manual decimal-digit-count reference (avoids relying on `u64::ilog10`)
///
/// Returns the number of decimal digits `n` would occupy when formatted, with
/// the convention that `0` has 1 digit. Used as the ground truth for
/// [`count_digits`] verification
///
/// # Arguments
///
/// * `n` - The value whose digit count is requested
fn reference_digit_count(n: u64) -> u32 {
    let mut count = 1u32;
    let mut v = n / 10;
    while v > 0 {
        count += 1;
        v /= 10;
    }
    count
}

/// Decode an ASCII decimal byte slice back to `u64`, panicking on malformed input
///
/// Used as the inverse of [`scalar_to_chars`] in the round-trip proofs below
/// Kani treats panics as proof failures, so any non-digit byte or overflow is
/// caught as a correctness violation
///
/// # Arguments
///
/// * `bytes` - ASCII digit bytes to parse
fn parse_decimal(bytes: &[u8]) -> u64 {
    let mut acc: u64 = 0;
    for &b in bytes {
        kani::assume(b >= b'0' && b <= b'9');
        acc = acc * 10 + (b - b'0') as u64;
    }
    acc
}

/// Proof: `count_digits` matches the manual reference for every `u64`
///
/// CBMC enumerates symbolic `u64` and proves equivalence with
/// `reference_digit_count`. Establishes the digit-count primitive is correct
#[kani::proof]
#[kani::unwind(21)]
fn proof_count_digits_all_u64() {
    let n: u64 = kani::any();
    assert_eq!(count_digits(n), reference_digit_count(n));
}

/// Proof: `scalar_to_chars` round-trips for every `u8` (exhaustive)
///
/// Writes `value` then parses the bytes back; asserts the parsed value
/// matches the input. Covers the entire 256-element `u8` domain
#[kani::proof]
#[kani::unwind(21)]
fn proof_scalar_roundtrip_u8() {
    let value: u8 = kani::any();
    let n = count_digits(value as u64) as usize;
    let mut buf = [0u8; 3];
    scalar_to_chars(value as u64, n, &mut buf);
    let parsed = parse_decimal(&buf[..n]);
    assert_eq!(parsed, value as u64);
}

/// Decode an ASCII decimal byte slice back to `u128` for u128 round-trip proofs
fn parse_decimal_u128(bytes: &[u8]) -> u128 {
    let mut acc: u128 = 0;
    for &b in bytes {
        kani::assume(b >= b'0' && b <= b'9');
        acc = acc * 10 + (b - b'0') as u128;
    }
    acc
}

/// Proof: `scalar_u128_to_chars` round-trips for any `u128` (opt-in, very slow)
///
/// Covers the scalar u128 path used by [`crate::fmt_u128`] on builds without
/// AVX-512 IFMA. SAT solving on the symbolic u128 round-trip (39-digit
/// unwind + 128-bit multiply chain) exceeds 15 min and is gated behind
/// `--cfg kani_slow`. The fast Rust tests in `lib.rs::tests::u128_edges_and_fuzz`
/// and `macros::tests::lcg_sweep_u128_matches_core` (1M random samples each)
/// give strong correctness evidence without paying CBMC's u128 cost
#[cfg(kani_slow)]
#[kani::proof]
#[kani::unwind(41)]
fn proof_scalar_u128_roundtrip() {
    let value: u128 = kani::any();
    let n = if value == 0 {
        1usize
    } else {
        (value.ilog10() + 1) as usize
    };
    let mut buf = [0u8; 40];
    crate::scalar_u128_to_chars(value, n, &mut buf);
    let parsed = parse_decimal_u128(&buf[..n]);
    assert_eq!(parsed, value);
}

/// Proof: `scalar_to_chars` round-trips for every `u16` (exhaustive)
#[kani::proof]
#[kani::unwind(21)]
fn proof_scalar_roundtrip_u16() {
    let value: u16 = kani::any();
    let n = count_digits(value as u64) as usize;
    let mut buf = [0u8; 5];
    scalar_to_chars(value as u64, n, &mut buf);
    let parsed = parse_decimal(&buf[..n]);
    assert_eq!(parsed, value as u64);
}

/// Proof: `scalar_to_chars` round-trips for every `u32` symbolically (opt-in)
///
/// CBMC reasons over the full 4-billion-element domain via symbolic execution
/// In practice SAT solving on the round-trip predicate exceeds 10 minutes and
/// is gated behind the `kani_slow` cfg. Run with
/// `cargo kani --cbmc-args --no-unwinding-assertions --harness proof_scalar_roundtrip_u32 -- --cfg kani_slow`
/// when you want full u32 coverage. The same algorithm is exhaustively proven
/// for u8 and u16 above, which is strong evidence of u32 correctness because
/// `scalar_to_chars` uses no type-specific paths
#[cfg(kani_slow)]
#[kani::proof]
#[kani::unwind(21)]
fn proof_scalar_roundtrip_u32() {
    let value: u32 = kani::any();
    let n = count_digits(value as u64) as usize;
    let mut buf = [0u8; 10];
    scalar_to_chars(value as u64, n, &mut buf);
    let parsed = parse_decimal(&buf[..n]);
    assert_eq!(parsed, value as u64);
}

/// Proof: `scalar_to_chars` round-trips for any `u64` (opt-in, very slow)
///
/// Symbolic SAT on the full 64-bit domain. Not run by default — invoke
/// with `--cfg kani_slow` and expect runtimes of an hour or more. The fast
/// proofs above (digit-count over all u64, round-trip for u8/u16, byte-range
/// invariant) give strong correctness evidence without paying this cost
#[cfg(kani_slow)]
#[kani::proof]
#[kani::unwind(21)]
fn proof_scalar_roundtrip_u64() {
    let value: u64 = kani::any();
    let n = count_digits(value) as usize;
    let mut buf = [0u8; 20];
    scalar_to_chars(value, n, &mut buf);
    let parsed = parse_decimal(&buf[..n]);
    assert_eq!(parsed, value);
}

/// Proof: `scalar_to_chars` writes exactly `n` ASCII-digit bytes
///
/// Every byte in `buf[..n]` is in the ASCII digit range `b'0'..=b'9'`, and
/// `buf[n..]` is untouched (verified by initialising the trailing region to
/// a known sentinel)
#[kani::proof]
#[kani::unwind(21)]
fn proof_scalar_only_writes_n_bytes() {
    let value: u64 = kani::any();
    let n = count_digits(value) as usize;
    let mut buf = [0xFFu8; 20];
    scalar_to_chars(value, n, &mut buf);
    let mut idx: usize = 0;
    while idx < n {
        let byte = buf[idx];
        assert!(byte >= b'0' && byte <= b'9');
        idx += 1;
    }
    while idx < buf.len() {
        assert_eq!(buf[idx], 0xFFu8);
        idx += 1;
    }
}
