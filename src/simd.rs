//! AVX-512 IFMA/VBMI/BW SIMD kernels for fast decimal conversion
//!
//! Implements the two SIMD paths described in the paper:
//! - §5.4, Figure 6: heterogeneous masked-store path (handles all digit lengths)
//! - §5.5, Figure 7: homogeneous unmasked-store path (17–20 digit values only)
//!
//! All public items in this module are `pub(crate)` — they are consumed by
//! [`crate::dispatch_simd`], [`crate::fmt_bounded_simd`], and the batch driver
//! in [`crate::batch`]. None are part of the public API
//!
//! Author: aav
// --------------------------------------------------
// local
// --------------------------------------------------
use crate::digits::{write_four_digits_10000, write_one_two_three_or_four_digits_10000};
// --------------------------------------------------
// external
// --------------------------------------------------
use core::arch::x86_64::*;

#[repr(align(64))]
/// 8-lane i64 vector aligned to a 64-byte cache line
///
/// Used to hold the IFMA coefficient table [`IFMA_C`] and the permutation
/// index table [`PERM_INDEX`] so that `_mm512_load_si512` can be used
/// (which requires 64-byte alignment and avoids an alignment-fault on older
/// microarchitectures)
struct AlignedI64x8([i64; 8]);

/// IFMA scaling coefficients `c_k = floor(2^52 / 10^k)` for k = 1..=8
///
/// Lane 0 holds `c_8` (coefficient for the most significant digit of an
/// 8-digit chunk); lane 7 holds `c_1`. These constants are used inside
/// [`to_string_8digits`] with `_mm512_madd52lo_epu64` / `_mm512_madd52hi_epu64`
/// to extract individual decimal digits via integer multiply-high
static IFMA_C: AlignedI64x8 = AlignedI64x8([
    45_035_996,          // c_8
    450_359_962,         // c_7
    4_503_599_627,       // c_6
    45_035_996_273,      // c_5
    450_359_962_737,     // c_4
    4_503_599_627_370,   // c_3
    45_035_996_273_704,  // c_2
    450_359_962_737_049, // c_1
]);

#[repr(align(64))]
/// 64-element i8 vector aligned to a 64-byte cache line
///
/// Holds the byte-permutation control table [`PERM_INDEX`]
struct AlignedI8x64([i8; 64]);

/// Permutation control for `_mm512_permutex2var_epi8`
///
/// Gathers byte 0 of each 64-bit lane from the high 512-bit register (A)
/// into output positions 0..7, then byte 0 of each lane from the low
/// register (B) into positions 8..15. The result packs the 8 most-significant
/// ASCII digits followed by the 8 least-significant into a contiguous 128-bit
/// value
static PERM_INDEX: AlignedI8x64 = {
    let mut a = [0i8; 64];
    let mut i = 0;
    // `get_unchecked` is not stable in const context; suppress the lint instead.
    // SAFETY: i < 8 by the while guard, so i < 8 and i + 8 < 16, both < 64.
    #[allow(clippy::indexing_slicing)]
    while i < 8 {
        a[i] = (i * 8) as i8; // from A (hi)
        a[i + 8] = (64 + i * 8) as i8; // from B (lo)
        i += 1;
    }
    AlignedI8x64(a)
};

#[target_feature(enable = "avx512f,avx512ifma")]
#[inline]
/// Compute 8 ASCII digit characters for `n` (n < 10^8)
///
/// Broadcasts `n` into all 8 lanes of a 512-bit register, multiplies by the
/// precomputed IFMA coefficients, and extracts the high 52 bits of each
/// product to isolate individual decimal digits. Each lane's low byte holds
/// one ASCII character; lane 0 is the most significant digit, lane 7 is the
/// least significant
///
/// # Arguments
///
/// * `n` - Value in `[0, 10^8)` to decompose into 8 ASCII digit lanes
///
/// # Safety
///
/// Caller must ensure `avx512f` and `avx512ifma` are available at runtime
/// `n` must be less than 100,000,000
pub(crate) unsafe fn to_string_8digits(n: u64) -> __m512i {
    unsafe {
        let vn = _mm512_set1_epi64(n as i64);
        let c = _mm512_load_si512(IFMA_C.0.as_ptr() as *const _);
        let vten = _mm512_set1_epi64(10);
        let vzero = _mm512_set1_epi64(b'0' as i64);
        let low = _mm512_madd52lo_epu64(c, vn, c);
        _mm512_madd52hi_epu64(vzero, vten, low)
    }
}

#[target_feature(enable = "avx512f,avx512ifma,avx512vbmi")]
#[inline]
/// Build 16 contiguous ASCII decimal digits in a 128-bit register
///
/// Converts the high 8-digit chunk (`hi8`) and low 8-digit chunk (`lo8`)
/// to ASCII via [`to_string_8digits`], then uses `_mm512_permutex2var_epi8`
/// to interleave the low bytes of each lane into a packed 128-bit result
/// The output layout is `digits[0..8]` from `hi8` followed by
/// `digits[8..16]` from `lo8`
///
/// # Arguments
///
/// * `hi8` - High 8-digit chunk; must be in `[0, 10^8)`
/// * `lo8` - Low 8-digit chunk; must be in `[0, 10^8)`
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512ifma`, and `avx512vbmi` are available
pub(crate) unsafe fn to_string_16digits(hi8: u64, lo8: u64) -> __m128i {
    unsafe {
        let hi = to_string_8digits(hi8);
        let lo = to_string_8digits(lo8);
        let idx = _mm512_load_si512(PERM_INDEX.0.as_ptr() as *const _);
        let perm = _mm512_permutex2var_epi8(hi, idx, lo);
        _mm512_castsi512_si128(perm)
    }
}

#[target_feature(enable = "avx512f,avx512ifma,avx512bw")]
#[inline]
/// 8-digit specialised conversion: 1–8 digits via one IFMA kernel + VPMOVQB
///
/// Cheaper than [`heterogeneous`] when the value fits in 8 digits because
/// only one IFMA kernel runs and the permute is replaced by a single
/// `_mm512_cvtepi64_epi8` (VPMOVQB) instruction that truncates each 64-bit
/// lane's low byte into a packed 8-byte vector
///
/// # Arguments
///
/// * `value` - Value in `[0, 10^8)` to convert
/// * `n` - Decimal digit count of `value`, must be in `[1, 8]`
/// * `result` - Pointer to the first output byte; must be valid for `n` writes,
///   and the masked store begins at `result + n - 8` so the allocation must
///   extend at least 8 bytes from that point
///
/// # Safety
///
/// Caller must ensure AVX-512 F/IFMA/BW are available. `value` < 10^8
/// `n` must equal `digit_count(value)`. `result` must satisfy the bounds above
pub(crate) unsafe fn u32_le_1e8(value: u32, n: usize, result: *mut u8) {
    unsafe {
        let lanes = to_string_8digits(value as u64);
        // VPMOVQB: truncate each 64-bit lane to its low byte.
        let bytes = _mm512_cvtepi64_epi8(lanes);
        let mask: __mmask16 = (0xFFu16 << (8 - n)) as __mmask16;
        // NOTE: wrapping_offset avoids pointer-provenance UB when n < 8 and the
        // buffer is exactly n bytes; the masked store only writes
        // [result..result+n), which are in bounds. The pre-`result` address is
        // never read or written.
        _mm_mask_storeu_epi8(
            result.wrapping_offset(n as isize - 8) as *mut i8,
            mask,
            bytes,
        );
    }
}

#[target_feature(enable = "avx512f,avx512ifma,avx512vbmi,avx512bw")]
#[inline]
/// Paper Figure 6 (heterogeneous masked-store path): convert any u64 to decimal
///
/// Handles values with 1–16 digits via a single masked 16-byte store; values
/// with 17–20 digits are split into a 16-digit SIMD chunk and a 1–4-digit
/// scalar prefix written by [`write_four_digits_10000`]. Returns `n`
///
/// # Arguments
///
/// * `value` - The `u64` value to convert
/// * `n` - Pre-computed decimal digit count (must equal `digit_count(value)`)
/// * `result` - Pointer to the first output byte
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA/VBMI/BW are available. `result` must be
/// valid for `n` bytes of write. The masked store begins at
/// `result + n - 16`; the allocation must extend at least 16 bytes from
/// that point (i.e., the caller must have at least `max(n, 16)` bytes
/// available from `result`)
pub(crate) unsafe fn heterogeneous(value: u64, n: usize, result: *mut u8) -> usize {
    unsafe {
        if value < 10_000_000_000_000_000u64 {
            let digits_15_0 = to_string_16digits(value / 100_000_000, value % 100_000_000);
            let mask: __mmask16 = (0xFFFFu16 << (16 - n)) as __mmask16;
            // NOTE: wrapping_offset prevents pointer-provenance UB when n < 16
            // (e.g. a single-digit value with a tight 1-byte buffer). The masked
            // store hardware writes only [result..result+n), in bounds.
            _mm_mask_storeu_epi8(
                result.wrapping_offset(n as isize - 16) as *mut i8,
                mask,
                digits_15_0,
            );
            return n;
        }
        // 17–20 digit path: split the bottom 4 digits for scalar write and
        // route the top 16 through SIMD.
        let q = value / 10_000;
        let r = (value % 10_000) as u32;
        let nq = n - 4;
        let v16 = to_string_16digits(q / 100_000_000, q % 100_000_000);
        let mask: __mmask16 = (0xFFFFu16 << (16 - nq)) as __mmask16;
        // NOTE: see wrapping_offset rationale above. nq >= 13 for the 17–20
        // digit path, so this is in-bounds; the form keeps soundness independent
        // of the n value.
        _mm_mask_storeu_epi8(
            result.wrapping_offset(nq as isize - 16) as *mut i8,
            mask,
            v16,
        );
        write_four_digits_10000(result.add(nq), r);
        n
    }
}

#[target_feature(enable = "avx512f,avx512ifma,avx512vbmi,avx512bw")]
#[inline]
/// Write exactly 16 ASCII digits for `value < 10^16` to `ptr` (unmasked)
///
/// Stores the full 16-byte SIMD vector — including any leading-zero digits —
/// via `_mm_storeu_si128`. Used by [`crate::fmt_u128`] to emit fixed-width
/// 16-digit mid/low chunks of a u128 in O(1) without per-call masking
///
/// # Arguments
///
/// * `value` - The value to convert; must be `< 10^16`
/// * `ptr` - Pointer to the first of 16 output bytes
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA/VBMI/BW are available. `value` < 10^16
/// (so the 16-digit representation including leading zeros fits). `ptr` must
/// be valid for 16 sequential byte writes
pub(crate) unsafe fn write_16_digits_unmasked(value: u64, ptr: *mut u8) {
    unsafe {
        let digits = to_string_16digits(value / 100_000_000, value % 100_000_000);
        _mm_storeu_si128(ptr as *mut __m128i, digits);
    }
}

#[target_feature(enable = "avx512f,avx512ifma,avx512vbmi,avx512bw")]
#[inline]
/// Paper Figure 7 (homogeneous unmasked-store path): convert values with 17–20 digits
///
/// Optimised for batches where the dominant digit length is in [17, 20]. Uses
/// an unmasked `_mm_storeu_si128` for the 16-byte suffix instead of the
/// masked store used by [`heterogeneous`], which removes the mask-store latency
/// at the cost of requiring the output buffer to have at least `prefix_len + 16`
/// bytes available from `result`
///
/// # Arguments
///
/// * `value` - The `u64` value to convert; must be >= 10^16
/// * `n` - Pre-computed decimal digit count; must be in [17, 20]
/// * `result` - Pointer to the first output byte
///
/// # Safety
///
/// - Caller must ensure AVX-512 IFMA/VBMI/BW are available
/// - `value` must be >= 10^16 (n ∈ [17, 20])
/// - `result` must be valid for `n` bytes of write, AND the 16-byte store
///   that begins at `result + prefix_len` must not overrun a live allocation
///   boundary (caller must have allocated at least `prefix_len + 16` bytes
///   at `result`). In practice the batch driver sizes its output buffer to
///   20 bytes per slot, so this is always satisfied
pub(crate) unsafe fn homogeneous_17_20(value: u64, n: usize, result: *mut u8) -> usize {
    unsafe {
        let q = value / 10_000_000_000_000_000; // 1..=1844 (1–4 digits)
        let r_lo16 = value % 10_000_000_000_000_000;
        let prefix_len = write_one_two_three_or_four_digits_10000(result, q as u32);
        let digits_15_0 = to_string_16digits(r_lo16 / 100_000_000, r_lo16 % 100_000_000);
        // Unmasked 128-bit store: batch driver guarantees >= 20 bytes per slot.
        _mm_storeu_si128(result.add(prefix_len) as *mut __m128i, digits_15_0);
        n
    }
}
