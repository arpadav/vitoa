//! Batch decimal conversion API with dynamic SIMD-variant selection
//!
//! Provides three public entry points:
//! - [`fmt_batch`]: convert a slice of `u64` values into a packed byte buffer
//!   with a companion offset array, using the default [`BatchConfig`]
//! - [`fmt_batch_with`]: same as above but accepts a custom [`BatchConfig`]
//! - [`fmt_batch_joined`]: convert and join values with a separator byte
//!
//! Author: aav
// --------------------------------------------------
// local
// --------------------------------------------------
use crate::digits::count_digits;
use crate::error::Error;
use crate::selector::{BatchConfig, Variant, select};
use crate::{dispatch_simd, fmt};

/// Convert a batch of `u64` values to decimal ASCII with offset tracking
///
/// Writes each value's decimal representation sequentially into `out`,
/// recording start offsets in `offsets`. `offsets` must have length
/// `values.len() + 1`; on return `offsets[i]` is the start byte of element
/// `i` and `offsets[values.len()]` is the total bytes written
///
/// Returns `Ok(total_bytes_written)` on success
///
/// Uses the default [`BatchConfig`] for variant selection (1% stride sample,
/// 95% homogeneity threshold). For custom tuning use [`fmt_batch_with`]
///
/// # Arguments
///
/// * `values` - Slice of `u64` values to convert
/// * `out` - Output byte buffer; must be large enough for all values
/// * `offsets` - Offset array of length `values.len() + 1`
///
/// # Errors
///
/// Returns [`Error::OffsetsLengthMismatch`] if `offsets.len() != values.len() + 1`,
/// or [`Error::BufferTooSmall`] if `out` is too small
pub fn fmt_batch(values: &[u64], out: &mut [u8], offsets: &mut [u32]) -> Result<usize, Error> {
    fmt_batch_with(values, out, offsets, BatchConfig::default())
}

/// Convert a batch of `u64` values to decimal ASCII with a custom [`BatchConfig`]
///
/// Two-pass algorithm:
/// 1. Compute per-element digit lengths, populate start offsets via prefix sum,
///    and determine the total byte count
/// 2. Sample the length histogram via [`select`], then emit each value through
///    the chosen SIMD or scalar path
///
/// Returns `Ok(total_bytes_written)` on success
///
/// # Arguments
///
/// * `values` - Slice of `u64` values to convert
/// * `out` - Output byte buffer
/// * `offsets` - Offset array of length `values.len() + 1`
/// * `cfg` - Tuning parameters for variant selection
///
/// # Errors
///
/// Returns [`Error::OffsetsLengthMismatch`] if `offsets.len() != values.len() + 1`,
/// or [`Error::BufferTooSmall`] if `out` is too small
pub fn fmt_batch_with(
    values: &[u64],
    out: &mut [u8],
    offsets: &mut [u32],
    cfg: BatchConfig,
) -> Result<usize, Error> {
    let expected_offsets = values.len() + 1;
    if offsets.len() != expected_offsets {
        return Err(Error::OffsetsLengthMismatch {
            expected: expected_offsets,
            got: offsets.len(),
        });
    }
    if values.is_empty() {
        // SAFETY: expected_offsets == 1 and we verified offsets.len() == 1 above.
        unsafe { *offsets.get_unchecked_mut(0) = 0 };
        return Ok(0);
    }
    // Pass 1: compute per-element digit lengths and build the prefix-sum offset table.
    let mut lengths = Vec::with_capacity(values.len());
    let mut total: u32 = 0;
    for (i, &v) in values.iter().enumerate() {
        let len = count_digits(v);
        lengths.push(len as u8);
        // SAFETY: i < values.len() < offsets.len() (verified above).
        unsafe { *offsets.get_unchecked_mut(i) = total };
        // checked_add prevents silent u32 wrap on huge batches (would lead to
        // an undersized total_bytes and OOB writes in pass 2).
        total = total.checked_add(len).ok_or(Error::TotalLengthOverflow)?;
    }
    // SAFETY: values.len() < offsets.len() verified above.
    unsafe { *offsets.get_unchecked_mut(values.len()) = total };
    let total_bytes = total as usize;
    if out.len() < total_bytes {
        return Err(Error::BufferTooSmall {
            need: total_bytes,
            have: out.len(),
        });
    }
    let variant = select(&lengths, cfg);
    // Pass 2: emit each value via the chosen path.
    let out_ptr = out.as_mut_ptr();

    match variant {
        Variant::Heterogeneous => {
            for (i, &v) in values.iter().enumerate() {
                // SAFETY: i < offsets.len() (offsets has values.len()+1 elements).
                let start = unsafe { *offsets.get_unchecked(i) } as usize;
                // SAFETY: i < lengths.len() == values.len().
                let n = unsafe { *lengths.get_unchecked(i) } as usize;
                // SAFETY: `start + n <= total_bytes <= out.len()`, verified above.
                unsafe {
                    dispatch_simd(v, n, out_ptr.add(start));
                }
            }
        }
        Variant::Homogeneous17_20 => {
            #[cfg(simd_ifma)]
            {
                for (i, &v) in values.iter().enumerate() {
                    // SAFETY: i < offsets.len().
                    let start = unsafe { *offsets.get_unchecked(i) } as usize;
                    // SAFETY: i < lengths.len().
                    let n = unsafe { *lengths.get_unchecked(i) } as usize;
                    // SAFETY: `start + n <= total_bytes <= out.len()`. The unmasked
                    // 16-byte store writes at most `prefix_len + 16` bytes; since each
                    // slot is >= 17 bytes (dominant length >= 17) the store stays in slot.
                    unsafe {
                        crate::simd::homogeneous_17_20(v, n, out_ptr.add(start));
                    }
                }
            }
            #[cfg(not(simd_ifma))]
            {
                // No AVX-512 IFMA at build time — fall through to scalar.
                for (i, &v) in values.iter().enumerate() {
                    // SAFETY: i < offsets.len() and i < lengths.len().
                    let start = unsafe { *offsets.get_unchecked(i) } as usize;
                    let n = unsafe { *lengths.get_unchecked(i) } as usize;
                    unsafe {
                        dispatch_simd(v, n, out_ptr.add(start));
                    }
                }
            }
        }
    }

    Ok(total_bytes)
}

/// Write `values` separated by `sep` into `out` and return `Ok(total_bytes_written)`
///
/// Output format: `v0 sep v1 sep v2 ... sep v_{N-1}` with no trailing
/// separator. Each value is converted via [`fmt`]
///
/// # Arguments
///
/// * `values` - Slice of `u64` values to convert and join
/// * `sep` - Separator byte inserted between adjacent values (e.g. `b','`)
/// * `out` - Output byte slice; must be large enough for all digits plus
///   `values.len() - 1` separator bytes
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] if `out` is too small
pub fn fmt_batch_joined(values: &[u64], sep: u8, out: &mut [u8]) -> Result<usize, Error> {
    if values.is_empty() {
        return Ok(0);
    }
    let mut offset = 0usize;
    let last = values.len() - 1;
    for (idx, &v) in values.iter().enumerate() {
        let have = out.len();
        if offset > have {
            return Err(Error::BufferTooSmall {
                need: offset + 1,
                have,
            });
        }
        // SAFETY: offset <= out.len() checked immediately above.
        let n = fmt(v, unsafe { out.get_unchecked_mut(offset..) })?;
        offset += n;
        if idx < last {
            let have = out.len();
            if offset >= have {
                return Err(Error::BufferTooSmall {
                    need: offset + 1,
                    have,
                });
            }
            // SAFETY: bounds checked immediately above.
            unsafe { *out.get_unchecked_mut(offset) = sep };
            offset += 1;
        }
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digit_count;

    /// `offsets[i+1] - offsets[i]` must equal `digit_count(values[i])` for all i
    #[test]
    fn offsets_consistency() {
        let mut lcg: u64 = 0xdeadbeef_cafef00d;
        let n = 1000usize;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            values.push(lcg);
        }

        let mut offsets = vec![0u32; n + 1];
        let mut out = vec![0u8; 20 * n];
        fmt_batch(&values, &mut out, &mut offsets).unwrap();

        for (i, &v) in values.iter().enumerate() {
            let got = offsets[i + 1] - offsets[i];
            let expected = digit_count(v);
            assert_eq!(
                got, expected,
                "offsets mismatch for values[{i}] = {v}: got {got}, expected {expected}"
            );
        }
    }

    /// Each output slice must match `values[i].to_string()`
    #[test]
    fn content_matches_std() {
        let mut lcg: u64 = 0xcafe_f00d_dead_beef;
        let n = 1000usize;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            values.push(lcg);
        }

        let mut offsets = vec![0u32; n + 1];
        let mut out = vec![0u8; 20 * n];
        fmt_batch(&values, &mut out, &mut offsets).unwrap();

        for (i, &v) in values.iter().enumerate() {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            let got = std::str::from_utf8(&out[s..e]).unwrap();
            let expected = v.to_string();
            assert_eq!(got, expected, "content mismatch at index {i}: value {v}");
        }
    }

    /// A batch of values in [10^16, 10^17) should trigger Homogeneous17_20 and
    /// produce correct output
    #[test]
    fn homogeneous_path() {
        let base: u64 = 10_000_000_000_000_000; // 10^16; values in [base, 10*base) have 17 digits
        let count = 10_000usize;
        let mut values = Vec::with_capacity(count);
        let mut lcg: u64 = 0xabcd_1234;
        for _ in 0..count {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Keep in [10^16, 10^17)
            let v = base + (lcg % (base * 10 - base));
            values.push(v);
        }

        // All should be 17 digits.
        for &v in &values {
            assert_eq!(digit_count(v), 17, "expected 17-digit value, got {v}");
        }

        // Verify selector picks Homogeneous17_20.
        let lengths: Vec<u8> = values.iter().map(|&v| digit_count(v) as u8).collect();
        let variant = select(&lengths, BatchConfig::default());
        assert_eq!(
            variant,
            Variant::Homogeneous17_20,
            "expected Homogeneous17_20 variant"
        );

        // Verify correctness.
        let mut offsets = vec![0u32; count + 1];
        let mut out = vec![0u8; 20 * count];
        fmt_batch(&values, &mut out, &mut offsets).unwrap();

        for (i, &v) in values.iter().enumerate() {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            let got = std::str::from_utf8(&out[s..e]).unwrap();
            let expected = v.to_string();
            assert_eq!(got, expected, "homogeneous path: mismatch at index {i}");
        }
    }

    /// `fmt_batch_joined` should produce the expected CSV
    #[test]
    fn joined() {
        let values: &[u64] = &[1, 22, 333, 4444];
        let mut out = [0u8; 64];
        let len = fmt_batch_joined(values, b',', &mut out).unwrap();
        let got = std::str::from_utf8(&out[..len]).unwrap();
        assert_eq!(got, "1,22,333,4444");
    }

    /// `fmt_batch` returns Err on wrong offsets length
    #[test]
    fn batch_err_offsets_mismatch() {
        let values = [1u64, 2, 3];
        let mut out = [0u8; 64];
        let mut offsets = [0u32; 2]; // wrong: should be 4
        let result = fmt_batch(&values, &mut out, &mut offsets);
        assert!(matches!(
            result,
            Err(Error::OffsetsLengthMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    /// `fmt_batch` returns Err on buffer too small
    #[test]
    fn batch_err_buf_too_small() {
        let values = [12345u64, 67890];
        let mut out = [0u8; 1]; // too small
        let mut offsets = [0u32; 3];
        let result = fmt_batch(&values, &mut out, &mut offsets);
        assert!(matches!(result, Err(Error::BufferTooSmall { .. })));
    }
}
