//! Error type for vitoa public APIs
//!
//! Author: aav

/// Errors returned by vitoa public functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The output buffer is too small to hold the conversion result
    BufferTooSmall {
        /// Number of bytes required
        need: usize,
        /// Number of bytes available
        have: usize,
    },
    /// The `offsets` slice passed to a batch function has the wrong length
    OffsetsLengthMismatch {
        /// Expected length (`values.len() + 1`)
        expected: usize,
        /// Actual length provided
        got: usize,
    },
    /// The cumulative byte length of a batch overflowed `u32`
    ///
    /// Raised by [`crate::fmt_batch`] when the sum of per-element digit
    /// counts exceeds `u32::MAX`. The offset array is `&mut [u32]`, so this
    /// is the largest representable total. Split the batch and call
    /// `fmt_batch` multiple times
    TotalLengthOverflow,
}

/// [`Error`] implementation of [`core::fmt::Display`]
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferTooSmall { need, have } => {
                f.write_str("vitoa: buffer too small: need ")?;
                let mut tmp = [0u8; 20];
                let n = u64_to_decimal(*need as u64, &mut tmp);
                // SAFETY: u64_to_decimal returns n <= 20 == tmp.len();
                // all bytes tmp[0..n] are ASCII decimal digits, valid UTF-8.
                let s = unsafe { core::str::from_utf8_unchecked(tmp.get_unchecked(..n)) };
                f.write_str(s)?;
                f.write_str(" bytes, have ")?;
                let n2 = u64_to_decimal(*have as u64, &mut tmp);
                // SAFETY: same as above.
                let s2 = unsafe { core::str::from_utf8_unchecked(tmp.get_unchecked(..n2)) };
                f.write_str(s2)?;
                f.write_str(" bytes")
            }
            Self::OffsetsLengthMismatch { expected, got } => {
                f.write_str("vitoa: offsets length mismatch: expected ")?;
                let mut tmp = [0u8; 20];
                let n = u64_to_decimal(*expected as u64, &mut tmp);
                // SAFETY: u64_to_decimal returns n <= 20; bytes are valid ASCII.
                let s = unsafe { core::str::from_utf8_unchecked(tmp.get_unchecked(..n)) };
                f.write_str(s)?;
                f.write_str(", got ")?;
                let n2 = u64_to_decimal(*got as u64, &mut tmp);
                // SAFETY: same as above.
                let s2 = unsafe { core::str::from_utf8_unchecked(tmp.get_unchecked(..n2)) };
                f.write_str(s2)
            }
            Self::TotalLengthOverflow => {
                f.write_str("vitoa: batch total byte length exceeds u32::MAX; split the batch")
            }
        }
    }
}

/// [`Error`] implementation of [`std::error::Error`]
impl std::error::Error for Error {}

/// Minimal decimal formatter for usize values used inside `Display` above
///
/// Writes the decimal representation of `v` into `buf[0..n]` and returns `n`
/// No external dependencies — avoids any chance of re-entering Display
///
/// The returned `n` satisfies `1 <= n <= 20` (u64::MAX has 20 decimal digits),
/// so callers may safely use `buf.get_unchecked(..n)`
fn u64_to_decimal(mut v: u64, buf: &mut [u8; 20]) -> usize {
    if v == 0 {
        // SAFETY: buf has length 20; index 0 is always valid.
        unsafe { *buf.get_unchecked_mut(0) = b'0' };
        return 1;
    }
    let mut i = 20usize;
    while v > 0 {
        i -= 1;
        // SAFETY: i starts at 19 and decrements once per digit;
        // u64::MAX has 20 digits so i never underflows below 0.
        unsafe { *buf.get_unchecked_mut(i) = b'0' + (v % 10) as u8 };
        v /= 10;
    }
    let n = 20 - i;
    // SAFETY: i..20 is a valid sub-range of buf (length 20); i >= 0.
    unsafe {
        core::ptr::copy(buf.get_unchecked(i), buf.get_unchecked_mut(0), n);
    }
    n
}
