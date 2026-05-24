//! Lookup tables and digit-counting utilities shared across SIMD and scalar paths
//!
//! Author: aav
// --------------------------------------------------
// constants
// --------------------------------------------------
/// 2-digit ASCII pair table, indexed 00..99
///
/// `TWO_DIGIT[n][0]` is the tens digit of `n` as ASCII, `TWO_DIGIT[n][1]`
/// is the ones digit. Built in a `const` block so it lives in rodata and is
/// never heap-allocated
pub(crate) const TWO_DIGIT: [[u8; 2]; 100] = {
    let mut t = [[0u8; 2]; 100];
    let mut i = 0;
    while i < 100 {
        // SAFETY: i < 100 == t.len(), both indices (0 and 1) are within the
        // inner array of length 2. Clippy cannot verify bounds in const
        // context, so we suppress the lint here.
        #[allow(clippy::indexing_slicing)]
        {
            t[i][0] = b'0' + (i / 10) as u8;
            t[i][1] = b'0' + (i % 10) as u8;
        }
        i += 1;
    }
    t
};

#[inline(always)]
/// Returns the number of decimal digits in `n` (returns 1 for n == 0)
///
/// Uses `u64::ilog10` for values > 0, which the compiler lowers to a BSR +
/// table lookup on x86-64. The special case for zero avoids undefined
/// behaviour from `ilog10(0)`
///
/// # Arguments
///
/// * `n` - The value whose digit count is requested
///
/// # Example
///
/// ```rust
/// use vitoa::count_digits;
/// assert_eq!(count_digits(0), 1);
/// assert_eq!(count_digits(9), 1);
/// assert_eq!(count_digits(10), 2);
/// assert_eq!(count_digits(u64::MAX), 20);
/// ```
pub fn count_digits(n: u64) -> u32 {
    if n == 0 { 1 } else { n.ilog10() + 1 }
}

#[cfg(simd_ifma)]
#[inline]
/// Write exactly 4 ASCII digits for `r ∈ [0, 9999]` at `ptr`
///
/// Splits `r` into two 2-digit halves and writes them using the [`TWO_DIGIT`]
/// table. Both halves are written unconditionally (including leading zeros),
/// so callers that need trimmed output should use
/// [`write_one_two_three_or_four_digits_10000`] instead
///
/// # Arguments
///
/// * `ptr` - Pointer to the first of 4 output bytes
/// * `r` - Value in `[0, 9999]` to write as exactly 4 ASCII digits
///
/// # Safety
///
/// `ptr` must be valid for 4 sequential byte writes
pub(crate) unsafe fn write_four_digits_10000(ptr: *mut u8, r: u32) {
    unsafe {
        let hi = (r / 100) as usize;
        let lo = (r % 100) as usize;
        // SAFETY: hi = r/100; r <= 9999 so hi <= 99 < TWO_DIGIT.len().
        //         lo = r%100; lo < 100 == TWO_DIGIT.len().
        let hi_pair = TWO_DIGIT.get_unchecked(hi);
        let lo_pair = TWO_DIGIT.get_unchecked(lo);
        ptr.add(0).write(hi_pair[0]);
        ptr.add(1).write(hi_pair[1]);
        ptr.add(2).write(lo_pair[0]);
        ptr.add(3).write(lo_pair[1]);
    }
}

#[cfg(simd_ifma)]
#[inline]
/// Write 1, 2, 3, or 4 ASCII digits for `q ∈ [1, 9999]` at `ptr`
///
/// Determines the minimum number of digits needed and writes only those bytes,
/// with no leading zeros. Used by the 17–20-digit SIMD path to emit the
/// variable-length prefix before the fixed 16-digit SIMD chunk. Returns the
/// number of bytes written
///
/// # Arguments
///
/// * `ptr` - Pointer to the first output byte
/// * `q` - Value in `[1, 9999]` to write (must be at least 1; 0 is not supported)
///
/// # Safety
///
/// `ptr` must be valid for up to 4 sequential byte writes
pub(crate) unsafe fn write_one_two_three_or_four_digits_10000(ptr: *mut u8, q: u32) -> usize {
    unsafe {
        if q < 10 {
            ptr.write(b'0' + q as u8);
            1
        } else if q < 100 {
            // SAFETY: q < 100 == TWO_DIGIT.len().
            let p = TWO_DIGIT.get_unchecked(q as usize);
            ptr.write(p[0]);
            ptr.add(1).write(p[1]);
            2
        } else if q < 1000 {
            // SAFETY: lo = q%100 < 100 == TWO_DIGIT.len().
            let lo = (q % 100) as usize;
            let lo_pair = TWO_DIGIT.get_unchecked(lo);
            ptr.write(b'0' + (q / 100) as u8);
            ptr.add(1).write(lo_pair[0]);
            ptr.add(2).write(lo_pair[1]);
            3
        } else {
            write_four_digits_10000(ptr, q);
            4
        }
    }
}
