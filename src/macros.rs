//! Drop-in `write!` / `writeln!` macros and a `write_joined!` helper
//!
//! The `write!` and `writeln!` macros are true drop-in replacements for
//! `core::write!` / `core::writeln!`:
//!
//! - Same call shape: `vitoa::write!(target, fmt_str, args...)`
//! - Same return type: `Result<(), core::fmt::Error>`
//! - Fast path when target implements [`crate::FastIntWrite`] (`String` or
//!   `Vec<u8>`) AND the format string is one of `"{}"` … `"{}{}{}{}{}{}{}{}"`
//! - Transparent fallback to `::core::write!` / `::core::writeln!` otherwise
//!
//! `write_joined!` is a crate-specific helper with no `core` equivalent;
//! it writes values separated by a byte, returning `Result<usize, crate::Error>`
//!
//! Author: aav

/// Sequence-write helper: routes every `$a` through [`crate::FastIntArg`]
///
/// `#[doc(hidden)]` — not part of the public API. The variadic body lives
/// here once; the public `write!` arms below are thin per-arity dispatchers
#[macro_export]
#[doc(hidden)]
macro_rules! __seq_write {
    ($w:expr, $($a:expr),+ $(,)?) => {{
        let __w = &mut $w;
        (|| -> ::core::result::Result<(), ::core::fmt::Error> {
            $( <_ as $crate::FastIntArg>::write_into($a, __w)?; )+
            Ok(())
        })()
    }};
}

/// Like [`__seq_write`] but appends a newline byte after the last value
#[macro_export]
#[doc(hidden)]
macro_rules! __seq_writeln {
    ($w:expr, $($a:expr),+ $(,)?) => {{
        let __w = &mut $w;
        (|| -> ::core::result::Result<(), ::core::fmt::Error> {
            $( <_ as $crate::FastIntArg>::write_into($a, __w)?; )+
            $crate::FastIntWrite::write_u64_fast_newline(__w)
        })()
    }};
}

/// Drop-in replacement for [`core::write!`]
///
/// Same call shape and return type. Fast path when target is `FastIntWrite`
/// (`String` / `Vec<u8>`) and format string is `"{}"` through
/// `"{}{}{}{}{}{}{}{}"` — args are routed through [`crate::FastIntArg`]
/// (u8..u64 → `write_u64_fast`, u128 → `write_u128_fast`, no truncation)
/// Anything else falls through to `::core::write!`
///
/// # Example
///
/// ```rust
/// let mut s = String::new();
/// vitoa::write!(&mut s, "{}", 42u64).unwrap();
/// assert_eq!(s, "42");
///
/// // u128 is correct via FastIntArg dispatch
/// let mut s = String::new();
/// vitoa::write!(&mut s, "{}", u128::MAX).unwrap();
/// assert_eq!(s, "340282366920938463463374607431768211455");
/// ```
#[macro_export]
macro_rules! write {
    ($w:expr, "{}",                 $a:expr $(,)?)                                                                  => { $crate::__seq_write!($w, $a) };
    ($w:expr, "{}{}",               $a:expr, $b:expr $(,)?)                                                         => { $crate::__seq_write!($w, $a, $b) };
    ($w:expr, "{}{}{}",             $a:expr, $b:expr, $c:expr $(,)?)                                                => { $crate::__seq_write!($w, $a, $b, $c) };
    ($w:expr, "{}{}{}{}",           $a:expr, $b:expr, $c:expr, $d:expr $(,)?)                                       => { $crate::__seq_write!($w, $a, $b, $c, $d) };
    ($w:expr, "{}{}{}{}{}",         $a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?)                              => { $crate::__seq_write!($w, $a, $b, $c, $d, $e) };
    ($w:expr, "{}{}{}{}{}{}",       $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr $(,)?)                     => { $crate::__seq_write!($w, $a, $b, $c, $d, $e, $f) };
    ($w:expr, "{}{}{}{}{}{}{}",     $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr $(,)?)            => { $crate::__seq_write!($w, $a, $b, $c, $d, $e, $f, $g) };
    ($w:expr, "{}{}{}{}{}{}{}{}",   $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr $(,)?)   => { $crate::__seq_write!($w, $a, $b, $c, $d, $e, $f, $g, $h) };
    ($w:expr, $($t:tt)*) => { ::core::write!($w, $($t)*) };
}

/// Drop-in replacement for [`core::writeln!`] — same as [`crate::write!`]
/// but appends a trailing `'\n'`
///
/// # Example
///
/// ```rust
/// let mut s = String::new();
/// vitoa::writeln!(&mut s, "{}", 99u64).unwrap();
/// assert_eq!(s, "99\n");
/// ```
#[macro_export]
macro_rules! writeln {
    ($w:expr, "{}",                 $a:expr $(,)?)                                                                  => { $crate::__seq_writeln!($w, $a) };
    ($w:expr, "{}{}",               $a:expr, $b:expr $(,)?)                                                         => { $crate::__seq_writeln!($w, $a, $b) };
    ($w:expr, "{}{}{}",             $a:expr, $b:expr, $c:expr $(,)?)                                                => { $crate::__seq_writeln!($w, $a, $b, $c) };
    ($w:expr, "{}{}{}{}",           $a:expr, $b:expr, $c:expr, $d:expr $(,)?)                                       => { $crate::__seq_writeln!($w, $a, $b, $c, $d) };
    ($w:expr, "{}{}{}{}{}",         $a:expr, $b:expr, $c:expr, $d:expr, $e:expr $(,)?)                              => { $crate::__seq_writeln!($w, $a, $b, $c, $d, $e) };
    ($w:expr, "{}{}{}{}{}{}",       $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr $(,)?)                     => { $crate::__seq_writeln!($w, $a, $b, $c, $d, $e, $f) };
    ($w:expr, "{}{}{}{}{}{}{}",     $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr $(,)?)            => { $crate::__seq_writeln!($w, $a, $b, $c, $d, $e, $f, $g) };
    ($w:expr, "{}{}{}{}{}{}{}{}",   $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr $(,)?)   => { $crate::__seq_writeln!($w, $a, $b, $c, $d, $e, $f, $g, $h) };
    ($w:expr, $($t:tt)*) => { ::core::writeln!($w, $($t)*) };
}

/// Write integer values separated by a compile-time byte separator
///
/// Emits `v0 sep v1 sep ... vN-1` into `$out` with no trailing separator
/// Returns `Ok(total_bytes_written)` on success, or `Err(crate::Error::BufferTooSmall)`
/// when the buffer is too small
///
/// Unlike `core::write!`, this macro writes to a raw `&mut [u8]` with no
/// `fmt::Write` overhead and has no `core` equivalent
///
/// # Arguments
///
/// * `$out` - `&mut [u8]` destination buffer
/// * `sep = $sep` - Byte separator written between values (e.g. `b','`)
/// * `$val` - One or more `as u64`-castable integer expressions
///
/// # Example
///
/// ```rust
/// let mut buf = [0u8; 64];
/// let len = vitoa::write_joined!(buf, sep = b',', 1u64, 22u64, 333u64).unwrap();
/// assert_eq!(&buf[..len], b"1,22,333");
/// ```
///
/// A single value emits no separator:
///
/// ```rust
/// let mut buf = [0u8; 64];
/// let len = vitoa::write_joined!(buf, sep = b'|', 42u64).unwrap();
/// assert_eq!(&buf[..len], b"42");
/// ```
///
/// Space-separated, mixed widths:
///
/// ```rust
/// let mut buf = [0u8; 64];
/// let len = vitoa::write_joined!(buf, sep = b' ', 10u8, 200u16, 3000u32, 40000u64).unwrap();
/// assert_eq!(&buf[..len], b"10 200 3000 40000");
/// ```
#[macro_export]
macro_rules! write_joined {
    ($out:expr, sep = $sep:expr, $($val:expr),+ $(,)?) => {{
        #[allow(unused_mut)]
        let mut __off: usize = 0;
        let __result: Result<(), $crate::Error> = (|| {
            $crate::write_joined!(@first $out, __off, $sep, $($val),+);
            Ok(())
        })();
        __result.map(|_| __off)
    }};

    (@first $out:expr, $off:ident, $sep:expr, $val:expr) => {
        {
            let __have = $out.len();
            let __slice_off: usize = $off;
            if __slice_off > __have {
                return Err($crate::Error::BufferTooSmall { need: __slice_off, have: __have });
            }
            // SAFETY: __slice_off <= __have == $out.len().
            let __n = $crate::fmt($val as u64, unsafe { $out.get_unchecked_mut(__slice_off..) })?;
            $off += __n;
        }
    };

    (@first $out:expr, $off:ident, $sep:expr, $val:expr, $($rest:expr),+) => {
        {
            let __have = $out.len();
            let __slice_off: usize = $off;
            if __slice_off > __have {
                return Err($crate::Error::BufferTooSmall { need: __slice_off, have: __have });
            }
            // SAFETY: __slice_off <= __have == $out.len().
            let __n = $crate::fmt($val as u64, unsafe { $out.get_unchecked_mut(__slice_off..) })?;
            $off += __n;
        }
        $crate::write_joined!(@rest $out, $off, $sep, $($rest),+);
    };

    (@rest $out:expr, $off:ident, $sep:expr, $val:expr) => {
        // Write the separator byte safely — capture metavariables before unsafe.
        {
            let __have = $out.len();
            let __sep_byte: u8 = $sep;
            let __sep_off: usize = $off;
            if __sep_off >= __have {
                return Err($crate::Error::BufferTooSmall { need: __sep_off + 1, have: __have });
            }
            // SAFETY: __sep_off < __have == $out.len(), checked immediately above.
            unsafe { *$out.get_unchecked_mut(__sep_off) = __sep_byte; }
            $off += 1;
        }
        {
            let __have = $out.len();
            let __slice_off: usize = $off;
            if __slice_off > __have {
                return Err($crate::Error::BufferTooSmall { need: __slice_off, have: __have });
            }
            // SAFETY: __slice_off <= __have == $out.len().
            let __n = $crate::fmt($val as u64, unsafe { $out.get_unchecked_mut(__slice_off..) })?;
            $off += __n;
        }
    };

    (@rest $out:expr, $off:ident, $sep:expr, $val:expr, $($rest:expr),+) => {
        {
            let __have = $out.len();
            let __sep_byte: u8 = $sep;
            let __sep_off: usize = $off;
            if __sep_off >= __have {
                return Err($crate::Error::BufferTooSmall { need: __sep_off + 1, have: __have });
            }
            // SAFETY: __sep_off < __have, checked immediately above.
            unsafe { *$out.get_unchecked_mut(__sep_off) = __sep_byte; }
            $off += 1;
        }
        {
            let __have = $out.len();
            let __slice_off: usize = $off;
            if __slice_off > __have {
                return Err($crate::Error::BufferTooSmall { need: __slice_off, have: __have });
            }
            // SAFETY: __slice_off <= __have == $out.len().
            let __n = $crate::fmt($val as u64, unsafe { $out.get_unchecked_mut(__slice_off..) })?;
            $off += __n;
        }
        $crate::write_joined!(@rest $out, $off, $sep, $($rest),+);
    };
}

#[cfg(test)]
mod tests {
    use crate::FastIntArg as _;

    /// Exhaustive: every `u8` written through the macro must match `core::write!`
    #[test]
    fn exhaustive_u8_matches_core() {
        use std::fmt::Write as _;
        let mut fast = String::new();
        let mut core_s = String::new();
        for v in 0u8..=u8::MAX {
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", v).unwrap();
            core::write!(&mut core_s, "{}", v).unwrap();
            assert_eq!(fast, core_s, "mismatch at u8 {v}");
        }
    }

    /// Exhaustive: every `u16` written through the macro must match `core::write!`
    #[test]
    fn exhaustive_u16_matches_core() {
        use std::fmt::Write as _;
        let mut fast = String::new();
        let mut core_s = String::new();
        for v in 0u16..=u16::MAX {
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", v).unwrap();
            core::write!(&mut core_s, "{}", v).unwrap();
            assert_eq!(fast, core_s, "mismatch at u16 {v}");
        }
    }

    /// LCG sweep for u32: 1M random samples through macro vs `core::write!`
    #[test]
    fn lcg_sweep_u32_matches_core() {
        use std::fmt::Write as _;
        let mut fast = String::new();
        let mut core_s = String::new();
        let mut s: u64 = 0xdead_beef_cafe_f00d;
        for _ in 0..1_000_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = s as u32;
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", v).unwrap();
            core::write!(&mut core_s, "{}", v).unwrap();
            assert_eq!(fast, core_s, "mismatch at u32 {v}");
        }
    }

    /// LCG sweep for u64: 1M random samples through macro vs `core::write!`
    #[test]
    fn lcg_sweep_u64_matches_core() {
        use std::fmt::Write as _;
        let mut fast = String::new();
        let mut core_s = String::new();
        let mut s: u64 = 0xfeed_face_dead_beef;
        for _ in 0..1_000_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", s).unwrap();
            core::write!(&mut core_s, "{}", s).unwrap();
            assert_eq!(fast, core_s, "mismatch at u64 {s}");
        }
    }

    /// LCG sweep for u128 — proves the NEW u128 dispatch is correct (no truncation)
    /// across 1M random samples spanning the full 128-bit range plus edge cases
    #[test]
    fn lcg_sweep_u128_matches_core() {
        use std::fmt::Write as _;
        let mut fast = String::new();
        let mut core_s = String::new();
        // Edge cases first
        let edges: &[u128] = &[
            0,
            1,
            u8::MAX as u128,
            u16::MAX as u128,
            u32::MAX as u128,
            u64::MAX as u128,
            (u64::MAX as u128) + 1,
            10u128.pow(16) - 1,
            10u128.pow(16),
            10u128.pow(16) + 1,
            10u128.pow(32) - 1,
            10u128.pow(32),
            10u128.pow(32) + 1,
            10u128.pow(38),
            u128::MAX,
        ];
        for &v in edges {
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", v).unwrap();
            core::write!(&mut core_s, "{}", v).unwrap();
            assert_eq!(fast, core_s, "u128 edge mismatch at {v}");
        }
        // Random sweep
        let mut s: u64 = 0xcafe_f00d_dead_beef;
        for _ in 0..1_000_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let s2 = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((s as u128) << 64) | s2 as u128;
            fast.clear();
            core_s.clear();
            crate::write!(&mut fast, "{}", v).unwrap();
            core::write!(&mut core_s, "{}", v).unwrap();
            assert_eq!(fast, core_s, "u128 sweep mismatch at {v}");
        }
    }

    /// FastIntArg trait (the macro's dispatch trait) — direct method calls
    /// for each width must produce `to_string()` output. Exhaustive u8/u16,
    /// LCG sweep u32/u64/u128. Catches any bug in the trait impls themselves
    /// independent of the macro plumbing
    #[test]
    fn fast_int_arg_matches_to_string() {
        let mut s = String::new();
        for v in 0u8..=u8::MAX {
            s.clear();
            v.write_into(&mut s).unwrap();
            assert_eq!(s, v.to_string(), "FastIntArg u8 at {v}");
        }
        for v in 0u16..=u16::MAX {
            s.clear();
            v.write_into(&mut s).unwrap();
            assert_eq!(s, v.to_string(), "FastIntArg u16 at {v}");
        }
        let mut lcg: u64 = 0xabcd;
        for _ in 0..200_000 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v32 = lcg as u32;
            s.clear();
            v32.write_into(&mut s).unwrap();
            assert_eq!(s, v32.to_string(), "FastIntArg u32 at {v32}");
            s.clear();
            lcg.write_into(&mut s).unwrap();
            assert_eq!(s, lcg.to_string(), "FastIntArg u64 at {lcg}");
            let lcg2 = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v128 = ((lcg as u128) << 64) | lcg2 as u128;
            s.clear();
            v128.write_into(&mut s).unwrap();
            assert_eq!(s, v128.to_string(), "FastIntArg u128 at {v128}");
        }
    }

    /// Verify `write!` fast-path on String produces the same bytes as core::write!
    #[test]
    fn write_string_fast_path() {
        use std::fmt::Write as _;

        let cases: &[(u64, u64, u64, u64)] = &[
            (0, 1, 22, 333),
            (u64::MAX, 0, 42, 1_234_567_890_123_456_789),
        ];

        for &(a, b, c, d) in cases {
            let mut core_s = String::new();
            core::write!(&mut core_s, "{}{}{}{}", a, b, c, d).unwrap();

            let mut fast_s = String::new();
            crate::write!(&mut fast_s, "{}{}{}{}", a, b, c, d).unwrap();

            assert_eq!(
                fast_s, core_s,
                "write! mismatch: fast={fast_s:?} core={core_s:?}"
            );
        }
    }

    /// Verify `writeln!` fast-path on String appends '\n'
    #[test]
    fn writeln_string_fast_path() {
        use std::fmt::Write as _;

        let mut core_s = String::new();
        core::writeln!(&mut core_s, "{}", 99u64).unwrap();

        let mut fast_s = String::new();
        crate::writeln!(&mut fast_s, "{}", 99u64).unwrap();

        assert_eq!(fast_s, core_s);
    }

    /// Verify `write!` generic fallback works for non-integer format strings
    #[test]
    fn write_generic_fallback() {
        use std::fmt::Write as _;
        let mut s = String::new();
        crate::write!(&mut s, "hello {}", "world").unwrap();
        assert_eq!(s, "hello world");
    }

    /// Verify `write!` fast-path on Vec<u8>
    #[test]
    fn write_vec_fast_path() {
        let mut v: Vec<u8> = Vec::new();
        crate::write!(&mut v, "{}{}", 1u64, 2u64).unwrap();
        assert_eq!(v, b"12");
    }

    /// Verify `write_joined!` returns Ok(bytes_written) and produces correct CSV
    #[test]
    fn write_joined_result() {
        let mut buf = [0u8; 64];
        let len = crate::write_joined!(buf, sep = b',', 1u64, 22u64, 333u64).unwrap();
        assert_eq!(&buf[..len], b"1,22,333");
    }

    /// Verify `write_joined!` errors on a too-small buffer
    #[test]
    fn write_joined_buffer_too_small() {
        let mut buf = [0u8; 2];
        let result = crate::write_joined!(buf, sep = b',', 1u64, 22u64, 333u64);
        assert!(result.is_err());
    }

    /// Verify return type and value semantics
    #[test]
    fn return_value_semantics() {
        let mut buf = [0u8; 64];
        let len: usize = crate::write_joined!(buf, sep = b'-', 1u64, 2u64, 3u64).unwrap();
        assert_eq!(len, 5);
    }
}
