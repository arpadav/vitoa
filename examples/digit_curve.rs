#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
//! Per-digit-count benchmark — 4 charts, apples-to-apples by target type
//!
//! Charts emitted (one CSV → one chart):
//!
//!   1. `digit_curve_write_string.csv`   — write single int into `String`
//!      Contenders: `core::write!`, `itoa::Buffer + push_str`, `vitoa::write!`
//!
//!   2. `digit_curve_writeln_string.csv` — writeln single int into `String`
//!      Contenders: `core::writeln!`, `itoa::Buffer + push_str + '\n'`, `vitoa::writeln!`
//!
//!   3. `digit_curve_write_bytes.csv`    — write single int into `&mut [u8]`
//!      Contenders: `itoa::Buffer::format`, `vitoa::fmt`
//!
//!   4. `digit_curve_csv_bytes.csv`      — comma-join CSV_N values into `&mut [u8]`
//!      Contenders: `itoa::Buffer + copy CSV loop`, `vitoa::write_joined!` (the
//!      crate's actual separator-join macro — NOT `vitoa::write!` in a loop)
//!
//! Methodology: target buffers + `itoa::Buffer` constructed ONCE before each
//! timed closure; inputs through `black_box`; median of `TRIALS` trials
//!
//! Run: `cargo run --release --example digit_curve --features macros [TRIALS] [OUT_DIR]`
//! Defaults: TRIALS=10, OUT_DIR=/tmp
//!
//! Author: aav

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::hint::black_box;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

const ITERS: usize = 100_000;
const WARMUP: usize = 10_000;
const DEFAULT_TRIALS: usize = 10;
const VALUES_PER_BATCH: usize = 4096;
const CSV_N: usize = 4;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn bench_once<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_median<F: FnMut()>(trials: usize, mut f: F) -> f64 {
    let mut samples: Vec<f64> = (0..trials).map(|_| bench_once(&mut f)).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn build_u32(d: u32, n: usize, lcg_state: &mut u64) -> Vec<u32> {
    let (lo, hi) = if d == 1 {
        (0u64, 10u64)
    } else {
        (10u64.pow(d - 1), 10u64.pow(d))
    };
    (0..n)
        .map(|_| (lo + lcg(lcg_state) % (hi - lo)) as u32)
        .collect()
}

fn build_u64(d: u32, n: usize, lcg_state: &mut u64) -> Vec<u64> {
    if d == 20 {
        let lo: u64 = 10_000_000_000_000_000_000;
        (0..n)
            .map(|_| lo + lcg(lcg_state) % (u64::MAX - lo))
            .collect()
    } else {
        let (lo, hi) = if d == 1 {
            (0u64, 10u64)
        } else {
            (10u64.pow(d - 1), 10u64.pow(d))
        };
        (0..n).map(|_| lo + lcg(lcg_state) % (hi - lo)).collect()
    }
}

fn build_u128(d: u32, n: usize, lcg_state: &mut u64) -> Vec<u128> {
    let (lo, hi) = if d == 1 {
        (0u128, 10u128)
    } else if d <= 38 {
        (10u128.pow(d - 1), 10u128.pow(d))
    } else {
        (10u128.pow(38), u128::MAX)
    };
    let span = hi.saturating_sub(lo).max(1);
    (0..n)
        .map(|_| {
            let r = ((lcg(lcg_state) as u128) << 64) | lcg(lcg_state) as u128;
            lo + r % span
        })
        .collect()
}

/// Measure the 4 charts' contenders for one (width, digits, values) cell
macro_rules! measure_cell {
    (
        $trials:expr, $width:literal, $d:expr, $values:expr, $tmp_len:expr,
        $write_string:expr, $writeln_string:expr, $write_bytes:expr, $csv_bytes:expr
    ) => {{
        let trials = $trials;
        let values: &[_] = &$values;
        let d = $d;
        let n_vals = values.len();

        // chart 1 — write single int into String
        let mut s = String::with_capacity($tmp_len);
        let mut i = 0usize;
        let s_core = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            write!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len());
        });
        let mut s = String::with_capacity($tmp_len);
        let mut ibuf = itoa::Buffer::new();
        let mut i = 0usize;
        let s_itoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            s.push_str(ibuf.format(black_box(v)));
            black_box(s.len());
        });
        let mut s = String::with_capacity($tmp_len);
        let mut i = 0usize;
        let s_vitoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            vitoa::write!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len());
        });
        writeln!(
            $write_string,
            "{},{d},{:.3},{:.3},{:.3}",
            $width, s_core, s_itoa, s_vitoa
        )
        .unwrap();

        // chart 2 — writeln single int into String
        let mut s = String::with_capacity($tmp_len + 1);
        let mut i = 0usize;
        let s_core = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            writeln!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len());
        });
        let mut s = String::with_capacity($tmp_len + 1);
        let mut ibuf = itoa::Buffer::new();
        let mut i = 0usize;
        let s_itoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            s.push_str(ibuf.format(black_box(v)));
            s.push('\n');
            black_box(s.len());
        });
        let mut s = String::with_capacity($tmp_len + 1);
        let mut i = 0usize;
        let s_vitoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            s.clear();
            vitoa::writeln!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len());
        });
        writeln!(
            $writeln_string,
            "{},{d},{:.3},{:.3},{:.3}",
            $width, s_core, s_itoa, s_vitoa
        )
        .unwrap();

        // chart 3 — write single int into &mut [u8]
        let mut ibuf = itoa::Buffer::new();
        let mut i = 0usize;
        let b_itoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            black_box(ibuf.format(black_box(v)).len());
        });
        let mut buf = [0u8; 40];
        let mut i = 0usize;
        let b_vitoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            black_box(vitoa::fmt(black_box(v), &mut buf).unwrap());
        });
        writeln!($write_bytes, "{},{d},{:.3},{:.3}", $width, b_itoa, b_vitoa).unwrap();

        // chart 4 — comma-join CSV_N values into &mut [u8]
        // vitoa uses write_joined! (the separator-join macro), not write! in a loop.
        let cap = $tmp_len * CSV_N + CSV_N;
        let mut ibuf = itoa::Buffer::new();
        let mut out = vec![0u8; cap];
        let mut i = 0usize;
        let c_itoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            let mut off = 0usize;
            for k in 0..CSV_N {
                if k > 0 {
                    out[off] = b',';
                    off += 1;
                }
                let s = ibuf.format(black_box(v));
                let len = s.len();
                out[off..off + len].copy_from_slice(s.as_bytes());
                off += len;
            }
            black_box(off);
        });
        let mut buf = vec![0u8; cap];
        let mut i = 0usize;
        let c_vitoa = bench_median(trials, || {
            let v = values[i];
            i = (i + 1) % n_vals;
            let len = vitoa::write_joined!(
                &mut buf[..],
                sep = b',',
                black_box(v),
                black_box(v),
                black_box(v),
                black_box(v)
            )
            .unwrap();
            black_box(len);
        });
        writeln!($csv_bytes, "{},{d},{:.3},{:.3}", $width, c_itoa, c_vitoa).unwrap();
    }};
}

fn main() {
    let trials = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TRIALS);
    let out_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    eprintln!(
        "running {trials} trials per measurement; reporting median ns; out_dir = {}",
        out_dir.display()
    );

    let s_header = "width,digits,core_ns,itoa_ns,vitoa_ns\n";
    let b_header = "width,digits,itoa_ns,vitoa_ns\n";
    let mut write_string = File::create(out_dir.join("digit_curve_write_string.csv")).unwrap();
    let mut writeln_string = File::create(out_dir.join("digit_curve_writeln_string.csv")).unwrap();
    let mut write_bytes = File::create(out_dir.join("digit_curve_write_bytes.csv")).unwrap();
    let mut csv_bytes = File::create(out_dir.join("digit_curve_csv_bytes.csv")).unwrap();
    write_string.write_all(s_header.as_bytes()).unwrap();
    writeln_string.write_all(s_header.as_bytes()).unwrap();
    write_bytes.write_all(b_header.as_bytes()).unwrap();
    csv_bytes.write_all(b_header.as_bytes()).unwrap();

    let mut lcg_state = 0xdeadbeef_cafef00d;
    for d in 1..=10u32 {
        let values = build_u32(d, VALUES_PER_BATCH, &mut lcg_state);
        measure_cell!(
            trials,
            "u32",
            d,
            values,
            20,
            write_string,
            writeln_string,
            write_bytes,
            csv_bytes
        );
    }
    for d in 1..=20u32 {
        let values = build_u64(d, VALUES_PER_BATCH, &mut lcg_state);
        measure_cell!(
            trials,
            "u64",
            d,
            values,
            20,
            write_string,
            writeln_string,
            write_bytes,
            csv_bytes
        );
    }
    for d in 1..=39u32 {
        let values = build_u128(d, VALUES_PER_BATCH, &mut lcg_state);
        measure_cell!(
            trials,
            "u128",
            d,
            values,
            40,
            write_string,
            writeln_string,
            write_bytes,
            csv_bytes
        );
    }

    eprintln!("wrote 4 CSVs to {}", out_dir.display());
}
