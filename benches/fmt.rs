#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Criterion benchmarks: vitoa vs `format!`, `core::write!`, `core::writeln!`, `itoa` crate
//!
//! Author: aav
// --------------------------------------------------
// local
// --------------------------------------------------
use vitoa::{fmt, fmt_batch};
// --------------------------------------------------
// external
// --------------------------------------------------
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::{fmt::Write as FmtWrite, hint::black_box};

const BATCH: usize = 10_000;

fn make_lcg() -> impl FnMut() -> u64 {
    let mut s: u64 = 0xdeadbeef_cafef00d;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s
    }
}

fn build_hetero_batch(n: usize) -> Vec<u64> {
    let mut next = make_lcg();
    (0..n).map(|_| next()).collect()
}

fn build_homo_batch(n: usize) -> Vec<u64> {
    let mut next = make_lcg();
    let base: u64 = 10_000_000_000_000_000;
    (0..n).map(|_| base + next() % base * 9).collect()
}

fn bench_single_u64(c: &mut Criterion) {
    let mut g = c.benchmark_group("single_u64");
    g.throughput(Throughput::Elements(1));

    let mut next = make_lcg();
    let mut buf = [0u8; 20];
    g.bench_function("vitoa::fmt", |b| {
        b.iter(|| {
            let v = next();
            black_box(fmt(black_box(v), &mut buf).unwrap())
        })
    });

    let mut next = make_lcg();
    let mut s = String::with_capacity(20);
    g.bench_function("core::write!", |b| {
        b.iter(|| {
            let v = next();
            s.clear();
            write!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len())
        })
    });

    let mut next = make_lcg();
    g.bench_function("format!", |b| {
        b.iter(|| {
            let v = next();
            black_box(format!("{}", black_box(v)))
        })
    });

    let mut next = make_lcg();
    g.bench_function("itoa crate", |b| {
        b.iter(|| {
            let v = next();
            let mut buf = itoa::Buffer::new();
            let s = buf.format(black_box(v));
            black_box(s.len())
        })
    });

    g.finish();
}

fn bench_single_u32(c: &mut Criterion) {
    let mut g = c.benchmark_group("single_u32");
    g.throughput(Throughput::Elements(1));

    let values: Vec<u32> = build_hetero_batch(BATCH)
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let mut idx = 0usize;
    let mut buf = [0u8; 10];
    g.bench_function("vitoa::fmt_u32", |b| {
        b.iter(|| {
            let v = values[idx];
            idx = (idx + 1) % values.len();
            black_box(fmt(black_box(v), &mut buf).unwrap())
        })
    });

    let mut idx = 0usize;
    let mut s = String::with_capacity(10);
    g.bench_function("core::write! (u32)", |b| {
        b.iter(|| {
            let v = values[idx];
            idx = (idx + 1) % values.len();
            s.clear();
            write!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len())
        })
    });

    let mut idx = 0usize;
    g.bench_function("format! (u32)", |b| {
        b.iter(|| {
            let v = values[idx];
            idx = (idx + 1) % values.len();
            black_box(format!("{}", black_box(v)))
        })
    });

    let mut idx = 0usize;
    g.bench_function("itoa crate", |b| {
        b.iter(|| {
            let v = values[idx];
            idx = (idx + 1) % values.len();
            let mut ibuf = itoa::Buffer::new();
            let s = ibuf.format(black_box(v));
            black_box(s.len())
        })
    });

    g.finish();
}

fn bench_concat_4(c: &mut Criterion) {
    let mut g = c.benchmark_group("concat_4xu64");
    g.throughput(Throughput::Elements(4));
    // Inputs wrapped in black_box so LLVM cannot constant-fold the conversion;
    // without this, some variants hoist the entire computation to compile time.
    let v: [u64; 4] = [1_234, 56_789, 99_999_999, 12_345_678_901];

    let mut s = String::with_capacity(64);
    g.bench_function("vitoa::write!", |b| {
        b.iter(|| {
            s.clear();
            vitoa::write!(
                &mut s,
                "{}{}{}{}",
                black_box(v[0]),
                black_box(v[1]),
                black_box(v[2]),
                black_box(v[3])
            )
            .unwrap();
            black_box(s.len())
        })
    });

    let mut s = String::with_capacity(64);
    g.bench_function("core::write!", |b| {
        b.iter(|| {
            s.clear();
            write!(
                &mut s,
                "{}{}{}{}",
                black_box(v[0]),
                black_box(v[1]),
                black_box(v[2]),
                black_box(v[3])
            )
            .unwrap();
            black_box(s.len())
        })
    });

    g.bench_function("format!", |b| {
        b.iter(|| {
            black_box(format!(
                "{}{}{}{}",
                black_box(v[0]),
                black_box(v[1]),
                black_box(v[2]),
                black_box(v[3])
            ))
        })
    });

    g.bench_function("itoa crate", |b| {
        b.iter(|| {
            let mut acc: Vec<u8> = Vec::with_capacity(64);
            let mut ibuf = itoa::Buffer::new();
            acc.extend_from_slice(ibuf.format(black_box(v[0])).as_bytes());
            acc.extend_from_slice(ibuf.format(black_box(v[1])).as_bytes());
            acc.extend_from_slice(ibuf.format(black_box(v[2])).as_bytes());
            acc.extend_from_slice(ibuf.format(black_box(v[3])).as_bytes());
            black_box(acc.len())
        })
    });

    g.finish();
}

fn bench_csv_4(c: &mut Criterion) {
    let mut g = c.benchmark_group("csv_4xu64");
    g.throughput(Throughput::Elements(4));
    let v: [u64; 4] = [1_234, 56_789, 99_999_999, 12_345_678_901];

    let mut buf = [0u8; 64];
    g.bench_function("vitoa::write_joined!", |b| {
        b.iter(|| {
            black_box(
                vitoa::write_joined!(
                    buf,
                    sep = b',',
                    black_box(v[0]),
                    black_box(v[1]),
                    black_box(v[2]),
                    black_box(v[3])
                )
                .unwrap(),
            )
        })
    });

    let mut s = String::with_capacity(64);
    g.bench_function("core::write!", |b| {
        b.iter(|| {
            s.clear();
            write!(
                &mut s,
                "{},{},{},{}",
                black_box(v[0]),
                black_box(v[1]),
                black_box(v[2]),
                black_box(v[3])
            )
            .unwrap();
            black_box(s.len())
        })
    });

    g.bench_function("format!", |b| {
        b.iter(|| {
            black_box(format!(
                "{},{},{},{}",
                black_box(v[0]),
                black_box(v[1]),
                black_box(v[2]),
                black_box(v[3])
            ))
        })
    });

    g.bench_function("itoa crate", |b| {
        b.iter(|| {
            let mut acc: Vec<u8> = Vec::with_capacity(64);
            let mut ibuf = itoa::Buffer::new();
            acc.extend_from_slice(ibuf.format(black_box(v[0])).as_bytes());
            acc.push(b',');
            acc.extend_from_slice(ibuf.format(black_box(v[1])).as_bytes());
            acc.push(b',');
            acc.extend_from_slice(ibuf.format(black_box(v[2])).as_bytes());
            acc.push(b',');
            acc.extend_from_slice(ibuf.format(black_box(v[3])).as_bytes());
            black_box(acc.len())
        })
    });

    g.finish();
}

fn bench_writeln(c: &mut Criterion) {
    let mut g = c.benchmark_group("writeln_u64");
    g.throughput(Throughput::Elements(1));
    let mut next = make_lcg();
    let mut s = String::with_capacity(32);
    g.bench_function("vitoa::writeln!", |b| {
        b.iter(|| {
            let v = next();
            s.clear();
            vitoa::writeln!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len())
        })
    });

    let mut next = make_lcg();
    let mut s = String::with_capacity(32);
    g.bench_function("core::writeln!", |b| {
        b.iter(|| {
            let v = next();
            s.clear();
            writeln!(&mut s, "{}", black_box(v)).unwrap();
            black_box(s.len())
        })
    });

    let mut next = make_lcg();
    g.bench_function("itoa crate", |b| {
        b.iter(|| {
            let v = next();
            let mut acc: Vec<u8> = Vec::with_capacity(32);
            let mut ibuf = itoa::Buffer::new();
            acc.extend_from_slice(ibuf.format(black_box(v)).as_bytes());
            acc.push(b'\n');
            black_box(acc.len())
        })
    });

    g.finish();
}

fn bench_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch_10K");
    g.throughput(Throughput::Elements(BATCH as u64));

    for (name, vals) in [
        ("heterogeneous", build_hetero_batch(BATCH)),
        ("homogeneous_17d", build_homo_batch(BATCH)),
    ] {
        g.bench_with_input(
            BenchmarkId::new("vitoa::fmt_batch", name),
            &vals,
            |b, vals| {
                let mut out = vec![0u8; 20 * vals.len()];
                let mut offsets = vec![0u32; vals.len() + 1];
                b.iter(|| black_box(fmt_batch(vals, &mut out, &mut offsets).unwrap()))
            },
        );

        g.bench_with_input(
            BenchmarkId::new("core::write! loop", name),
            &vals,
            |b, vals| {
                let mut s = String::with_capacity(20 * vals.len());
                b.iter(|| {
                    s.clear();
                    for &v in vals {
                        write!(&mut s, "{v}").unwrap();
                    }
                    black_box(s.len())
                })
            },
        );

        g.bench_with_input(BenchmarkId::new("format! loop", name), &vals, |b, vals| {
            b.iter_batched(
                || 0usize,
                |mut total| {
                    for &v in vals {
                        total += format!("{v}").len();
                    }
                    black_box(total)
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_with_input(BenchmarkId::new("itoa crate", name), &vals, |b, vals| {
            let mut acc: Vec<u8> = Vec::with_capacity(20 * vals.len());
            b.iter(|| {
                acc.clear();
                let mut ibuf = itoa::Buffer::new();
                for &v in vals {
                    acc.extend_from_slice(ibuf.format(v).as_bytes());
                }
                black_box(acc.len())
            })
        });
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_single_u64,
    bench_single_u32,
    bench_concat_4,
    bench_csv_4,
    bench_writeln,
    bench_batch
);
criterion_main!(benches);
