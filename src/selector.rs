//! Dynamic SIMD-variant selector implementing paper §5.6, Algorithm 1
//!
//! Samples the digit-length distribution of a batch using a stride walk,
//! builds a histogram, and decides between the heterogeneous masked-store
//! path and the homogeneous unmasked-store path based on the dominant length
//! and a configurable frequency threshold
//!
//! Author: aav

/// Which SIMD variant the batch driver should use
///
/// Selected by [`select`] based on the sampled digit-length histogram of the
/// input batch. The choice is advisory — the batch driver is responsible for
/// falling back to [`Variant::Heterogeneous`] if the required CPU features are
/// not available at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Heterogeneous masked-store path (paper §5.4, Figure 6)
    ///
    /// Works for all digit lengths 1–20. Always safe to use
    Heterogeneous,
    /// Homogeneous unmasked-store path (paper §5.5, Figure 7)
    ///
    /// Only valid when the dominant digit length is in [17, 20] and its
    /// sampled frequency meets or exceeds [`BatchConfig::homo_threshold`]
    Homogeneous17_20,
}

/// Tuning parameters for the dynamic selector (paper §5.6, Algorithm 1)
///
/// Controls the trade-off between sampling overhead and selector accuracy
/// The defaults are calibrated for large batches (>= 1000 elements) with
/// strongly clustered digit lengths; for mixed-length batches the selector
/// will correctly fall back to [`Variant::Heterogeneous`]
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Fraction of the batch to stride-sample
    ///
    /// Must be in `(0.0, 1.0]`. Default: `0.01` (1%)
    pub sample_rate: f64,

    /// Minimum sampled frequency of the dominant digit length required to
    /// choose the homogeneous path
    ///
    /// Must be in `(0.0, 1.0]`. Default: `0.95`
    pub homo_threshold: f64,
}

/// [`BatchConfig`] implementation of [`Default`]
impl Default for BatchConfig {
    #[inline]
    fn default() -> Self {
        Self {
            sample_rate: 0.01,
            homo_threshold: 0.95,
        }
    }
}

#[inline]
/// Select the appropriate SIMD [`Variant`] for a batch of digit lengths
///
/// Implements Algorithm 1 from the paper. Stride-samples `lengths` at a
/// deterministic step computed from `cfg.sample_rate`, builds a 21-bucket
/// histogram (indexed by digit count 1..=20), and returns
/// [`Variant::Homogeneous17_20`] only if the dominant length is in [17, 20]
/// and its relative frequency meets or exceeds `cfg.homo_threshold`
/// Returns [`Variant::Heterogeneous`] for all other cases including an
/// empty input slice
///
/// Stride sampling is deterministic (no RNG dependency); for large batches
/// with strongly clustered lengths it is as discriminating as random sampling
///
/// # Arguments
///
/// * `lengths` - Per-element digit counts as `u8` values in [1, 20]
/// * `cfg` - Tuning parameters controlling sample rate and homogeneity threshold
pub(crate) fn select(lengths: &[u8], cfg: BatchConfig) -> Variant {
    if lengths.is_empty() {
        return Variant::Heterogeneous;
    }
    let m = ((cfg.sample_rate * lengths.len() as f64).ceil() as usize)
        .max(1)
        .min(lengths.len());
    let mut hist = [0u32; 21]; // indexed by digit count 1..=20
    let step = (lengths.len() / m).max(1);
    let mut taken = 0u32;
    let mut i = 0usize;
    while i < lengths.len() && taken < m as u32 {
        // SAFETY: i < lengths.len() by loop condition.
        let digit_len = unsafe { *lengths.get_unchecked(i) } as usize;
        let slot = digit_len.min(20);
        // SAFETY: slot <= 20 < hist.len() (21 elements).
        unsafe { *hist.get_unchecked_mut(slot) += 1 };
        i += step;
        taken += 1;
    }
    // fold avoids `max_by_key` which calls unwrap internally.
    let (dom_len, dom_count) =
        hist.iter()
            .enumerate()
            .fold((0usize, 0u32), |(best_i, best_c), (i, &c)| {
                if c > best_c { (i, c) } else { (best_i, best_c) }
            });
    if taken > 0
        && (dom_count as f64 / taken as f64) >= cfg.homo_threshold
        && (17..=20).contains(&dom_len)
    {
        Variant::Homogeneous17_20
    } else {
        Variant::Heterogeneous
    }
}
