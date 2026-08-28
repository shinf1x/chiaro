//! Minimal row-band parallelism for per-frame kernels.
//!
//! Frames are processed one at a time to bound memory; instead, the rows of a
//! single frame are split into contiguous bands that run on scoped threads.
//! Kernels stay ordinary sequential Rust over a row range.

use std::thread;

/// Number of worker threads to use when a caller passes `0` ("auto").
pub fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
}

/// Resolve a requested thread count: `0` means auto, otherwise at least one.
pub fn resolve_threads(requested: usize) -> usize {
    if requested == 0 {
        default_threads()
    } else {
        requested
    }
}

/// Split `0..rows` into at most `threads` contiguous bands (`0` = auto). Bands
/// start on a multiple of `align` rows so Bayer phases stay consistent inside
/// a band.
pub fn row_bands(rows: usize, threads: usize, align: usize) -> Vec<std::ops::Range<usize>> {
    let threads = resolve_threads(threads);
    let align = align.max(1);
    if rows == 0 {
        return Vec::new();
    }
    let aligned_units = rows.div_ceil(align);
    let bands = threads.min(aligned_units);
    let units_per_band = aligned_units.div_ceil(bands);
    let mut ranges = Vec::with_capacity(bands);
    let mut start = 0;
    while start < rows {
        let end = (start + units_per_band * align).min(rows);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Run `kernel` over every band of `0..rows` and collect the results in band
/// order. A single band runs inline without spawning.
pub fn map_row_bands<R: Send>(
    rows: usize,
    threads: usize,
    align: usize,
    kernel: impl Fn(std::ops::Range<usize>) -> R + Sync,
) -> Vec<R> {
    let bands = row_bands(rows, threads, align);
    if bands.len() <= 1 {
        return bands.into_iter().map(&kernel).collect();
    }
    thread::scope(|scope| {
        let handles = bands
            .into_iter()
            .map(|band| {
                let kernel = &kernel;
                scope.spawn(move || kernel(band))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("row-band worker panicked"))
            .collect()
    })
}

/// Run `kernel` over disjoint mutable row bands of `buffer`, where each row is
/// `row_len` elements. Results are returned in band order.
pub fn map_row_bands_mut<T: Send, R: Send>(
    buffer: &mut [T],
    row_len: usize,
    threads: usize,
    align: usize,
    kernel: impl Fn(std::ops::Range<usize>, &mut [T]) -> R + Sync,
) -> Vec<R> {
    let rows = buffer.len().checked_div(row_len).unwrap_or(0);
    let bands = row_bands(rows, threads, align);
    if bands.len() <= 1 {
        return bands
            .into_iter()
            .map(|band| {
                let slice = &mut buffer[band.start * row_len..band.end * row_len];
                kernel(band, slice)
            })
            .collect();
    }
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(bands.len());
        let mut rest = buffer;
        let mut consumed = 0;
        for band in bands {
            let (slice, tail) = rest.split_at_mut((band.end - consumed) * row_len);
            consumed = band.end;
            rest = tail;
            let kernel = &kernel;
            handles.push(scope.spawn(move || kernel(band, slice)));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("row-band worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_rows_exactly_and_respect_alignment() {
        let bands = row_bands(10, 4, 2);
        assert_eq!(bands, vec![0..4, 4..8, 8..10]);
        assert_eq!(row_bands(3, 8, 1), vec![0..1, 1..2, 2..3]);
        assert_eq!(row_bands(0, 4, 2), Vec::<std::ops::Range<usize>>::new());
        assert_eq!(row_bands(7, 1, 2), vec![0..7]);
        assert_eq!(row_bands(64, 0, 2).len(), default_threads().min(32));
    }

    #[test]
    fn mutable_bands_see_disjoint_rows() {
        let mut buffer = vec![0u32; 6 * 4];
        let sums = map_row_bands_mut(&mut buffer, 4, 3, 2, |band, slice| {
            for (offset, value) in slice.iter_mut().enumerate() {
                *value = (band.start * 4 + offset) as u32;
            }
            band.len()
        });
        assert_eq!(sums, vec![2, 2, 2]);
        assert_eq!(buffer, (0..24).collect::<Vec<u32>>());
    }
}
