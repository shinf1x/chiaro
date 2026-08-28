use anyhow::{Result, bail};

/// Decode Light L16's reversed packed-RAW10 byte stream.
///
/// Light stores the complete sequence of five-byte groups in reverse order,
/// while each reconstructed 40-bit group contains four big-endian 10-bit samples.
pub fn unpack_l16_10bit(packed: &[u8], sample_count: usize) -> Result<Vec<u16>> {
    if !sample_count.is_multiple_of(4) || packed.len() != sample_count / 4 * 5 {
        bail!(
            "RAW10 length mismatch: {} bytes for {} samples",
            packed.len(),
            sample_count
        );
    }

    let mut output = vec![0u16; sample_count];
    unpack_groups(packed, &mut output);
    Ok(output)
}

/// [`unpack_l16_10bit`] for a `width * height` plane with rows split across
/// `threads` workers (`0` = all cores). Rows are a whole number of groups
/// because `width` is a multiple of four on every L16 sensor.
pub fn unpack_l16_10bit_threaded(
    packed: &[u8],
    width: usize,
    height: usize,
    threads: usize,
) -> Result<Vec<u16>> {
    let sample_count = width * height;
    if !width.is_multiple_of(4) {
        return unpack_l16_10bit(packed, sample_count);
    }
    if packed.len() != sample_count / 4 * 5 {
        bail!(
            "RAW10 length mismatch: {} bytes for {} samples",
            packed.len(),
            sample_count
        );
    }
    let row_bytes = width / 4 * 5;
    let mut output = vec![0u16; sample_count];
    crate::parallel::map_row_bands_mut(&mut output, width, threads, 1, |rows, band| {
        // Groups are stored in reverse: the first samples live at the end.
        let end = packed.len() - rows.start * row_bytes;
        let start = packed.len() - rows.end * row_bytes;
        unpack_groups(&packed[start..end], band);
    });
    Ok(output)
}

/// Decode reversed five-byte groups into `output`, which holds exactly four
/// samples per group.
fn unpack_groups(packed: &[u8], output: &mut [u16]) {
    debug_assert_eq!(packed.len() / 5 * 4, output.len());
    for (chunk, samples) in packed
        .rchunks_exact(5)
        .zip(output.as_chunks_mut::<4>().0.iter_mut())
    {
        let word = ((chunk[4] as u64) << 32)
            | ((chunk[3] as u64) << 24)
            | ((chunk[2] as u64) << 16)
            | ((chunk[1] as u64) << 8)
            | chunk[0] as u64;
        samples[0] = ((word >> 30) & 1023) as u16;
        samples[1] = ((word >> 20) & 1023) as u16;
        samples[2] = ((word >> 10) & 1023) as u16;
        samples[3] = (word & 1023) as u16;
    }
}

#[cfg(test)]
pub fn pack_l16_10bit(samples: &[u16]) -> Result<Vec<u8>> {
    if !samples.len().is_multiple_of(4) {
        bail!("RAW10 sample count must be divisible by four");
    }
    let mut forward = Vec::with_capacity(samples.len() / 4 * 5);
    for group in samples.chunks_exact(4) {
        if group.iter().any(|&value| value > 1023) {
            bail!("RAW10 sample exceeds 1023");
        }
        let word = ((group[0] as u64) << 30)
            | ((group[1] as u64) << 20)
            | ((group[2] as u64) << 10)
            | group[3] as u64;
        forward.extend_from_slice(&[
            (word >> 32) as u8,
            (word >> 24) as u8,
            (word >> 16) as u8,
            (word >> 8) as u8,
            word as u8,
        ]);
    }
    forward.reverse();
    Ok(forward)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw10_roundtrip() {
        let samples: Vec<u16> = (0..400)
            .map(|index| ((index * 37 + 11) & 1023) as u16)
            .collect();
        let packed = pack_l16_10bit(&samples).unwrap();
        assert_eq!(unpack_l16_10bit(&packed, samples.len()).unwrap(), samples);
        for threads in [1, 3, 7] {
            assert_eq!(
                unpack_l16_10bit_threaded(&packed, 8, 50, threads).unwrap(),
                samples,
                "threads={threads}"
            );
        }
        assert!(unpack_l16_10bit_threaded(&packed, 8, 49, 2).is_err());
    }
}
