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

    let mut output = Vec::with_capacity(sample_count);
    for chunk in packed.rchunks_exact(5) {
        let word = ((chunk[4] as u64) << 32)
            | ((chunk[3] as u64) << 24)
            | ((chunk[2] as u64) << 16)
            | ((chunk[1] as u64) << 8)
            | chunk[0] as u64;
        output.push(((word >> 30) & 1023) as u16);
        output.push(((word >> 20) & 1023) as u16);
        output.push(((word >> 10) & 1023) as u16);
        output.push((word & 1023) as u16);
    }
    Ok(output)
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
    }
}
