use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use flate2::read::ZlibDecoder;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"LELR";
const FILE_HEADER_SIZE: u64 = 32;
const MAP_HEADER_SIZE: u64 = 20;

#[derive(Clone, Debug, Serialize)]
pub struct MapRecord {
    pub index: usize,
    pub header_offset: u64,
    pub compressed_offset: u64,
    pub compressed_length: u32,
    pub width: usize,
    pub height: usize,
    pub unknown_u64: u64,
}

#[derive(Clone, Debug)]
pub struct HotpixelRec {
    path: PathBuf,
    pub sha256: String,
    pub total_size: u64,
    pub footer_offset: u64,
    pub footer_length: u64,
    pub records: Vec<MapRecord>,
}

impl HotpixelRec {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut reader =
            BufReader::new(File::open(&path).with_context(|| format!("open {}", path.display()))?);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("{} is not a Light LELR hotpixel record", path.display());
        }
        let total_size = reader.read_u64::<LittleEndian>()?;
        let footer_offset = reader.read_u64::<LittleEndian>()?;
        let footer_length = reader.read_u64::<LittleEndian>()?;
        let actual_size = reader.get_ref().metadata()?.len();

        if total_size != actual_size {
            bail!("hotpixel.rec size mismatch: header={total_size}, actual={actual_size}");
        }
        if footer_offset + footer_length != actual_size {
            bail!("hotpixel.rec footer bounds are inconsistent");
        }

        let mut records = Vec::new();
        let mut offset = FILE_HEADER_SIZE;
        while offset < footer_offset {
            reader.seek(SeekFrom::Start(offset))?;
            let unknown_u64 = reader.read_u64::<LittleEndian>()?;
            let compressed_length = reader.read_u32::<LittleEndian>()?;
            let width = reader.read_u32::<LittleEndian>()? as usize;
            let height = reader.read_u32::<LittleEndian>()? as usize;
            let compressed_offset = offset + MAP_HEADER_SIZE;
            let end = compressed_offset + compressed_length as u64;
            if end > footer_offset {
                bail!("hotpixel record {} crosses the footer", records.len());
            }
            records.push(MapRecord {
                index: records.len(),
                header_offset: offset,
                compressed_offset,
                compressed_length,
                width,
                height,
                unknown_u64,
            });
            offset = end;
        }
        if offset != footer_offset {
            bail!("hotpixel records do not end at the declared footer");
        }

        let sha256 = sha256_file(&path)?;
        Ok(Self {
            path,
            sha256,
            total_size,
            footer_offset,
            footer_length,
            records,
        })
    }

    pub fn load_rotated_map(
        &self,
        record_index: usize,
        expected_width: usize,
        expected_height: usize,
    ) -> Result<Vec<u8>> {
        let record = self.records.get(record_index).with_context(|| {
            format!(
                "camera maps to hotpixel record {record_index}, but the file has {} records",
                self.records.len()
            )
        })?;
        if record.width != expected_width || record.height != expected_height {
            bail!(
                "hotpixel record {record_index} is {}x{}, camera RAW is {}x{}",
                record.width,
                record.height,
                expected_width,
                expected_height
            );
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(record.compressed_offset))?;
        let compressed = file.take(record.compressed_length as u64);
        let mut decoder = ZlibDecoder::new(compressed);
        let expected = record
            .width
            .checked_mul(record.height)
            .context("hotpixel map dimensions overflow")?;
        let mut raw = Vec::with_capacity(expected);
        decoder.read_to_end(&mut raw)?;
        if raw.len() != expected {
            bail!(
                "hotpixel record {record_index} expands to {} bytes; expected {expected}",
                raw.len()
            );
        }

        // Factory maps align to decoded Light RAW after a 180-degree rotation.
        raw.reverse();
        Ok(raw)
    }
}

/// Write a `hotpixel.rec` container from raw severity maps.
///
/// `maps` are `(width, height, row-major severity bytes)` in record order
/// (`A1..A5, B1..B5, C1..C6`) in the factory orientation, that is, *before*
/// the 180-degree rotation `load_rotated_map` applies. The footer is opaque to
/// this crate and is written as zeros. Intended for tests and tooling.
pub fn write_hotpixel_rec(path: impl AsRef<Path>, maps: &[(usize, usize, Vec<u8>)]) -> Result<()> {
    use byteorder::WriteBytesExt;
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;

    let mut records = Vec::new();
    for (index, (width, height, severity)) in maps.iter().enumerate() {
        if severity.len() != width * height {
            bail!(
                "record {index} has {} bytes; expected {width}x{height}",
                severity.len()
            );
        }
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(severity)?;
        let compressed = encoder.finish()?;
        records.push((*width, *height, compressed));
    }
    let footer = vec![0u8; 16];
    let footer_offset = FILE_HEADER_SIZE
        + records
            .iter()
            .map(|(_, _, compressed)| MAP_HEADER_SIZE + compressed.len() as u64)
            .sum::<u64>();
    let total = footer_offset + footer.len() as u64;

    let path = path.as_ref();
    let mut output = std::io::BufWriter::new(
        File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    output.write_all(MAGIC)?;
    output.write_u64::<LittleEndian>(total)?;
    output.write_u64::<LittleEndian>(footer_offset)?;
    output.write_u64::<LittleEndian>(footer.len() as u64)?;
    output.write_u32::<LittleEndian>(0)?;
    for (width, height, compressed) in &records {
        output.write_u64::<LittleEndian>(0)?;
        output.write_u32::<LittleEndian>(compressed.len() as u32)?;
        output.write_u32::<LittleEndian>(*width as u32)?;
        output.write_u32::<LittleEndian>(*height as u32)?;
        output.write_all(compressed)?;
    }
    output.write_all(&footer)?;
    output.flush()?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn written_records_read_back_in_rotated_raw_order() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hotpixel.rec");
        let maps = (0..16)
            .map(|record| {
                (
                    4usize,
                    2usize,
                    (0..8).map(|i| (record * 8 + i) as u8).collect(),
                )
            })
            .collect::<Vec<_>>();
        write_hotpixel_rec(&path, &maps).unwrap();

        let rec = HotpixelRec::open(&path).unwrap();
        assert_eq!(rec.records.len(), 16);
        assert_eq!(rec.total_size, std::fs::metadata(&path).unwrap().len());
        let mut expected = maps[5].2.clone();
        expected.reverse();
        assert_eq!(rec.load_rotated_map(5, 4, 2).unwrap(), expected);
        assert!(rec.load_rotated_map(5, 2, 4).is_err());
    }

    #[test]
    fn parses_and_rotates_a_synthetic_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hotpixel.rec");
        let pixels = vec![1u8, 2, 3, 4, 5, 6];
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&pixels).unwrap();
        let compressed = encoder.finish().unwrap();
        let footer = vec![9u8; 12];
        let footer_offset = 32 + 20 + compressed.len() as u64;
        let total = footer_offset + footer.len() as u64;

        let mut output = File::create(&path).unwrap();
        output.write_all(MAGIC).unwrap();
        output.write_u64::<LittleEndian>(total).unwrap();
        output.write_u64::<LittleEndian>(footer_offset).unwrap();
        output
            .write_u64::<LittleEndian>(footer.len() as u64)
            .unwrap();
        output.write_u32::<LittleEndian>(0).unwrap();
        output.write_u64::<LittleEndian>(0).unwrap();
        output
            .write_u32::<LittleEndian>(compressed.len() as u32)
            .unwrap();
        output.write_u32::<LittleEndian>(3).unwrap();
        output.write_u32::<LittleEndian>(2).unwrap();
        output.write_all(&compressed).unwrap();
        output.write_all(&footer).unwrap();
        drop(output);

        let rec = HotpixelRec::open(&path).unwrap();
        assert_eq!(rec.records.len(), 1);
        assert_eq!(
            rec.load_rotated_map(0, 3, 2).unwrap(),
            vec![6, 5, 4, 3, 2, 1]
        );
    }
}
