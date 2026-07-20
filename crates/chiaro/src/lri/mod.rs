//! Minimal, read-only Light L16 LRI support for gallery previews.
//!
//! The parser deliberately extracts only container framing, the embedded
//! reference-camera choice, orientation, and captured RAW surface metadata.
//! Keeping that boundary small makes this module suitable as the source side
//! of future exporters without coupling file processing to the UI.

use std::{collections::HashMap, fs::File, ops::Range, path::Path};

use chiaro_proto::{
    Message,
    camera_module::{CameraModule, camera_module::Surface},
    color_calibration::ColorCalibration as ProtoColorCalibration,
    hw_info::HwInfo,
    lightheader::{FactoryModuleCalibration, LightHeader, SensorData},
    matrix3x3f::Matrix3x3F,
    point2i::Point2I,
    sensor_characterization::SensorCharacterization,
    time_stamp::TimeStamp,
    view_preferences::{ViewPreferences, view_preferences::ChannelGain},
};
use memmap2::MmapOptions;
use thiserror::Error;

const LELR_HEADER_SIZE: usize = 32;
const RAW_BAYER_JPEG: u64 = 0;
const RAW_PACKED_10BPP: u64 = 7;
const BAYER_JPEG_HEADER_SIZE: usize = 1576;
const CAMERA_NAMES: [&str; 16] = [
    "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "C6",
];

#[derive(Debug, Error)]
pub enum LriError {
    #[error("could not open {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid LRI: {0}")]
    Format(String),
    #[error("this LRI does not declare an image reference camera")]
    MissingReferenceCamera,
    #[error("reference camera {0} has no captured image in this LRI")]
    MissingReferenceImage(String),
    #[error("camera {0} has no captured image in this LRI")]
    MissingCamera(String),
    #[error("reference camera {camera} uses unsupported RAW format {format}")]
    UnsupportedRaw { camera: String, format: u64 },
}

#[derive(Debug)]
pub struct PreviewImage {
    pub size: [usize; 2],
    pub rgb: Vec<u8>,
    pub camera: String,
    pub color_calibrated: bool,
    pub metadata: CaptureMetadata,
}

#[derive(Clone, Debug)]
pub struct CameraSummary {
    pub camera: String,
    pub dimensions: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct CaptureSummary {
    pub reference_camera: String,
    pub cameras: Vec<CameraSummary>,
    pub metadata: CaptureMetadata,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CaptureMetadata {
    pub iso: Option<u32>,
    pub shutter_ns: Option<u64>,
    pub focal_length_mm: Option<i32>,
    pub night_mode: bool,
    pub tripod: Option<bool>,
    pub captured_at: Option<CaptureDateTime>,
    pub orientation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDateTime {
    pub year: u32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub timezone_offset_minutes: Option<i32>,
}

/// Color-filter layout for a captured RAW surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensorPattern {
    Rggb,
    Grbg,
    Gbrg,
    Bggr,
    Mono,
}

impl SensorPattern {
    pub fn parse(value: &str) -> Result<Self, LriError> {
        match value.to_ascii_uppercase().as_str() {
            "RGGB" => Ok(Self::Rggb),
            "GRBG" => Ok(Self::Grbg),
            "GBRG" => Ok(Self::Gbrg),
            "BGGR" => Ok(Self::Bggr),
            "MONO" => Ok(Self::Mono),
            other => Err(format_error(format!("unsupported sensor pattern {other}"))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rggb => "RGGB",
            Self::Grbg => "GRBG",
            Self::Gbrg => "GBRG",
            Self::Bggr => "BGGR",
            Self::Mono => "MONO",
        }
    }

    pub fn color_at(self, row: usize, column: usize) -> usize {
        if self == Self::Mono {
            return 0;
        }
        let colors = match self {
            Self::Rggb => [[0, 1], [1, 2]],
            Self::Grbg => [[1, 0], [2, 1]],
            Self::Gbrg => [[1, 2], [0, 1]],
            Self::Bggr => [[2, 1], [1, 0]],
            Self::Mono => unreachable!(),
        };
        colors[row & 1][column & 1]
    }
}

/// One tightly-addressable packed RAW10 surface inside an LRI container.
#[derive(Clone, Debug)]
pub struct RawCamera {
    pub id: usize,
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub absolute_offset: usize,
    pub byte_len: usize,
    pub pattern: SensorPattern,
}

#[derive(Clone, Debug)]
pub struct CaptureLayout {
    pub cameras: Vec<RawCamera>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LelrBlockDescriptor {
    pub block_offset: usize,
    pub block_length: usize,
    pub message_start: usize,
    pub message_length: usize,
    pub message_type: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReferencePayloadRead {
    /// Fetch this small header first; it contains the compressed plane sizes.
    BayerJpegHeader(Range<usize>),
    /// Fetch this complete range to decode the reference camera.
    Payload(Range<usize>),
}

impl LelrBlockDescriptor {
    pub fn block_range(&self) -> Range<usize> {
        self.block_offset..self.block_offset + self.block_length
    }

    pub fn message_range(&self) -> Range<usize> {
        self.message_start..self.message_start + self.message_length
    }
}

#[derive(Clone, Debug)]
struct ColorCalibration {
    illuminant: u64,
    forward_matrix: [f32; 9],
    rg_ratio: f32,
    bg_ratio: f32,
}

#[derive(Clone, Debug, Default)]
struct CapturedModule {
    camera: u64,
    enabled: bool,
    width: usize,
    height: usize,
    format: u64,
    row_stride: usize,
    payload_offset: usize,
    payload_limit: usize,
    bayer_red: Option<(i32, i32)>,
    frame_index: u64,
    analog_gain: f32,
    digital_gain: f32,
    sensor_exposure_ns: u64,
}

#[derive(Default)]
struct Capture {
    reference_camera: Option<u64>,
    orientation: u64,
    awb_gains: Option<[f32; 3]>,
    modules: Vec<CapturedModule>,
    colors: HashMap<u64, Vec<ColorCalibration>>,
    sensor_types: HashMap<u64, u64>,
    sensor_levels: HashMap<u64, (f32, f32)>,
    focal_length_mm: Option<i32>,
    captured_at: Option<CaptureDateTime>,
    image_gain: Option<f32>,
    image_integration_time_ns: Option<u64>,
    scene_mode: Option<u64>,
    tripod: Option<bool>,
}

pub fn inspect_capture(path: &Path) -> Result<CaptureSummary, LriError> {
    let (_data, capture) = open_capture(path)?;
    capture_summary(&capture)
}

/// Inspect an LRI already held in memory, as used by PTP/MTP sources.
pub fn inspect_capture_bytes(data: &[u8]) -> Result<CaptureSummary, LriError> {
    let capture = parse_capture(data)?;
    capture_summary(&capture)
}

/// Locate the packed RAW10 surfaces used by processing tools.
///
/// This is the shared, UI-independent layout API used by the hot-pixel binary.
/// Pattern overrides are keyed by camera name (`A1` through `C6`). Repeated
/// night-mode frames select the first frame for each physical camera.
pub fn parse_raw_layout(
    data: &[u8],
    pattern_overrides: &HashMap<String, SensorPattern>,
) -> Result<CaptureLayout, LriError> {
    let capture = parse_capture(data)?;
    let mut modules = capture
        .modules
        .iter()
        .filter(|module| module.enabled && module.format == RAW_PACKED_10BPP)
        .collect::<Vec<_>>();
    modules.sort_by_key(|module| (module.camera, module.frame_index));
    modules.dedup_by_key(|module| module.camera);

    let mut cameras = Vec::with_capacity(modules.len());
    for module in modules {
        let Some(name) = CAMERA_NAMES.get(module.camera as usize).copied() else {
            return Err(format_error(format!(
                "invalid Light L16 camera id {}",
                module.camera
            )));
        };
        let byte_len = module
            .row_stride
            .checked_mul(module.height)
            .ok_or_else(|| format_error("RAW byte span overflow"))?;
        module
            .payload_offset
            .checked_add(byte_len)
            .filter(|end| *end <= module.payload_limit && *end <= data.len())
            .ok_or_else(|| format_error(format!("RAW span for {name} escapes its LRI block")))?;

        let sensor = capture
            .sensor_types
            .get(&module.camera)
            .copied()
            .unwrap_or(0);
        let pattern = if let Some(pattern) = pattern_overrides.get(name) {
            *pattern
        } else if matches!(sensor, 3 | 5) {
            SensorPattern::Mono
        } else if let Some((red_x, red_y)) = module.bayer_red {
            match (red_x & 1, red_y & 1) {
                (0, 0) => SensorPattern::Rggb,
                (1, 0) => SensorPattern::Grbg,
                (0, 1) => SensorPattern::Gbrg,
                _ => SensorPattern::Bggr,
            }
        } else {
            return Err(format_error(format!(
                "no Bayer pattern is available for {name}; pass --pattern {name}=RGGB (or MONO)"
            )));
        };
        cameras.push(RawCamera {
            id: module.camera as usize,
            name: name.to_owned(),
            width: module.width,
            height: module.height,
            row_stride: module.row_stride,
            absolute_offset: module.payload_offset,
            byte_len,
            pattern,
        });
    }

    cameras.sort_by_key(|camera| camera.absolute_offset);
    for pair in cameras.windows(2) {
        if pair[0].absolute_offset + pair[0].byte_len > pair[1].absolute_offset {
            return Err(format_error("overlapping RAW spans in LightHeader"));
        }
    }
    Ok(CaptureLayout { cameras })
}

/// Inspect metadata messages obtained through sparse remote reads, without
/// allocating zero-filled buffers for the intervening RAW payloads.
pub fn inspect_capture_metadata_blocks(
    blocks: &[(u8, usize, usize, &[u8])],
) -> Result<CaptureMetadata, LriError> {
    let mut capture = Capture::default();
    for (message_type, block_offset, block_end, message) in blocks {
        match message_type {
            0 => parse_light_header(message, *block_offset, *block_end, &mut capture)?,
            1 => parse_view_preferences(message, &mut capture)?,
            _ => {}
        }
    }
    let reference = capture
        .reference_camera
        .ok_or(LriError::MissingReferenceCamera)?;
    Ok(capture_metadata(&capture, reference))
}

/// Inspect all complete LELR blocks currently available in a streamed prefix.
/// Returns `None` until the reference camera has been declared.
pub fn inspect_capture_prefix(data: &[u8]) -> Result<Option<CaptureSummary>, LriError> {
    let capture = parse_capture_prefix(data)?;
    if capture.reference_camera.is_none() {
        return Ok(None);
    }
    capture_summary(&capture).map(Some)
}

/// Return the byte boundary after the last complete LELR block in `data`.
///
/// Streaming callers can use this to run the more expensive protobuf and image
/// parsing only when another full block has arrived, rather than after every
/// small USB packet belonging to the same block.
pub fn complete_lelr_prefix_len(data: &[u8]) -> Result<usize, LriError> {
    let mut block_offset = 0usize;
    let mut block_index = 0usize;
    while block_offset < data.len() {
        let Some(header) = data.get(block_offset..block_offset + LELR_HEADER_SIZE) else {
            break;
        };
        if &header[0..4] != b"LELR" {
            return Err(format_error(format!(
                "bad block magic at 0x{block_offset:x}"
            )));
        }
        let block_length = read_u64(&header[4..12])? as usize;
        if block_length < LELR_HEADER_SIZE {
            return Err(format_error(format!(
                "block {block_index} has invalid length {block_length}"
            )));
        }
        let Some(block_end) = block_offset.checked_add(block_length) else {
            return Err(format_error(format!(
                "block {block_index} length overflows the file"
            )));
        };
        if block_end > data.len() {
            break;
        }
        block_offset = block_end;
        block_index += 1;
    }
    Ok(block_offset)
}

/// Parse one 32-byte LELR header without reading its payload.
pub fn inspect_lelr_block_header(
    header: &[u8],
    block_offset: usize,
) -> Result<LelrBlockDescriptor, LriError> {
    let header = header
        .get(..LELR_HEADER_SIZE)
        .ok_or_else(|| format_error("truncated LELR block header"))?;
    if &header[0..4] != b"LELR" {
        return Err(format_error(format!(
            "bad block magic at 0x{block_offset:x}"
        )));
    }
    let block_length = read_u64(&header[4..12])? as usize;
    let relative_message = read_u64(&header[12..20])? as usize;
    let message_length = read_u32(&header[20..24])? as usize;
    if block_length < LELR_HEADER_SIZE {
        return Err(format_error(format!(
            "block at 0x{block_offset:x} has invalid length {block_length}"
        )));
    }
    let block_end = block_offset
        .checked_add(block_length)
        .ok_or_else(|| format_error("LELR block length overflow"))?;
    let message_start = block_offset
        .checked_add(relative_message)
        .filter(|start| *start <= block_end)
        .ok_or_else(|| format_error("invalid LELR message offset"))?;
    message_start
        .checked_add(message_length)
        .filter(|end| *end <= block_end)
        .ok_or_else(|| format_error("LELR message escapes its block"))?;
    Ok(LelrBlockDescriptor {
        block_offset,
        block_length,
        message_start,
        message_length,
        message_type: header[24],
    })
}

/// Locate the complete block containing the declared reference camera.
///
/// `data` may contain zero-filled payloads as long as every block header and
/// protobuf message is present. This supports sparse camera reads: metadata can
/// be inspected before downloading the selected RAW payload.
pub fn reference_camera_block_range(data: &[u8]) -> Result<Option<Range<usize>>, LriError> {
    let capture = parse_capture(data)?;
    let Some(reference) = capture.reference_camera else {
        return Ok(None);
    };
    let Some(module) = capture
        .modules
        .iter()
        .find(|module| module.enabled && module.camera == reference)
    else {
        return Ok(None);
    };
    let mut block_offset = 0usize;
    while block_offset < data.len() {
        let descriptor = inspect_lelr_block_header(&data[block_offset..], block_offset)?;
        let range = descriptor.block_range();
        if range.end == module.payload_limit {
            return Ok(Some(range));
        }
        block_offset = range.end;
    }
    Ok(None)
}

/// Determine the minimal remote range needed for the reference camera payload.
///
/// Packed RAW surfaces return their complete byte range immediately. Bayer-JPEG
/// surfaces first request their fixed-size header; after the caller fills that
/// range and calls again, the four encoded plane lengths determine the final
/// payload range.
pub fn reference_camera_payload_read(
    data: &[u8],
) -> Result<Option<ReferencePayloadRead>, LriError> {
    let capture = parse_capture(data)?;
    let Some(reference) = capture.reference_camera else {
        return Ok(None);
    };
    let Some(module) = capture
        .modules
        .iter()
        .find(|module| module.enabled && module.camera == reference)
    else {
        return Ok(None);
    };
    match module.format {
        RAW_PACKED_10BPP => {
            let pixel_count = module
                .width
                .checked_mul(module.height)
                .ok_or_else(|| format_error("RAW dimensions overflow"))?;
            let packed_len = pixel_count
                .checked_mul(10)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| format_error("RAW dimensions overflow"))?;
            let end = module
                .payload_offset
                .checked_add(packed_len)
                .filter(|end| *end <= module.payload_limit)
                .ok_or_else(|| format_error("packed RAW payload escapes its LELR block"))?;
            Ok(Some(ReferencePayloadRead::Payload(
                module.payload_offset..end,
            )))
        }
        RAW_BAYER_JPEG => {
            let header_end = module
                .payload_offset
                .checked_add(BAYER_JPEG_HEADER_SIZE)
                .filter(|end| *end <= module.payload_limit)
                .ok_or_else(|| format_error("Bayer-JPEG header escapes its LELR block"))?;
            let header_range = module.payload_offset..header_end;
            let header = &data[header_range.clone()];
            if &header[..4] != b"BJPG" {
                return Ok(Some(ReferencePayloadRead::BayerJpegHeader(header_range)));
            }
            let encoded_len = [8..12, 12..16, 16..20, 20..24].into_iter().try_fold(
                BAYER_JPEG_HEADER_SIZE,
                |total, range| {
                    total
                        .checked_add(read_u32(&header[range])? as usize)
                        .ok_or_else(|| format_error("Bayer-JPEG payload length overflow"))
                },
            )?;
            let end = module
                .payload_offset
                .checked_add(encoded_len)
                .filter(|end| *end <= module.payload_limit)
                .ok_or_else(|| format_error("Bayer-JPEG payload escapes its LELR block"))?;
            Ok(Some(ReferencePayloadRead::Payload(
                module.payload_offset..end,
            )))
        }
        format => Err(LriError::UnsupportedRaw {
            camera: camera_name(reference),
            format,
        }),
    }
}

/// Whether a sparse reference-camera preview has enough metadata for its final
/// color treatment. Mono cameras need no matrix; Bayer cameras wait until a
/// matching calibration profile has been encountered.
pub fn reference_preview_color_ready(data: &[u8]) -> Result<Option<bool>, LriError> {
    let capture = parse_capture(data)?;
    let Some(reference) = capture.reference_camera else {
        return Ok(None);
    };
    let Some(module) = capture
        .modules
        .iter()
        .find(|module| module.enabled && module.camera == reference)
    else {
        return Ok(None);
    };
    Ok(Some(
        !valid_bayer(module.bayer_red)
            || preferred_color_calibration(&capture, reference).is_some(),
    ))
}

fn capture_summary(capture: &Capture) -> Result<CaptureSummary, LriError> {
    let reference = capture
        .reference_camera
        .ok_or(LriError::MissingReferenceCamera)?;
    let mut cameras = capture
        .modules
        .iter()
        .filter(|module| {
            module.enabled && matches!(module.format, RAW_BAYER_JPEG | RAW_PACKED_10BPP)
        })
        .map(|module| CameraSummary {
            camera: camera_name(module.camera),
            dimensions: [module.width, module.height],
        })
        .collect::<Vec<_>>();
    cameras.sort_by_key(|camera| camera_index(&camera.camera).unwrap_or(u64::MAX));
    cameras.dedup_by(|left, right| left.camera == right.camera);
    Ok(CaptureSummary {
        reference_camera: camera_name(reference),
        cameras,
        metadata: capture_metadata(capture, reference),
    })
}

fn capture_metadata(capture: &Capture, reference: u64) -> CaptureMetadata {
    let module = capture
        .modules
        .iter()
        .find(|module| module.enabled && module.camera == reference);
    let gain = capture.image_gain.or_else(|| {
        module.map(|module| module.analog_gain.max(0.0) * module.digital_gain.max(0.0))
    });
    let shutter_ns = capture
        .image_integration_time_ns
        .or_else(|| module.map(|module| module.sensor_exposure_ns))
        .filter(|value| *value > 0);
    CaptureMetadata {
        iso: gain
            .filter(|gain| gain.is_finite() && *gain > 0.0)
            .map(|gain| (gain * 100.0).round().clamp(1.0, u32::MAX as f32) as u32),
        shutter_ns,
        focal_length_mm: capture.focal_length_mm,
        night_mode: capture.scene_mode == Some(4)
            || capture
                .modules
                .iter()
                .any(|module| module.enabled && module.format == RAW_BAYER_JPEG),
        tripod: capture.tripod,
        captured_at: capture.captured_at.clone(),
        orientation: capture.orientation,
    }
}

/// Decode the LRI's embedded reference camera directly to an RGB thumbnail.
///
/// The source file is memory-mapped, and only pixels needed for the requested
/// preview are unpacked. Nothing is persisted beside the source LRI.
pub fn decode_reference_preview(path: &Path, max_edge: usize) -> Result<PreviewImage, LriError> {
    let (data, capture) = open_capture(path)?;
    let camera = capture
        .reference_camera
        .ok_or(LriError::MissingReferenceCamera)?;
    decode_from_capture(&data, &capture, camera, max_edge, true)
}

/// Decode the reference camera from an LRI already held in memory.
pub fn decode_reference_preview_bytes(
    data: &[u8],
    max_edge: usize,
) -> Result<PreviewImage, LriError> {
    let capture = parse_capture(data)?;
    let camera = capture
        .reference_camera
        .ok_or(LriError::MissingReferenceCamera)?;
    decode_from_capture(data, &capture, camera, max_edge, true)
}

/// Try to decode the reference camera from a prefix of an LRI stream.
///
/// Only complete LELR blocks are inspected. `Ok(None)` means the prefix does
/// not yet contain both the reference-camera metadata and its complete image
/// payload, so a caller can fetch another transport window and try again.
pub fn try_decode_reference_preview_prefix(
    data: &[u8],
    max_edge: usize,
) -> Result<Option<PreviewImage>, LriError> {
    let capture = parse_capture_prefix(data)?;
    let Some(camera) = capture.reference_camera else {
        return Ok(None);
    };
    if !capture
        .modules
        .iter()
        .any(|module| module.enabled && module.camera == camera)
    {
        return Ok(None);
    }
    decode_from_capture(data, &capture, camera, max_edge, true).map(Some)
}

/// Try to decode a camera from the complete LELR blocks in a streamed prefix.
pub fn try_decode_camera_preview_prefix(
    data: &[u8],
    camera: &str,
    max_edge: usize,
) -> Result<Option<PreviewImage>, LriError> {
    let camera_id =
        camera_index(camera).ok_or_else(|| LriError::MissingCamera(camera.to_owned()))?;
    let capture = parse_capture_prefix(data)?;
    if !capture
        .modules
        .iter()
        .any(|module| module.enabled && module.camera == camera_id)
    {
        return Ok(None);
    }
    decode_from_capture(data, &capture, camera_id, max_edge, false).map(Some)
}

pub fn decode_camera_preview(
    path: &Path,
    camera: &str,
    max_edge: usize,
) -> Result<PreviewImage, LriError> {
    let camera_id =
        camera_index(camera).ok_or_else(|| LriError::MissingCamera(camera.to_owned()))?;
    let (data, capture) = open_capture(path)?;
    decode_from_capture(&data, &capture, camera_id, max_edge, false)
}

/// Decode one camera from an LRI already held in memory.
pub fn decode_camera_preview_bytes(
    data: &[u8],
    camera: &str,
    max_edge: usize,
) -> Result<PreviewImage, LriError> {
    let camera_id =
        camera_index(camera).ok_or_else(|| LriError::MissingCamera(camera.to_owned()))?;
    let capture = parse_capture(data)?;
    decode_from_capture(data, &capture, camera_id, max_edge, false)
}

fn open_capture(path: &Path) -> Result<(memmap2::Mmap, Capture), LriError> {
    let file = File::open(path).map_err(|source| LriError::Open {
        path: path.display().to_string(),
        source,
    })?;
    // SAFETY: the mapping is read-only and remains owned for the full decode.
    let data = unsafe { MmapOptions::new().map(&file) }.map_err(|source| LriError::Open {
        path: path.display().to_string(),
        source,
    })?;

    let capture = parse_capture(&data)?;
    Ok((data, capture))
}

fn decode_from_capture(
    data: &[u8],
    capture: &Capture,
    camera: u64,
    max_edge: usize,
    is_reference: bool,
) -> Result<PreviewImage, LriError> {
    let camera_name = camera_name(camera);
    let module = capture
        .modules
        .iter()
        .filter(|module| module.enabled && module.camera == camera)
        .min_by_key(|module| module.frame_index)
        .ok_or_else(|| {
            if is_reference {
                LriError::MissingReferenceImage(camera_name.clone())
            } else {
                LriError::MissingCamera(camera_name.clone())
            }
        })?;
    if !matches!(module.format, RAW_BAYER_JPEG | RAW_PACKED_10BPP) {
        return Err(LriError::UnsupportedRaw {
            camera: camera_name,
            format: module.format,
        });
    }

    let sensor_levels = capture
        .sensor_types
        .get(&camera)
        .and_then(|sensor| capture.sensor_levels.get(sensor))
        .copied()
        .unwrap_or((42.0, 1023.0));
    let color = preferred_color_calibration(capture, camera);
    let sensor_levels = if module.format == RAW_BAYER_JPEG {
        (sensor_levels.0.min(254.0), 255.0)
    } else {
        sensor_levels
    };
    let (width, height, rgb) = render_preview(
        data,
        module,
        capture.orientation,
        max_edge,
        sensor_levels,
        color,
        capture.awb_gains,
    )?;
    Ok(PreviewImage {
        size: [width, height],
        rgb,
        camera: camera_name,
        color_calibrated: color.is_some() && valid_bayer(module.bayer_red),
        metadata: capture_metadata(capture, camera),
    })
}

fn preferred_color_calibration(capture: &Capture, camera: u64) -> Option<&ColorCalibration> {
    capture.colors.get(&camera).and_then(|profiles| {
        profiles
            .iter()
            .find(|profile| profile.illuminant == 2)
            .or_else(|| profiles.iter().find(|profile| profile.illuminant == 5))
            .or_else(|| profiles.first())
    })
}

fn parse_capture(data: &[u8]) -> Result<Capture, LriError> {
    parse_capture_impl(data, false)
}

fn parse_capture_prefix(data: &[u8]) -> Result<Capture, LriError> {
    parse_capture_impl(data, true)
}

fn parse_capture_impl(data: &[u8], allow_truncated_tail: bool) -> Result<Capture, LriError> {
    let mut capture = Capture::default();
    let mut block_offset = 0usize;
    let mut block_index = 0usize;

    while block_offset < data.len() {
        let Some(header) = data.get(block_offset..block_offset + LELR_HEADER_SIZE) else {
            if allow_truncated_tail {
                break;
            }
            return Err(format_error(format!(
                "truncated block header at 0x{block_offset:x}"
            )));
        };
        if &header[0..4] != b"LELR" {
            return Err(format_error(format!(
                "bad block magic at 0x{block_offset:x}"
            )));
        }
        let block_length = read_u64(&header[4..12])? as usize;
        let message_offset = read_u64(&header[12..20])? as usize;
        let message_length = read_u32(&header[20..24])? as usize;
        let message_type = header[24];
        if block_length < LELR_HEADER_SIZE {
            return Err(format_error(format!(
                "block {block_index} has invalid length {block_length}"
            )));
        }
        let Some(block_end) = block_offset.checked_add(block_length) else {
            return Err(format_error(format!(
                "block {block_index} length overflows the file"
            )));
        };
        if block_end > data.len() {
            if allow_truncated_tail {
                break;
            }
            return Err(format_error(format!(
                "block {block_index} extends past end of file"
            )));
        }
        let message_start = block_offset
            .checked_add(message_offset)
            .filter(|start| *start <= block_end)
            .ok_or_else(|| {
                format_error(format!("invalid message offset in block {block_index}"))
            })?;
        let message_end = message_start
            .checked_add(message_length)
            .filter(|end| *end <= block_end)
            .ok_or_else(|| format_error(format!("message escapes block {block_index}")))?;

        match message_type {
            0 => parse_light_header(
                &data[message_start..message_end],
                block_offset,
                block_end,
                &mut capture,
            )?,
            1 => parse_view_preferences(&data[message_start..message_end], &mut capture)?,
            _ => {}
        }

        block_offset = block_end;
        block_index += 1;
    }

    Ok(capture)
}

fn parse_light_header(
    message: &[u8],
    block_offset: usize,
    block_end: usize,
    capture: &mut Capture,
) -> Result<(), LriError> {
    let header = decode_proto::<LightHeader>(message, "LightHeader")?;
    if let Some(timestamp) = header.image_time_stamp.as_ref() {
        capture.captured_at = parse_timestamp(timestamp);
    }
    if let Some(focal_length) = header.image_focal_length {
        capture.focal_length_mm = Some(focal_length);
    }
    if let Some(camera) = header.image_reference_camera {
        capture.reference_camera = Some(enum_number(camera.value()));
    }
    for module in &header.modules {
        if let Some(module) = parse_camera_module(module, block_offset, block_end)? {
            capture.modules.push(module);
        }
    }
    for calibration in &header.module_calibration {
        parse_module_calibration(calibration, capture);
    }
    for sensor in &header.sensor_data {
        parse_sensor_data(sensor, capture);
    }
    if let Some(hardware) = header.hw_info.as_ref() {
        parse_hardware_info(hardware, capture);
    }
    if let Some(preferences) = header.view_preferences.as_ref() {
        apply_view_preferences(preferences, capture);
    }
    Ok(())
}

fn parse_view_preferences(message: &[u8], capture: &mut Capture) -> Result<(), LriError> {
    let preferences = decode_proto::<ViewPreferences>(message, "ViewPreferences")?;
    apply_view_preferences(&preferences, capture);
    Ok(())
}

fn apply_view_preferences(preferences: &ViewPreferences, capture: &mut Capture) {
    if let Some(scene_mode) = preferences.scene_mode {
        capture.scene_mode = Some(enum_number(scene_mode.value()));
    }
    if let Some(orientation) = preferences.orientation {
        capture.orientation = enum_number(orientation.value());
    }
    if let Some(gain) = preferences.image_gain {
        capture.image_gain = Some(gain);
    }
    if let Some(integration) = preferences.image_integration_time_ns {
        capture.image_integration_time_ns = Some(integration);
    }
    if let Some(gains) = preferences.awb_gains.as_ref() {
        capture.awb_gains = parse_channel_gains(gains);
    }
    if let Some(tripod) = preferences.is_on_tripod {
        capture.tripod = Some(tripod);
    }
}

fn parse_camera_module(
    message: &CameraModule,
    block_offset: usize,
    block_end: usize,
) -> Result<Option<CapturedModule>, LriError> {
    let Some(camera) = message.id.map(|id| enum_number(id.value())) else {
        return Ok(None);
    };
    let mut module = CapturedModule {
        camera,
        enabled: message.is_enabled.unwrap_or(true),
        analog_gain: message.sensor_analog_gain.unwrap_or(1.0),
        digital_gain: message.sensor_digital_gain.unwrap_or(1.0),
        sensor_exposure_ns: message.sensor_exposure.unwrap_or(0),
        frame_index: message.frame_index.map_or(0, u64::from),
        ..Default::default()
    };
    let Some(surface) = message.sensor_data_surface.as_ref() else {
        return Ok(None);
    };
    let data_offset = parse_surface(surface, &mut module)?;
    if let Some(point) = message.sensor_bayer_red_override.as_ref() {
        module.bayer_red = Some(parse_point(point));
    }

    if module.width == 0
        || module.height == 0
        || (module.format == RAW_PACKED_10BPP && module.row_stride == 0)
    {
        return Ok(None);
    }
    let Some(relative) = data_offset else {
        return Ok(None);
    };
    module.payload_offset = block_offset
        .checked_add(relative)
        .ok_or_else(|| format_error("camera payload offset overflow"))?;
    module.payload_limit = block_end;
    if module.format == RAW_PACKED_10BPP {
        module
            .payload_offset
            .checked_add(
                module
                    .row_stride
                    .checked_mul(module.height)
                    .ok_or_else(|| format_error("camera payload length overflow"))?,
            )
            .filter(|end| *end <= block_end)
            .ok_or_else(|| format_error("camera payload escapes its LELR block"))?;
    } else if module.format == RAW_BAYER_JPEG {
        let Some(header_end) = module.payload_offset.checked_add(BAYER_JPEG_HEADER_SIZE) else {
            return Ok(None);
        };
        if header_end > block_end {
            return Ok(None);
        }
    }
    Ok(Some(module))
}

fn parse_timestamp(timestamp: &TimeStamp) -> Option<CaptureDateTime> {
    match (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    ) {
        (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) => {
            Some(CaptureDateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
                timezone_offset_minutes: timestamp.tz_offset,
            })
        }
        _ => None,
    }
}

fn parse_module_calibration(message: &FactoryModuleCalibration, capture: &mut Capture) {
    if let Some(camera) = message.camera_id.map(|id| enum_number(id.value())) {
        let profiles = capture.colors.entry(camera).or_default();
        for color in message.color.iter().filter_map(parse_color_calibration) {
            if !profiles.iter().any(|existing| {
                existing.illuminant == color.illuminant
                    && existing.forward_matrix == color.forward_matrix
            }) {
                profiles.push(color);
            }
        }
    }
}

fn parse_color_calibration(message: &ProtoColorCalibration) -> Option<ColorCalibration> {
    match (
        message.type_.map(|value| enum_number(value.value())),
        message.forward_matrix.as_ref().and_then(parse_matrix3),
        message.rg_ratio,
        message.bg_ratio,
    ) {
        (Some(illuminant), Some(forward_matrix), Some(rg_ratio), Some(bg_ratio)) => {
            Some(ColorCalibration {
                illuminant,
                forward_matrix,
                rg_ratio,
                bg_ratio,
            })
        }
        _ => None,
    }
}

fn parse_matrix3(message: &Matrix3x3F) -> Option<[f32; 9]> {
    match (
        message.x00,
        message.x01,
        message.x02,
        message.x10,
        message.x11,
        message.x12,
        message.x20,
        message.x21,
        message.x22,
    ) {
        (
            Some(x00),
            Some(x01),
            Some(x02),
            Some(x10),
            Some(x11),
            Some(x12),
            Some(x20),
            Some(x21),
            Some(x22),
        ) => Some([x00, x01, x02, x10, x11, x12, x20, x21, x22]),
        _ => None,
    }
}

fn parse_hardware_info(message: &HwInfo, capture: &mut Capture) {
    for camera in &message.camera {
        if let (Some(id), Some(sensor)) = (camera.id, camera.sensor) {
            capture
                .sensor_types
                .insert(enum_number(id.value()), enum_number(sensor.value()));
        }
    }
}

fn parse_sensor_data(message: &SensorData, capture: &mut Capture) {
    if let (Some(sensor), Some(levels)) = (
        message.type_.map(|value| enum_number(value.value())),
        message
            .data
            .as_ref()
            .and_then(parse_sensor_characterization),
    ) {
        capture.sensor_levels.insert(sensor, levels);
    }
}

fn parse_sensor_characterization(message: &SensorCharacterization) -> Option<(f32, f32)> {
    message
        .black_level
        .zip(message.white_level)
        .filter(|(black, white)| white > black)
}

fn parse_channel_gains(message: &ChannelGain) -> Option<[f32; 3]> {
    match [message.r, message.g_r, message.g_b, message.b] {
        [Some(red), Some(green_red), Some(green_blue), Some(blue)] => {
            Some([red, (green_red + green_blue) * 0.5, blue])
        }
        _ => None,
    }
}

fn parse_surface(
    message: &Surface,
    module: &mut CapturedModule,
) -> Result<Option<usize>, LriError> {
    if let Some(size) = message.size.as_ref() {
        let (width, height) = parse_point(size);
        module.width = usize::try_from(width)
            .map_err(|_| format_error(format!("invalid RAW width {width}")))?;
        module.height = usize::try_from(height)
            .map_err(|_| format_error(format!("invalid RAW height {height}")))?;
    }
    module.format = message
        .format
        .map_or(RAW_BAYER_JPEG, |format| enum_number(format.value()));
    module.row_stride = message.row_stride.map_or(0, |stride| stride as usize);
    message
        .data_offset
        .map(|offset| {
            usize::try_from(offset).map_err(|_| format_error("RAW data offset is too large"))
        })
        .transpose()
}

fn parse_point(message: &Point2I) -> (i32, i32) {
    (message.x.unwrap_or(0), message.y.unwrap_or(0))
}

fn decode_proto<M: Message>(bytes: &[u8], name: &str) -> Result<M, LriError> {
    let mut message = M::new();
    message
        .merge_from_bytes(bytes)
        .map_err(|error| format_error(format!("invalid {name} protobuf: {error}")))?;
    Ok(message)
}

fn enum_number(value: i32) -> u64 {
    value as u32 as u64
}

fn render_preview(
    data: &[u8],
    module: &CapturedModule,
    orientation: u64,
    max_edge: usize,
    sensor_levels: (f32, f32),
    color: Option<&ColorCalibration>,
    awb_gains: Option<[f32; 3]>,
) -> Result<(usize, usize, Vec<u8>), LriError> {
    if max_edge == 0 {
        return Err(format_error("preview edge must be greater than zero"));
    }
    match module.format {
        RAW_PACKED_10BPP => {
            let pixel_count = module
                .width
                .checked_mul(module.height)
                .ok_or_else(|| format_error("RAW dimensions overflow"))?;
            if pixel_count % 4 != 0 {
                return Err(format_error(
                    "packed 10-bit surface has a partial sample group",
                ));
            }
            let packed_len = pixel_count
                .checked_mul(10)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| format_error("RAW dimensions overflow"))?;
            let payload_end = module
                .payload_offset
                .checked_add(packed_len)
                .ok_or_else(|| format_error("packed RAW payload offset overflow"))?;
            let payload = data
                .get(module.payload_offset..payload_end)
                .ok_or_else(|| format_error("truncated packed RAW payload"))?;
            let sampler = Packed10Sampler {
                payload,
                width: module.width,
                height: module.height,
            };
            render_sampled(
                &sampler,
                module,
                orientation,
                max_edge,
                sensor_levels,
                color,
                awb_gains,
            )
        }
        RAW_BAYER_JPEG => {
            let bayer = decode_bayer_jpeg(data, module)?;
            let sampler = BayerJpegSampler {
                bayer,
                width: module.width,
                height: module.height,
            };
            render_sampled(
                &sampler,
                module,
                orientation,
                max_edge,
                sensor_levels,
                color,
                awb_gains,
            )
        }
        format => Err(LriError::UnsupportedRaw {
            camera: camera_name(module.camera),
            format,
        }),
    }
}

fn render_sampled(
    sampler: &impl RawSampler,
    module: &CapturedModule,
    orientation: u64,
    max_edge: usize,
    sensor_levels: (f32, f32),
    color: Option<&ColorCalibration>,
    awb_gains: Option<[f32; 3]>,
) -> Result<(usize, usize, Vec<u8>), LriError> {
    let rotated = matches!(orientation, 1..=4);
    let (display_width, display_height) = if rotated {
        (module.height, module.width)
    } else {
        (module.width, module.height)
    };
    let scale = (max_edge as f64 / display_width.max(display_height) as f64).min(1.0);
    let out_width = (display_width as f64 * scale).round().max(1.0) as usize;
    let out_height = (display_height as f64 * scale).round().max(1.0) as usize;
    let mut linear = Vec::with_capacity(out_width * out_height);

    for y in 0..out_height {
        let dy = ((y as f64 + 0.5) * display_height as f64 / out_height as f64)
            .floor()
            .min((display_height - 1) as f64) as usize;
        for x in 0..out_width {
            let dx = ((x as f64 + 0.5) * display_width as f64 / out_width as f64)
                .floor()
                .min((display_width - 1) as f64) as usize;
            let (sx, sy) = oriented_source(dx, dy, module.width, module.height, orientation);
            linear.push(sample_rgb(sampler, sx, sy, module.bayer_red));
        }
    }

    Ok((
        out_width,
        out_height,
        tone_map(
            &mut linear,
            valid_bayer(module.bayer_red),
            sensor_levels,
            color,
            awb_gains,
        ),
    ))
}

fn decode_bayer_jpeg(data: &[u8], module: &CapturedModule) -> Result<Vec<u8>, LriError> {
    let payload = data
        .get(module.payload_offset..module.payload_limit)
        .ok_or_else(|| format_error("invalid Bayer-JPEG payload range"))?;
    let header = payload
        .get(..BAYER_JPEG_HEADER_SIZE)
        .ok_or_else(|| format_error("truncated Bayer-JPEG header"))?;
    if &header[..4] != b"BJPG" {
        return Err(format_error("invalid Bayer-JPEG magic"));
    }
    let layout = read_u32(&header[4..8])?;
    let lengths = [
        read_u32(&header[8..12])? as usize,
        read_u32(&header[12..16])? as usize,
        read_u32(&header[16..20])? as usize,
        read_u32(&header[20..24])? as usize,
    ];
    let mut cursor = BAYER_JPEG_HEADER_SIZE;
    let mut jpeg = Vec::with_capacity(4);
    for length in lengths {
        let end = cursor
            .checked_add(length)
            .filter(|end| *end <= payload.len())
            .ok_or_else(|| format_error("Bayer-JPEG channel escapes its LELR block"))?;
        jpeg.push(&payload[cursor..end]);
        cursor = end;
    }

    let pixel_count = module
        .width
        .checked_mul(module.height)
        .ok_or_else(|| format_error("Bayer-JPEG dimensions overflow"))?;
    match layout {
        0 => {
            if !module.width.is_multiple_of(2) || !module.height.is_multiple_of(2) {
                return Err(format_error("four-plane Bayer-JPEG has odd dimensions"));
            }
            let plane_size = pixel_count / 4;
            let mut bayer = vec![0; pixel_count];
            for (offset, bytes) in jpeg.into_iter().enumerate() {
                let plane = decode_gray_jpeg(bytes)?;
                if plane.len() != plane_size {
                    return Err(format_error(format!(
                        "Bayer-JPEG plane has {} pixels; expected {plane_size}",
                        plane.len()
                    )));
                }
                for (index, value) in plane.into_iter().enumerate() {
                    let x = index % (module.width / 2);
                    let y = index / (module.width / 2);
                    let bayer_x = x * 2 + offset % 2;
                    let bayer_y = y * 2 + offset / 2;
                    bayer[bayer_y * module.width + bayer_x] = value;
                }
            }
            Ok(bayer)
        }
        1 => {
            let bayer = decode_gray_jpeg(jpeg[0])?;
            if bayer.len() != pixel_count {
                return Err(format_error(format!(
                    "Bayer-JPEG image has {} pixels; expected {pixel_count}",
                    bayer.len()
                )));
            }
            Ok(bayer)
        }
        other => Err(format_error(format!(
            "unsupported Bayer-JPEG layout {other}"
        ))),
    }
}

fn decode_gray_jpeg(bytes: &[u8]) -> Result<Vec<u8>, LriError> {
    zune_jpeg::JpegDecoder::new(bytes)
        .decode()
        .map_err(|error| format_error(format!("JPEG decode failed: {error}")))
}

fn oriented_source(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    orientation: u64,
) -> (usize, usize) {
    match orientation {
        1 => (y, height - 1 - x),
        2 => (width - 1 - y, x),
        3 => (width - 1 - y, height - 1 - x),
        4 => (y, x),
        5 => (x, height - 1 - y),
        6 => (width - 1 - x, y),
        7 => (width - 1 - x, height - 1 - y),
        _ => (x, y),
    }
}

struct Packed10Sampler<'a> {
    payload: &'a [u8],
    width: usize,
    height: usize,
}

trait RawSampler {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn calibrated(&self, x: usize, y: usize) -> u16;
}

impl Packed10Sampler<'_> {
    fn stream(&self, index: usize) -> u16 {
        let group = index / 4;
        let within = index % 4;
        let reversed_start = group * 5;
        let byte = |offset: usize| self.payload[self.payload.len() - 1 - reversed_start - offset];
        let b0 = byte(0) as u16;
        let b1 = byte(1) as u16;
        let b2 = byte(2) as u16;
        let b3 = byte(3) as u16;
        let b4 = byte(4) as u16;
        match within {
            0 => (b0 << 2) | (b1 >> 6),
            1 => ((b1 & 0x3f) << 4) | (b2 >> 4),
            2 => ((b2 & 0x0f) << 6) | (b3 >> 2),
            _ => ((b3 & 0x03) << 8) | b4,
        }
    }
}

impl RawSampler for Packed10Sampler<'_> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn calibrated(&self, x: usize, y: usize) -> u16 {
        // Captured stream raster is rotated 180 degrees relative to the pixel
        // coordinates used by L16 calibration and CFA metadata.
        let stream_x = self.width - 1 - x.min(self.width - 1);
        let stream_y = self.height - 1 - y.min(self.height - 1);
        self.stream(stream_y * self.width + stream_x)
    }
}

struct BayerJpegSampler {
    bayer: Vec<u8>,
    width: usize,
    height: usize,
}

impl RawSampler for BayerJpegSampler {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn calibrated(&self, x: usize, y: usize) -> u16 {
        let stream_x = self.width - 1 - x.min(self.width - 1);
        let stream_y = self.height - 1 - y.min(self.height - 1);
        u16::from(self.bayer[stream_y * self.width + stream_x])
    }
}

fn sample_rgb(
    sampler: &impl RawSampler,
    x: usize,
    y: usize,
    bayer_red: Option<(i32, i32)>,
) -> [f32; 3] {
    let Some(red) = bayer_red.filter(|&(x, y)| (0..=1).contains(&x) && (0..=1).contains(&y)) else {
        let value = sampler.calibrated(x, y) as f32;
        return [value; 3];
    };

    let mut sum = [0.0f32; 3];
    let mut count = [0u8; 3];
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    let x1 = (x + 1).min(sampler.width() - 1);
    let y1 = (y + 1).min(sampler.height() - 1);
    for sy in y0..=y1 {
        for sx in x0..=x1 {
            let channel = cfa_channel(sx, sy, red);
            sum[channel] += sampler.calibrated(sx, sy) as f32;
            count[channel] += 1;
        }
    }
    for channel in 0..3 {
        if count[channel] == 0 {
            sum[channel] = sampler.calibrated(x, y) as f32;
        } else {
            sum[channel] /= count[channel] as f32;
        }
    }
    sum
}

fn cfa_channel(x: usize, y: usize, red: (i32, i32)) -> usize {
    let parity = ((x & 1) as i32, (y & 1) as i32);
    if parity == red {
        0
    } else if parity == (1 - red.0, 1 - red.1) {
        2
    } else {
        1
    }
}

fn tone_map(
    raw: &mut [[f32; 3]],
    is_color: bool,
    (black, white): (f32, f32),
    calibration: Option<&ColorCalibration>,
    capture_gains: Option<[f32; 3]>,
) -> Vec<u8> {
    let range = (white - black).max(1.0);
    let fallback_gains = calibration.map(|profile| {
        [
            profile.rg_ratio.max(0.01).recip(),
            1.0,
            profile.bg_ratio.max(0.01).recip(),
        ]
    });
    let gains = capture_gains.or(fallback_gains).unwrap_or([1.0; 3]);
    let mut luminance = Vec::with_capacity(raw.len());

    for pixel in raw.iter_mut() {
        let mut corrected = pixel.map(|value| ((value - black) / range).max(0.0));
        if is_color {
            for channel in 0..3 {
                corrected[channel] *= gains[channel];
            }
            if let Some(profile) = calibration {
                corrected = camera_to_srgb(corrected, &profile.forward_matrix);
            }
        }
        corrected = corrected.map(|value| value.max(0.0));
        luminance.push(0.2126 * corrected[0] + 0.7152 * corrected[1] + 0.0722 * corrected[2]);
        *pixel = corrected;
    }

    luminance.sort_unstable_by(f32::total_cmp);
    let percentile = luminance
        .get(((luminance.len().saturating_sub(1)) as f32 * 0.995) as usize)
        .copied()
        .unwrap_or(1.0)
        .max(0.000_1);
    let exposure = 0.92 / percentile;

    let mut rgb = Vec::with_capacity(raw.len() * 3);
    for pixel in raw {
        for channel in pixel.iter() {
            let value = (channel * exposure).clamp(0.0, 1.0).powf(1.0 / 2.2);
            rgb.push((value * 255.0).round() as u8);
        }
    }
    rgb
}

fn camera_to_srgb(camera: [f32; 3], forward: &[f32; 9]) -> [f32; 3] {
    const XYZ_TO_SRGB: [f32; 9] = [
        3.240_454_2,
        -1.537_138_5,
        -0.498_531_4,
        -0.969_266,
        1.876_010_8,
        0.041_556,
        0.055_643_4,
        -0.204_025_9,
        1.057_225_2,
    ];
    let xyz = multiply_matrix_vector(forward, camera);
    multiply_matrix_vector(&XYZ_TO_SRGB, xyz)
}

fn multiply_matrix_vector(matrix: &[f32; 9], value: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
        matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
        matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2],
    ]
}

fn valid_bayer(value: Option<(i32, i32)>) -> bool {
    value.is_some_and(|(x, y)| (0..=1).contains(&x) && (0..=1).contains(&y))
}

fn camera_name(camera: u64) -> String {
    CAMERA_NAMES
        .get(camera as usize)
        .map(|name| (*name).to_owned())
        .unwrap_or_else(|| format!("camera {camera}"))
}

fn camera_index(camera: &str) -> Option<u64> {
    CAMERA_NAMES
        .iter()
        .position(|name| name.eq_ignore_ascii_case(camera))
        .map(|index| index as u64)
}

fn read_u64(bytes: &[u8]) -> Result<u64, LriError> {
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| format_error("truncated 64-bit LELR field"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(bytes: &[u8]) -> Result<u32, LriError> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| format_error("truncated 32-bit LELR field"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn format_error(message: impl Into<String>) -> LriError {
    LriError::Format(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_access_unpack_matches_known_samples() {
        let samples = [0, 1, 1022, 1023, 42, 512, 17, 900];
        let packed = pack_10bit(&samples);
        let sampler = Packed10Sampler {
            payload: &packed,
            width: 4,
            height: 2,
        };
        let unpacked = (0..samples.len())
            .map(|index| sampler.stream(index))
            .collect::<Vec<_>>();
        assert_eq!(unpacked, samples);
    }

    #[test]
    fn reference_camera_and_module_are_read_from_light_header() {
        let size = message(&[(1, varint_value(4)), (2, varint_value(2))]);
        let surface = message(&[
            (2, bytes_value(&size)),
            (3, varint_value(7)),
            (4, varint_value(5)),
            (5, varint_value(32)),
        ]);
        let bayer = message(&[(1, varint_value(1)), (2, varint_value(0))]);
        let module = message(&[
            (2, varint_value(1)),
            (9, bytes_value(&surface)),
            (13, bytes_value(&bayer)),
        ]);
        let light_header = message(&[(5, varint_value(1)), (12, bytes_value(&module))]);
        let raw = pack_10bit(&[100, 200, 300, 400, 500, 600, 700, 800]);
        let message_offset = LELR_HEADER_SIZE + raw.len();
        let block_length = message_offset + light_header.len();
        let mut lri = Vec::with_capacity(block_length);
        lri.extend_from_slice(b"LELR");
        lri.extend_from_slice(&(block_length as u64).to_le_bytes());
        lri.extend_from_slice(&(message_offset as u64).to_le_bytes());
        lri.extend_from_slice(&(light_header.len() as u32).to_le_bytes());
        lri.push(0);
        lri.extend_from_slice(&[0; 7]);
        lri.extend_from_slice(&raw);
        lri.extend_from_slice(&light_header);

        assert_eq!(complete_lelr_prefix_len(&lri[..lri.len() - 1]).unwrap(), 0);
        assert_eq!(complete_lelr_prefix_len(&lri).unwrap(), lri.len());
        let descriptor = inspect_lelr_block_header(&lri[..LELR_HEADER_SIZE], 0).unwrap();
        assert_eq!(descriptor.block_range(), 0..lri.len());
        assert_eq!(
            descriptor.message_range(),
            message_offset..message_offset + light_header.len()
        );
        assert_eq!(
            reference_camera_block_range(&lri).unwrap(),
            Some(0..lri.len())
        );
        assert_eq!(
            reference_camera_payload_read(&lri).unwrap(),
            Some(ReferencePayloadRead::Payload(
                LELR_HEADER_SIZE..LELR_HEADER_SIZE + raw.len()
            ))
        );
        assert_eq!(reference_preview_color_ready(&lri).unwrap(), Some(false));
        let capture = parse_capture(&lri).unwrap();
        assert_eq!(capture.reference_camera, Some(1));
        assert_eq!(capture.modules.len(), 1);
        assert_eq!(capture.modules[0].camera, 1);
        assert_eq!(capture.modules[0].width, 4);
        assert_eq!(capture.modules[0].height, 2);
        assert_eq!(capture.modules[0].bayer_red, Some((1, 0)));
        let layout = parse_raw_layout(&lri, &HashMap::new()).unwrap();
        assert_eq!(layout.cameras.len(), 1);
        assert_eq!(layout.cameras[0].name, "A2");
        assert_eq!(layout.cameras[0].absolute_offset, LELR_HEADER_SIZE);
        assert_eq!(layout.cameras[0].byte_len, raw.len());
        assert_eq!(layout.cameras[0].pattern, SensorPattern::Grbg);

        assert!(
            try_decode_reference_preview_prefix(&lri[..lri.len() - 1], 320)
                .unwrap()
                .is_none()
        );
        assert!(
            inspect_capture_prefix(&lri[..lri.len() - 1])
                .unwrap()
                .is_none()
        );
        assert_eq!(
            inspect_capture_prefix(&lri)
                .unwrap()
                .expect("complete block should be inspectable")
                .cameras
                .len(),
            1
        );
        let preview = try_decode_reference_preview_prefix(&lri, 320)
            .unwrap()
            .expect("complete first block should decode");
        assert_eq!(preview.camera, "A2");
        assert_eq!(preview.size, [4, 2]);
        assert!(
            try_decode_camera_preview_prefix(&lri, "A2", 320)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn incomplete_camera_record_is_ignored_without_rejecting_capture() {
        let module = message(&[(2, varint_value(4))]);
        let light_header = message(&[(5, varint_value(1)), (12, bytes_value(&module))]);
        let block_length = LELR_HEADER_SIZE + light_header.len();
        let mut lri = Vec::with_capacity(block_length);
        lri.extend_from_slice(b"LELR");
        lri.extend_from_slice(&(block_length as u64).to_le_bytes());
        lri.extend_from_slice(&(LELR_HEADER_SIZE as u64).to_le_bytes());
        lri.extend_from_slice(&(light_header.len() as u32).to_le_bytes());
        lri.push(0);
        lri.extend_from_slice(&[0; 7]);
        lri.extend_from_slice(&light_header);

        let capture = parse_capture(&lri).unwrap();
        assert_eq!(capture.reference_camera, Some(1));
        assert!(capture.modules.is_empty());
    }

    #[test]
    fn bayer_jpeg_surface_does_not_require_a_raw_row_stride() {
        let size = message(&[(1, varint_value(4)), (2, varint_value(2))]);
        let surface = message(&[
            (2, bytes_value(&size)),
            (3, varint_value(RAW_BAYER_JPEG)),
            (4, varint_value(0)),
            (5, varint_value(LELR_HEADER_SIZE as u64)),
        ]);
        let module = message(&[
            (2, varint_value(1)),
            (9, bytes_value(&surface)),
            (15, varint_value(0)),
        ]);
        let light_header = message(&[(5, varint_value(1)), (12, bytes_value(&module))]);
        let message_offset = LELR_HEADER_SIZE + BAYER_JPEG_HEADER_SIZE;
        let block_length = message_offset + light_header.len();
        let mut lri = Vec::with_capacity(block_length);
        lri.extend_from_slice(b"LELR");
        lri.extend_from_slice(&(block_length as u64).to_le_bytes());
        lri.extend_from_slice(&(message_offset as u64).to_le_bytes());
        lri.extend_from_slice(&(light_header.len() as u32).to_le_bytes());
        lri.push(0);
        lri.extend_from_slice(&[0; 7]);
        lri.resize(message_offset, 0);
        lri.extend_from_slice(&light_header);

        assert_eq!(
            reference_camera_payload_read(&lri).unwrap(),
            Some(ReferencePayloadRead::BayerJpegHeader(
                LELR_HEADER_SIZE..LELR_HEADER_SIZE + BAYER_JPEG_HEADER_SIZE
            ))
        );
        lri[LELR_HEADER_SIZE..LELR_HEADER_SIZE + 4].copy_from_slice(b"BJPG");
        assert_eq!(
            reference_camera_payload_read(&lri).unwrap(),
            Some(ReferencePayloadRead::Payload(
                LELR_HEADER_SIZE..LELR_HEADER_SIZE + BAYER_JPEG_HEADER_SIZE
            ))
        );
        let capture = parse_capture(&lri).unwrap();
        assert_eq!(capture.modules.len(), 1);
        assert_eq!(capture.modules[0].format, RAW_BAYER_JPEG);
        assert_eq!(capture.modules[0].row_stride, 0);
    }

    #[test]
    fn parses_d65_forward_matrix_and_white_balance_gains() {
        let mut matrix = Vec::new();
        for field in 1..=9 {
            matrix.extend(encode_varint((field << 3) | 5));
            matrix.extend_from_slice(&(field as f32).to_le_bytes());
        }
        let mut color = Vec::new();
        color.extend(encode_varint(1 << 3));
        color.extend(encode_varint(2));
        color.extend(encode_varint((2 << 3) | 2));
        color.extend(encode_varint(matrix.len() as u64));
        color.extend_from_slice(&matrix);
        color.extend(encode_varint((4 << 3) | 5));
        color.extend_from_slice(&0.5f32.to_le_bytes());
        color.extend(encode_varint((5 << 3) | 5));
        color.extend_from_slice(&0.25f32.to_le_bytes());

        let color = decode_proto::<ProtoColorCalibration>(&color, "ColorCalibration").unwrap();
        let parsed = parse_color_calibration(&color).unwrap();
        assert_eq!(parsed.illuminant, 2);
        assert_eq!(
            parsed.forward_matrix,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(parsed.rg_ratio, 0.5);
        assert_eq!(parsed.bg_ratio, 0.25);

        let mut gains = Vec::new();
        for (field, value) in [(1, 1.9f32), (2, 1.0), (3, 1.0), (4, 1.4)] {
            gains.extend(encode_varint((field << 3) | 5));
            gains.extend_from_slice(&value.to_le_bytes());
        }
        let gains = decode_proto::<ChannelGain>(&gains, "ChannelGain").unwrap();
        assert_eq!(parse_channel_gains(&gains), Some([1.9, 1.0, 1.4]));
    }

    fn pack_10bit(samples: &[u16]) -> Vec<u8> {
        assert_eq!(samples.len() % 4, 0);
        let mut packed = Vec::new();
        for values in samples.chunks_exact(4) {
            let word = (u64::from(values[0]) << 30)
                | (u64::from(values[1]) << 20)
                | (u64::from(values[2]) << 10)
                | u64::from(values[3]);
            packed.extend_from_slice(&[
                (word >> 32) as u8,
                (word >> 24) as u8,
                (word >> 16) as u8,
                (word >> 8) as u8,
                word as u8,
            ]);
        }
        packed.reverse();
        packed
    }

    fn message(fields: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (field, value) in fields {
            bytes.extend(encode_varint(
                (field << 3) | if value.first() == Some(&0xff) { 2 } else { 0 },
            ));
            if value.first() == Some(&0xff) {
                bytes.extend(encode_varint((value.len() - 1) as u64));
                bytes.extend_from_slice(&value[1..]);
            } else {
                bytes.extend_from_slice(value);
            }
        }
        bytes
    }

    fn varint_value(value: u64) -> Vec<u8> {
        encode_varint(value)
    }

    fn bytes_value(value: &[u8]) -> Vec<u8> {
        let mut tagged = vec![0xff];
        tagged.extend_from_slice(value);
        tagged
    }

    fn encode_varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }
}
