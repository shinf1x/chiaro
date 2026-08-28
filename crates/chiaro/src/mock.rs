//! Synthetic Light L16 captures for tests, examples, and UI development.
//!
//! Enabled with the `mock` cargo feature. The generated files use the real
//! LELR framing and recovered `LightHeader` / `ViewPreferences` schemas, so
//! every parser in this crate (and downstream processors such as
//! `chiaro-hotpixel-core`) treats them exactly like camera output:
//!
//! - one `LightHeader` block carrying every `RAW_PACKED_10BPP` camera payload
//!   in Light's reversed five-byte packing;
//! - per-sensor black/white levels and sensor types (colour or monochrome);
//! - an identity D65 colour calibration for each Bayer camera so previews
//!   become colour-ready;
//! - a trailing `ViewPreferences` block with orientation and scene mode.
//!
//! Pixel content is deliberately simple (a smooth gradient plus optional
//! defects) so files stay compressible and visually recognisable.

use std::collections::HashMap;

use chiaro_proto::{
    Enum, Message,
    camera_id::CameraID,
    camera_module::{CameraModule, camera_module::Surface, camera_module::surface::FormatType},
    color_calibration::{ColorCalibration, color_calibration::IlluminantType},
    hw_info::{CameraModuleHwInfo, HwInfo},
    lightheader::{FactoryModuleCalibration, LightHeader, SensorData},
    matrix3x3f::Matrix3x3F,
    point2i::Point2I,
    sensor_characterization::SensorCharacterization,
    sensor_type::SensorType,
    time_stamp::TimeStamp,
    view_preferences::{
        ViewPreferences, view_preferences::Orientation, view_preferences::SceneMode,
    },
};

use crate::lri::SensorPattern;

const LELR_HEADER_SIZE: usize = 32;
const CAMERA_NAMES: [&str; 16] = [
    "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "C6",
];

/// One camera module in a [`MockCapture`].
#[derive(Clone, Debug)]
pub struct MockCamera {
    /// Physical camera name, `A1`..`C6`.
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub pattern: SensorPattern,
    pub sensor_temperature_c: Option<i32>,
    pub exposure_ns: u64,
    pub analog_gain: f32,
    pub digital_gain: f32,
    /// `width * height` 10-bit samples in decoded RAW order.
    pub samples: Vec<u16>,
}

impl MockCamera {
    /// A camera filled with a smooth diagonal gradient between `low` and `high`.
    pub fn gradient(
        name: &str,
        width: usize,
        height: usize,
        pattern: SensorPattern,
        low: u16,
        high: u16,
    ) -> Self {
        let low = low.min(1023);
        let high = high.min(1023);
        let span = f64::from(high) - f64::from(low);
        let denominator = ((width + height).saturating_sub(2)).max(1) as f64;
        let mut samples = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let t = (x + y) as f64 / denominator;
                samples.push((f64::from(low) + span * t).round() as u16);
            }
        }
        Self {
            name: name.to_owned(),
            width,
            height,
            pattern,
            sensor_temperature_c: Some(40),
            exposure_ns: 14_999_805_952,
            analog_gain: 6.25,
            digital_gain: 1.015625,
            samples,
        }
    }

    /// Force specific RAW coordinates to a value, for example to plant hot
    /// pixels at factory-listed positions.
    pub fn with_defects(mut self, defects: &[(usize, usize, u16)]) -> Self {
        for &(x, y, value) in defects {
            if x < self.width && y < self.height {
                self.samples[y * self.width + x] = value.min(1023);
            }
        }
        self
    }

    fn id(&self) -> Option<usize> {
        CAMERA_NAMES
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(&self.name))
    }
}

/// A complete synthetic capture.
#[derive(Clone, Debug)]
pub struct MockCapture {
    pub cameras: Vec<MockCamera>,
    /// Name of the reference camera; defaults to the first camera.
    pub reference_camera: Option<String>,
    pub focal_length_mm: Option<i32>,
    pub captured_at: Option<(u32, u32, u32, u32, u32, u32)>,
    /// `ViewPreferences` orientation code (0 = normal).
    pub orientation: i32,
    pub night_mode: bool,
    pub tripod: Option<bool>,
}

impl Default for MockCapture {
    fn default() -> Self {
        Self {
            cameras: Vec::new(),
            reference_camera: None,
            focal_length_mm: Some(70),
            captured_at: Some((2024, 8, 23, 1, 30, 0)),
            orientation: 0,
            night_mode: true,
            tripod: Some(true),
        }
    }
}

impl MockCapture {
    /// A small, fast three-camera capture suitable for unit tests.
    pub fn small() -> Self {
        Self {
            cameras: vec![
                MockCamera::gradient("A1", 64, 48, SensorPattern::Bggr, 64, 700),
                MockCamera::gradient("A2", 64, 48, SensorPattern::Mono, 64, 700),
                MockCamera::gradient("B1", 64, 48, SensorPattern::Grbg, 64, 700),
            ],
            ..Self::default()
        }
    }

    /// Serialise to LRI bytes.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        if self.cameras.is_empty() {
            return Err("mock capture needs at least one camera".to_owned());
        }
        let mut header = LightHeader::new();
        header.image_focal_length = self.focal_length_mm;
        if let Some((year, month, day, hour, minute, second)) = self.captured_at {
            let mut stamp = TimeStamp::new();
            stamp.year = Some(year);
            stamp.month = Some(month);
            stamp.day = Some(day);
            stamp.hour = Some(hour);
            stamp.minute = Some(minute);
            stamp.second = Some(second);
            header.image_time_stamp = Some(stamp).into();
        }
        let reference_name = self
            .reference_camera
            .clone()
            .unwrap_or_else(|| self.cameras[0].name.clone());
        let reference = self
            .cameras
            .iter()
            .find(|camera| camera.name.eq_ignore_ascii_case(&reference_name))
            .and_then(MockCamera::id)
            .ok_or_else(|| format!("reference camera {reference_name} is not in the capture"))?;
        header.image_reference_camera = Some(camera_id(reference));
        header.device_model_name = Some("L16".to_owned());

        let mut hw_info = HwInfo::new();
        let mut sensor_types = HashMap::new();
        let mut payloads = Vec::new();
        let mut offset = LELR_HEADER_SIZE;
        for camera in &self.cameras {
            let id = camera
                .id()
                .ok_or_else(|| format!("unknown Light L16 camera name {}", camera.name))?;
            if camera.samples.len() != camera.width * camera.height {
                return Err(format!(
                    "{} has {} samples; expected {}x{}",
                    camera.name,
                    camera.samples.len(),
                    camera.width,
                    camera.height
                ));
            }
            let packed = pack_raw10(&camera.samples)?;
            let row_stride = camera.width * 5 / 4;
            let sensor = if camera.pattern == SensorPattern::Mono {
                SensorType::SENSOR_AR1335_MONO
            } else {
                SensorType::SENSOR_AR1335
            };
            sensor_types.insert(sensor, ());

            let mut size = Point2I::new();
            size.x = Some(camera.width as i32);
            size.y = Some(camera.height as i32);
            let mut surface = Surface::new();
            surface.size = Some(size).into();
            surface.format = Some(FormatType::RAW_PACKED_10BPP.into());
            surface.row_stride = Some(row_stride as u32);
            surface.data_offset = Some(offset as u64);

            let mut module = CameraModule::new();
            module.id = Some(camera_id(id));
            module.is_enabled = Some(true);
            module.sensor_analog_gain = Some(camera.analog_gain);
            module.sensor_digital_gain = Some(camera.digital_gain);
            module.sensor_exposure = Some(camera.exposure_ns);
            module.sensor_temparature = camera.sensor_temperature_c;
            module.frame_index = Some(0);
            module.sensor_data_surface = Some(surface).into();
            if camera.pattern != SensorPattern::Mono {
                let (x, y) = match camera.pattern {
                    SensorPattern::Rggb => (0, 0),
                    SensorPattern::Grbg => (1, 0),
                    SensorPattern::Gbrg => (0, 1),
                    SensorPattern::Bggr | SensorPattern::Mono => (1, 1),
                };
                let mut red = Point2I::new();
                red.x = Some(x);
                red.y = Some(y);
                module.sensor_bayer_red_override = Some(red).into();

                let mut calibration = FactoryModuleCalibration::new();
                calibration.camera_id = Some(camera_id(id));
                calibration.color.push(identity_color_calibration());
                header.module_calibration.push(calibration);
            }
            header.modules.push(module);

            let mut hardware = CameraModuleHwInfo::new();
            hardware.id = Some(camera_id(id));
            hardware.sensor = Some(sensor.into());
            hw_info.camera.push(hardware);

            offset += packed.len();
            payloads.push(packed);
        }
        header.hw_info = Some(hw_info).into();
        for sensor in sensor_types.into_keys() {
            let mut levels = SensorCharacterization::new();
            levels.black_level = Some(42.0);
            levels.white_level = Some(1023.0);
            let mut data = SensorData::new();
            data.type_ = Some(sensor.into());
            data.data = Some(levels).into();
            header.sensor_data.push(data);
        }

        let message = header
            .write_to_bytes()
            .map_err(|error| format!("encode LightHeader: {error}"))?;
        let message_offset = offset;
        let block_length = message_offset + message.len();
        let mut lri = Vec::with_capacity(block_length + 96);
        push_block_header(&mut lri, block_length, message_offset, message.len(), 0);
        for payload in &payloads {
            lri.extend_from_slice(payload);
        }
        lri.extend_from_slice(&message);

        let mut preferences = ViewPreferences::new();
        preferences.orientation = Some(
            Orientation::from_i32(self.orientation)
                .unwrap_or(Orientation::ORIENTATION_NORMAL)
                .into(),
        );
        preferences.scene_mode = Some(
            if self.night_mode {
                SceneMode::SCENE_MODE_NIGHT
            } else {
                SceneMode::SCENE_MODE_NONE
            }
            .into(),
        );
        preferences.is_on_tripod = self.tripod;
        let preferences = preferences
            .write_to_bytes()
            .map_err(|error| format!("encode ViewPreferences: {error}"))?;
        push_block_header(
            &mut lri,
            LELR_HEADER_SIZE + preferences.len(),
            LELR_HEADER_SIZE,
            preferences.len(),
            1,
        );
        lri.extend_from_slice(&preferences);
        Ok(lri)
    }
}

fn camera_id(index: usize) -> chiaro_proto::EnumOrUnknown<CameraID> {
    CameraID::from_i32(index as i32)
        .unwrap_or(CameraID::A1)
        .into()
}

fn identity_color_calibration() -> ColorCalibration {
    let mut matrix = Matrix3x3F::new();
    matrix.x00 = Some(1.0);
    matrix.x01 = Some(0.0);
    matrix.x02 = Some(0.0);
    matrix.x10 = Some(0.0);
    matrix.x11 = Some(1.0);
    matrix.x12 = Some(0.0);
    matrix.x20 = Some(0.0);
    matrix.x21 = Some(0.0);
    matrix.x22 = Some(1.0);
    let mut color = ColorCalibration::new();
    color.type_ = Some(IlluminantType::D65.into());
    color.forward_matrix = Some(matrix).into();
    color.rg_ratio = Some(1.0);
    color.bg_ratio = Some(1.0);
    color
}

fn push_block_header(
    lri: &mut Vec<u8>,
    block_length: usize,
    message_offset: usize,
    message_length: usize,
    message_type: u8,
) {
    lri.extend_from_slice(b"LELR");
    lri.extend_from_slice(&(block_length as u64).to_le_bytes());
    lri.extend_from_slice(&(message_offset as u64).to_le_bytes());
    lri.extend_from_slice(&(message_length as u32).to_le_bytes());
    lri.push(message_type);
    lri.extend_from_slice(&[0; 7]);
}

/// Pack 10-bit samples into Light's reversed five-byte-group stream.
pub fn pack_raw10(samples: &[u16]) -> Result<Vec<u8>, String> {
    if !samples.len().is_multiple_of(4) {
        return Err("RAW10 sample count must be divisible by four".to_owned());
    }
    if let Some(&value) = samples.iter().find(|&&value| value > 1023) {
        return Err(format!("RAW10 sample {value} exceeds 1023"));
    }
    let mut forward = Vec::with_capacity(samples.len() / 4 * 5);
    for group in samples.as_chunks::<4>().0 {
        let word = (u64::from(group[0]) << 30)
            | (u64::from(group[1]) << 20)
            | (u64::from(group[2]) << 10)
            | u64::from(group[3]);
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
    use crate::lri::{decode_reference_preview_bytes, inspect_capture_bytes, parse_raw_layout};

    #[test]
    fn mock_capture_round_trips_through_the_real_parsers() {
        let mock = MockCapture::small();
        let bytes = mock.encode().unwrap();

        let summary = inspect_capture_bytes(&bytes).unwrap();
        assert_eq!(summary.reference_camera, "A1");
        assert_eq!(summary.cameras.len(), 3);
        assert_eq!(summary.metadata.focal_length_mm, Some(70));
        assert!(summary.metadata.night_mode);
        assert_eq!(summary.metadata.shutter_ns, Some(14_999_805_952));

        let layout = parse_raw_layout(&bytes, &HashMap::new()).unwrap();
        assert_eq!(layout.cameras.len(), 3);
        let a1 = layout.cameras.iter().find(|c| c.name == "A1").unwrap();
        assert_eq!((a1.width, a1.height), (64, 48));
        assert_eq!(a1.pattern, SensorPattern::Bggr);
        assert_eq!(a1.sensor_temperature_c, Some(40));
        assert_eq!(a1.byte_len, 64 * 48 * 5 / 4);
        let a2 = layout.cameras.iter().find(|c| c.name == "A2").unwrap();
        assert_eq!(a2.pattern, SensorPattern::Mono);
        assert_eq!(a2.black_level, 42.0);

        let preview = decode_reference_preview_bytes(&bytes, 32).unwrap();
        assert_eq!(preview.camera, "A1");
        assert!(preview.color_calibrated);
        assert_eq!(preview.size[0].max(preview.size[1]), 32);
    }

    #[test]
    fn raw_payload_bytes_decode_to_the_planted_samples() {
        let camera = MockCamera::gradient("C6", 8, 4, SensorPattern::Mono, 10, 20)
            .with_defects(&[(3, 1, 1000)]);
        let bytes = MockCapture {
            cameras: vec![camera.clone()],
            ..MockCapture::default()
        }
        .encode()
        .unwrap();
        let layout = parse_raw_layout(&bytes, &HashMap::new()).unwrap();
        let raw = &layout.cameras[0];
        let packed = &bytes[raw.absolute_offset..raw.absolute_offset + raw.byte_len];
        assert_eq!(packed, pack_raw10(&camera.samples).unwrap().as_slice());
        assert_eq!(camera.samples[8 + 3], 1000);
    }
}
