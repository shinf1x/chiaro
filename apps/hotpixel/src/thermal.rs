use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use chiaro::lri::{RawCamera, SensorPattern};

const PROFILE_VERSION: u32 = 2;
const MANIFEST_NAME: &str = "manifest.json";
const REFERENCE_SCALE: f32 = 256.0;
const SLOPE_SCALE: f32 = 4096.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalProfileManifest {
    pub format: String,
    pub version: u32,
    pub source: String,
    pub sensor_family: String,
    pub width: usize,
    pub height: usize,
    pub grid_width: usize,
    pub grid_height: usize,
    pub reference_temperature_c: f32,
    pub temperature_min_c: i32,
    pub temperature_max_c: i32,
    pub exposure_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_min_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_max_ns: Option<u64>,
    pub reference_analog_gain: f32,
    pub reference_digital_gain: f32,
    pub contributing_cameras: Vec<ThermalSourceCamera>,
    pub coefficients: String,
    pub coefficient_layout: String,
    pub orientation_rule: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalSourceCamera {
    pub camera: String,
    pub pattern: String,
    pub frame_count: usize,
    pub temperature_min_c: i32,
    pub temperature_max_c: i32,
    pub analog_gain: f32,
    pub digital_gain: f32,
}

pub struct ThermalProfile {
    pub root: PathBuf,
    pub manifest: ThermalProfileManifest,
    coefficients: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ThermalCorrectionStats {
    pub applied: bool,
    pub reason: Option<String>,
    pub requested_temperature_c: Option<i32>,
    pub applied_temperature_c: Option<f32>,
    pub temperature_clamped: bool,
    pub exposure_scale: Option<f32>,
    pub mean_absolute_dark_change: f64,
    pub maximum_absolute_dark_change: u16,
}

impl ThermalProfile {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join(MANIFEST_NAME);
        let manifest: ThermalProfileManifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?,
        )
        .with_context(|| format!("parse {}", manifest_path.display()))?;
        let coefficient_path = root.join(&manifest.coefficients);
        let coefficients = fs::read(&coefficient_path)
            .with_context(|| format!("read {}", coefficient_path.display()))?;
        Self::from_parts(root, manifest, coefficients)
    }

    pub fn bundled() -> Result<Self> {
        let manifest: ThermalProfileManifest =
            serde_json::from_str(include_str!("../assets/l16-glow-v2-manifest.json"))
                .context("parse bundled glow profile manifest")?;
        let coefficients = decode_base64(include_str!("../assets/l16-glow-v2.coefficients.b64"))?;
        Self::from_parts(PathBuf::from("<bundled>"), manifest, coefficients)
    }

    fn from_parts(
        root: PathBuf,
        manifest: ThermalProfileManifest,
        coefficients: Vec<u8>,
    ) -> Result<Self> {
        if manifest.format != "chiaro-sensor-glow-profile" || manifest.version != PROFILE_VERSION {
            bail!(
                "unsupported glow profile format/version: {}/{}",
                manifest.format,
                manifest.version
            );
        }
        if manifest.grid_width < 2 || manifest.grid_height < 2 {
            bail!("glow profile grid must be at least 2x2");
        }
        let expected = manifest
            .grid_width
            .checked_mul(manifest.grid_height)
            .and_then(|count| count.checked_mul(4))
            .context("glow profile dimensions overflow")?;
        if coefficients.len() != expected {
            bail!(
                "{} has {} coefficient bytes; expected {}",
                manifest.coefficients,
                coefficients.len(),
                expected
            );
        }
        Ok(Self {
            root,
            manifest,
            coefficients,
        })
    }

    /// Subtract the smooth, sensor-family glow field from a RAW plane in
    /// calibrated sensor orientation. Factory defect interpolation is a
    /// separate operation and must be performed before calling this method.
    pub fn correct_calibrated_plane(
        &self,
        camera: &RawCamera,
        raw: &mut [u16],
    ) -> Result<ThermalCorrectionStats> {
        self.correct_calibrated_plane_with_scale(camera, raw, 1)
    }

    /// Apply glow subtraction to samples expressed in Q6 RAW codes. This
    /// preserves fractional correction in a linear 16-bit output and avoids
    /// visible whole-code contour bands around smooth glow gradients.
    pub fn correct_calibrated_plane_q6(
        &self,
        camera: &RawCamera,
        raw: &mut [u16],
    ) -> Result<ThermalCorrectionStats> {
        self.correct_calibrated_plane_with_scale(camera, raw, 64)
    }

    fn correct_calibrated_plane_with_scale(
        &self,
        camera: &RawCamera,
        raw: &mut [u16],
        sample_scale: u16,
    ) -> Result<ThermalCorrectionStats> {
        if camera.width != self.manifest.width
            || camera.height != self.manifest.height
            || raw.len() != camera.width * camera.height
        {
            return Ok(not_applied(
                camera,
                "camera dimensions do not match the sensor-family glow profile",
            ));
        }
        if camera.pattern == SensorPattern::Mono
            && !matches!(camera.name.to_ascii_uppercase().as_str(), "A2" | "C6")
        {
            return Ok(not_applied(
                camera,
                "the bundled monochrome orientation is known only for A2 and C6",
            ));
        }
        let exposure_min_ns = self
            .manifest
            .exposure_min_ns
            .unwrap_or(self.manifest.exposure_ns);
        let exposure_max_ns = self
            .manifest
            .exposure_max_ns
            .unwrap_or(self.manifest.exposure_ns);
        if !(exposure_min_ns..=exposure_max_ns).contains(&camera.exposure_ns) {
            return Ok(not_applied(
                camera,
                "exposure is outside the validated glow-profile range",
            ));
        }
        let exposure_scale = camera.exposure_ns as f32 / self.manifest.exposure_ns as f32;
        let Some(requested_temperature) = camera.sensor_temperature_c else {
            return Ok(not_applied(
                camera,
                "capture has no per-camera sensor temperature",
            ));
        };
        let applied_temperature = (requested_temperature as f32).clamp(
            self.manifest.temperature_min_c as f32,
            self.manifest.temperature_max_c as f32,
        );
        let gain_scale = camera.analog_gain / self.manifest.reference_analog_gain
            * (camera.digital_gain / self.manifest.reference_digital_gain);

        let mut total_change = 0u64;
        let mut maximum_change = 0u16;
        for y in 0..camera.height {
            for x in 0..camera.width {
                let index = y * camera.width + x;
                let (canonical_x, canonical_y) = canonical_coordinates_for_camera(
                    &camera.name,
                    camera.pattern,
                    x,
                    y,
                    camera.width,
                    camera.height,
                );
                let (reference, slope) = self.interpolate(canonical_x, canonical_y);
                let glow = (reference
                    + slope * (applied_temperature - self.manifest.reference_temperature_c))
                    * gain_scale
                    * exposure_scale;
                let corrected = (raw[index] as f32 - glow * f32::from(sample_scale))
                    .round()
                    .clamp(0.0, camera.white_level.max(1.0) * f32::from(sample_scale))
                    as u16;
                let change = raw[index].abs_diff(corrected);
                total_change += u64::from(change);
                maximum_change = maximum_change.max(change);
                raw[index] = corrected;
            }
        }

        Ok(ThermalCorrectionStats {
            applied: true,
            reason: None,
            requested_temperature_c: Some(requested_temperature),
            applied_temperature_c: Some(applied_temperature),
            temperature_clamped: applied_temperature != requested_temperature as f32,
            exposure_scale: Some(exposure_scale),
            mean_absolute_dark_change: total_change as f64
                / raw.len().max(1) as f64
                / f64::from(sample_scale),
            maximum_absolute_dark_change: ((u32::from(maximum_change)
                + u32::from(sample_scale) / 2)
                / u32::from(sample_scale)) as u16,
        })
    }

    fn interpolate(&self, x: usize, y: usize) -> (f32, f32) {
        let grid_width = self.manifest.grid_width;
        let grid_height = self.manifest.grid_height;
        let fx = (((x as f32 + 0.5) * grid_width as f32 / self.manifest.width as f32) - 0.5)
            .clamp(0.0, (grid_width - 1) as f32);
        let fy = (((y as f32 + 0.5) * grid_height as f32 / self.manifest.height as f32) - 0.5)
            .clamp(0.0, (grid_height - 1) as f32);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(grid_width - 1);
        let y1 = (y0 + 1).min(grid_height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let top = mix_pair(self.coefficient(x0, y0), self.coefficient(x1, y0), tx);
        let bottom = mix_pair(self.coefficient(x0, y1), self.coefficient(x1, y1), tx);
        mix_pair(top, bottom, ty)
    }

    fn coefficient(&self, x: usize, y: usize) -> (f32, f32) {
        let offset = (y * self.manifest.grid_width + x) * 4;
        let reference =
            i16::from_le_bytes([self.coefficients[offset], self.coefficients[offset + 1]]) as f32
                / REFERENCE_SCALE;
        let slope =
            i16::from_le_bytes([self.coefficients[offset + 2], self.coefficients[offset + 3]])
                as f32
                / SLOPE_SCALE;
        (reference, slope)
    }
}

fn canonical_coordinates(
    pattern: SensorPattern,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> (usize, usize) {
    match pattern {
        SensorPattern::Rggb => (x, y),
        SensorPattern::Bggr => (width - 1 - x, height - 1 - y),
        SensorPattern::Grbg => (width - 1 - x, y),
        SensorPattern::Gbrg => (x, height - 1 - y),
        SensorPattern::Mono => (width - 1 - x, height - 1 - y),
    }
}

fn canonical_coordinates_for_camera(
    camera_name: &str,
    pattern: SensorPattern,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> (usize, usize) {
    // Monochrome planes provide no CFA layout from which to infer mounting.
    // A2 was measured as horizontally flipped in calibrated sensor space;
    // C6 retains the generic MONO 180-degree transform below.
    if pattern == SensorPattern::Mono && camera_name.eq_ignore_ascii_case("A2") {
        (width - 1 - x, y)
    } else {
        canonical_coordinates(pattern, x, y, width, height)
    }
}

fn mix_pair(left: (f32, f32), right: (f32, f32), amount: f32) -> (f32, f32) {
    (
        left.0 + (right.0 - left.0) * amount,
        left.1 + (right.1 - left.1) * amount,
    )
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for byte in input.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => bail!("bundled glow coefficients contain invalid base64"),
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xff) as u8);
            buffer &= if bits == 0 { 0 } else { (1 << bits) - 1 };
        }
    }
    Ok(output)
}

fn not_applied(camera: &RawCamera, reason: impl Into<String>) -> ThermalCorrectionStats {
    ThermalCorrectionStats {
        reason: Some(reason.into()),
        requested_temperature_c: camera.sensor_temperature_c,
        ..ThermalCorrectionStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfa_orientation_maps_every_layout_to_rggb_coordinates() {
        assert_eq!(
            canonical_coordinates(SensorPattern::Rggb, 1, 2, 8, 6),
            (1, 2)
        );
        assert_eq!(
            canonical_coordinates(SensorPattern::Bggr, 1, 2, 8, 6),
            (6, 3)
        );
        assert_eq!(
            canonical_coordinates(SensorPattern::Grbg, 1, 2, 8, 6),
            (6, 2)
        );
        assert_eq!(
            canonical_coordinates(SensorPattern::Gbrg, 1, 2, 8, 6),
            (1, 3)
        );
        assert_eq!(
            canonical_coordinates(SensorPattern::Mono, 1, 2, 8, 6),
            (6, 3)
        );
    }

    #[test]
    fn monochrome_mountings_are_camera_specific() {
        assert_eq!(
            canonical_coordinates_for_camera("A2", SensorPattern::Mono, 1, 2, 8, 6),
            (6, 2)
        );
        assert_eq!(
            canonical_coordinates_for_camera("C6", SensorPattern::Mono, 1, 2, 8, 6),
            (6, 3)
        );
    }

    #[test]
    fn bundled_profile_is_complete() {
        let profile = ThermalProfile::bundled().unwrap();
        assert_eq!(profile.coefficients.len(), 64 * 48 * 4);
        assert_eq!(profile.manifest.exposure_ns, 14_999_805_952);
        assert_eq!(profile.manifest.exposure_min_ns, Some(4_999_935_488));
        assert_eq!(profile.manifest.exposure_max_ns, Some(14_999_805_952));
    }

    #[test]
    fn q6_correction_preserves_fractional_raw_codes() {
        let manifest = ThermalProfileManifest {
            format: "chiaro-sensor-glow-profile".to_owned(),
            version: PROFILE_VERSION,
            source: "test".to_owned(),
            sensor_family: "test".to_owned(),
            width: 2,
            height: 2,
            grid_width: 2,
            grid_height: 2,
            reference_temperature_c: 20.0,
            temperature_min_c: 20,
            temperature_max_c: 20,
            exposure_ns: 1,
            exposure_min_ns: None,
            exposure_max_ns: None,
            reference_analog_gain: 1.0,
            reference_digital_gain: 1.0,
            contributing_cameras: Vec::new(),
            coefficients: "test".to_owned(),
            coefficient_layout: "test".to_owned(),
            orientation_rule: "test".to_owned(),
        };
        let mut coefficients = Vec::new();
        for _ in 0..4 {
            coefficients.extend_from_slice(&64i16.to_le_bytes()); // 0.25 RAW code.
            coefficients.extend_from_slice(&0i16.to_le_bytes());
        }
        let profile = ThermalProfile::from_parts(PathBuf::new(), manifest, coefficients).unwrap();
        let camera = RawCamera {
            id: 0,
            name: "A1".to_owned(),
            width: 2,
            height: 2,
            row_stride: 0,
            absolute_offset: 0,
            byte_len: 0,
            pattern: SensorPattern::Rggb,
            sensor_temperature_c: Some(20),
            analog_gain: 1.0,
            digital_gain: 1.0,
            exposure_ns: 1,
            black_level: 0.0,
            white_level: 1023.0,
        };
        let mut whole_codes = vec![100; 4];
        profile
            .correct_calibrated_plane(&camera, &mut whole_codes)
            .unwrap();
        assert_eq!(whole_codes, vec![100; 4]);

        let mut q6_codes = vec![100 << 6; 4];
        profile
            .correct_calibrated_plane_q6(&camera, &mut q6_codes)
            .unwrap();
        assert_eq!(q6_codes, vec![(100 << 6) - 16; 4]);
    }
}
