use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use chiaro::lri::{RawCamera, SensorPattern};

const PROFILE_FORMAT: &str = "chiaro-universal-hotpixel-profile";
const PROFILE_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize)]
pub struct UniversalHotpixelProfile {
    pub format: String,
    pub version: u32,
    pub training_quantile: f32,
    pub reference_temperature_c: f32,
    pub reference_exposure_ns: u64,
    pub reference_analog_gain: f32,
    pub reference_digital_gain: f32,
    pub temperature_nodes_c: Vec<f32>,
    pub families: UniversalFamilies,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UniversalFamilies {
    pub color: Vec<SeverityResponse>,
    pub mono: Vec<SeverityResponse>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SeverityResponse {
    pub severity_min: u8,
    pub severity_max_exclusive: u16,
    pub excess_raw_codes: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct UniversalHotpixelStats {
    pub applied: bool,
    pub reason: Option<String>,
    pub requested_temperature_c: Option<i32>,
    pub applied_temperature_c: Option<f32>,
    pub temperature_clamped: bool,
    pub exposure_scale: Option<f32>,
    pub analog_gain_scale: Option<f32>,
    pub digital_gain_scale: Option<f32>,
    pub active_pixels: usize,
}

impl UniversalHotpixelProfile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let profile: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn bundled() -> Result<Self> {
        let profile: Self =
            serde_json::from_str(include_str!("../assets/l16-universal-hotpixel-v1.json"))
                .context("parse bundled universal hotpixel profile")?;
        profile.validate()?;
        Ok(profile)
    }

    fn validate(&self) -> Result<()> {
        if self.format != PROFILE_FORMAT || self.version != PROFILE_VERSION {
            bail!(
                "unsupported universal hotpixel profile format/version: {}/{}",
                self.format,
                self.version
            );
        }
        if self.temperature_nodes_c.len() < 2
            || self.reference_exposure_ns == 0
            || self.reference_analog_gain <= 0.0
            || self.reference_digital_gain <= 0.0
            || !(0.0..1.0).contains(&self.training_quantile)
        {
            bail!("invalid universal hotpixel profile metadata");
        }
        for family in [&self.families.color, &self.families.mono] {
            for response in family {
                if response.severity_max_exclusive <= u16::from(response.severity_min)
                    || response.excess_raw_codes.len() != self.temperature_nodes_c.len()
                {
                    bail!("invalid universal hotpixel severity response");
                }
            }
        }
        Ok(())
    }

    pub fn active_map(
        &self,
        camera: &RawCamera,
        severity_map: &[u8],
        activation_threshold: f32,
    ) -> (Vec<bool>, UniversalHotpixelStats) {
        let mut active = vec![false; severity_map.len()];
        let requested_temperature = camera.sensor_temperature_c;
        if severity_map.len() != camera.width * camera.height {
            return (
                active,
                UniversalHotpixelStats {
                    reason: Some("factory map dimensions do not match camera".to_owned()),
                    requested_temperature_c: requested_temperature,
                    ..UniversalHotpixelStats::default()
                },
            );
        }
        let Some(requested_temperature) = requested_temperature else {
            return (
                active,
                UniversalHotpixelStats {
                    reason: Some("capture has no sensor temperature".to_owned()),
                    ..UniversalHotpixelStats::default()
                },
            );
        };
        if camera.exposure_ns == 0 || camera.analog_gain <= 0.0 || camera.digital_gain <= 0.0 {
            return (
                active,
                UniversalHotpixelStats {
                    reason: Some("capture has incomplete exposure/gain metadata".to_owned()),
                    requested_temperature_c: Some(requested_temperature),
                    ..UniversalHotpixelStats::default()
                },
            );
        }

        let minimum_temperature = self.temperature_nodes_c[0];
        let maximum_temperature = *self.temperature_nodes_c.last().unwrap();
        let temperature =
            (requested_temperature as f32).clamp(minimum_temperature, maximum_temperature);
        let exposure_scale = camera.exposure_ns as f32 / self.reference_exposure_ns as f32;
        let analog_gain_scale = camera.analog_gain / self.reference_analog_gain;
        let digital_gain_scale = camera.digital_gain / self.reference_digital_gain;
        let total_scale = exposure_scale * analog_gain_scale * digital_gain_scale;
        let family = if camera.pattern == SensorPattern::Mono {
            &self.families.mono
        } else {
            &self.families.color
        };

        let mut active_severities = [false; 256];
        for severity in 1u8..=254 {
            let Some(response) = family.iter().find(|response| {
                severity >= response.severity_min
                    && u16::from(severity) < response.severity_max_exclusive
            }) else {
                continue;
            };
            let predicted = interpolate(
                &self.temperature_nodes_c,
                &response.excess_raw_codes,
                temperature,
            ) * total_scale;
            if predicted >= activation_threshold {
                active_severities[usize::from(severity)] = true;
            }
        }
        let mut active_pixels = 0usize;
        for (index, &severity) in severity_map.iter().enumerate() {
            if active_severities[usize::from(severity)] {
                active[index] = true;
                active_pixels += 1;
            }
        }

        (
            active,
            UniversalHotpixelStats {
                applied: true,
                reason: None,
                requested_temperature_c: Some(requested_temperature),
                applied_temperature_c: Some(temperature),
                temperature_clamped: temperature != requested_temperature as f32,
                exposure_scale: Some(exposure_scale),
                analog_gain_scale: Some(analog_gain_scale),
                digital_gain_scale: Some(digital_gain_scale),
                active_pixels,
            },
        )
    }
}

fn interpolate(nodes: &[f32], values: &[f32], input: f32) -> f32 {
    if input <= nodes[0] {
        return values[0];
    }
    if input >= nodes[nodes.len() - 1] {
        return values[values.len() - 1];
    }
    let upper = nodes.partition_point(|node| *node < input);
    let lower = upper - 1;
    let fraction = (input - nodes[lower]) / (nodes[upper] - nodes[lower]);
    values[lower] + fraction * (values[upper] - values[lower])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(temperature: Option<i32>) -> RawCamera {
        RawCamera {
            id: 0,
            name: "A1".to_owned(),
            width: 3,
            height: 1,
            row_stride: 0,
            absolute_offset: 0,
            byte_len: 0,
            pattern: SensorPattern::Rggb,
            sensor_temperature_c: temperature,
            analog_gain: 6.25,
            digital_gain: 1.015625,
            exposure_ns: 14_999_805_952,
            black_level: 0.0,
            white_level: 1023.0,
        }
    }

    #[test]
    fn bundled_profile_is_complete_for_color_and_mono() {
        let profile = UniversalHotpixelProfile::bundled().unwrap();
        assert_eq!(profile.families.color.len(), 8);
        assert_eq!(profile.families.mono.len(), 8);
        assert_eq!(
            profile.temperature_nodes_c,
            [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
        );
    }

    #[test]
    fn interpolation_clamps_and_interpolates() {
        let nodes = [10.0, 20.0, 30.0];
        let values = [1.0, 3.0, 9.0];
        assert_eq!(interpolate(&nodes, &values, 5.0), 1.0);
        assert_eq!(interpolate(&nodes, &values, 15.0), 2.0);
        assert_eq!(interpolate(&nodes, &values, 35.0), 9.0);
    }

    #[test]
    fn active_map_uses_factory_coordinates_but_never_forces_class_255() {
        let profile = UniversalHotpixelProfile::bundled().unwrap();
        let (active, stats) = profile.active_map(&camera(Some(40)), &[16, 255, 0], 4.0);
        assert_eq!(active, [true, false, false]);
        assert!(stats.applied);
        assert_eq!(stats.active_pixels, 1);
    }

    #[test]
    fn missing_temperature_skips_only_the_universal_decision() {
        let profile = UniversalHotpixelProfile::bundled().unwrap();
        let (active, stats) = profile.active_map(&camera(None), &[16, 32, 64], 4.0);
        assert_eq!(active, [false, false, false]);
        assert!(!stats.applied);
        assert_eq!(
            stats.reason.as_deref(),
            Some("capture has no sensor temperature")
        );
    }
}
