//! High-level, UI-independent correction pipeline for one camera frame.
//!
//! The pipeline takes a camera's packed RAW10 surface (or an entire LRI plus its
//! [`RawCamera`] layout record), applies the configured correction stages in
//! the validated order, and returns linear Q6 samples together with per-stage
//! statistics. It never touches the filesystem except through the explicit
//! [`CorrectedFrame::write_png`] helper, so it can be driven equally well by
//! the `chiaro-hotpixel` CLI and by an interactive application.
//!
//! Stage order:
//!
//! 1. Unpack RAW10 (`extract_raw_plane`).
//! 2. Predict universally active factory-listed pixels from temperature,
//!    exposure, and gain (`UniversalHotpixelProfile`).
//! 3. Optionally subtract camera-specific row/column bias and merge its own
//!    temperature-active defect decisions (`CleanupCameraProfile`).
//! 4. Interpolate factory hot/dead pixels with same-color neighbours
//!    (`correct_hot_pixels_with_forced_map`).
//! 5. Subtract the smooth sensor-family corner glow (`ThermalProfile`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use chiaro::lri::{RawCamera, SensorPattern};

use crate::cleanup::{CleanupCameraProfile, CleanupCorrectionStats};
use crate::correct::{
    CorrectionConfig, CorrectionStats, correct_hot_pixels_threaded, demosaic_rows,
};
use crate::demosaic::{DemosaicMethod, demosaic};
use crate::png16::{
    DEFAULT_DEFLATE_LEVEL, PngColor, samples_to_be_bytes, write_png16_streaming_atomic_with_level,
};
use crate::raw10::unpack_l16_10bit_threaded;
use crate::thermal::{ThermalCorrectionStats, ThermalProfile};
use crate::universal_hotpixel::{UniversalHotpixelProfile, UniversalHotpixelStats};

/// Scale between a 10-bit RAW code and the linear 16-bit output (`1 << 6`).
pub const Q6_SCALE: u16 = 64;

/// How Bayer cameras are laid out in the output image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// Bayer cameras become linear 16-bit RGB. Monochrome cameras remain
    /// 16-bit grayscale.
    #[default]
    Rgb,
    /// Preserve Bayer mosaics as linear 16-bit grayscale.
    Mosaic,
}

impl OutputMode {
    /// PNG colour type that a frame from `pattern` produces in this mode.
    pub fn png_color_type(self, pattern: SensorPattern) -> &'static str {
        if self == OutputMode::Rgb && pattern != SensorPattern::Mono {
            "RGB16"
        } else {
            "GRAY16"
        }
    }

    /// Short description of the colour handling used for run manifests.
    pub fn color_processing(self) -> &'static str {
        match self {
            OutputMode::Rgb => "Bayer cameras: selectable demosaic; mono: grayscale",
            OutputMode::Mosaic => "corrected RAW mosaic/grayscale; no demosaic",
        }
    }
}

/// Availability of the optional camera-specific cleanup stage.
#[derive(Clone, Copy, Debug, Default)]
pub enum CleanupStage<'a> {
    /// No cleanup profile was supplied.
    #[default]
    Disabled,
    /// A cleanup profile was supplied but contains no entry for this camera.
    NotCalibrated,
    /// Apply this camera's learned defect and row/column model.
    Profile(&'a CleanupCameraProfile),
}

impl<'a> CleanupStage<'a> {
    /// Build the stage from a `CleanupProfile::load_camera` result when a
    /// profile was supplied, or `Disabled` when it was not.
    pub fn from_loaded(requested: bool, camera: Option<&'a CleanupCameraProfile>) -> Self {
        match (requested, camera) {
            (_, Some(profile)) => Self::Profile(profile),
            (true, None) => Self::NotCalibrated,
            (false, None) => Self::Disabled,
        }
    }

    fn profile(self) -> Option<&'a CleanupCameraProfile> {
        match self {
            Self::Profile(profile) => Some(profile),
            _ => None,
        }
    }
}

/// Correction stages and their models for one camera.
///
/// All model references are borrowed so a single set of loaded profiles can be
/// shared across cameras, frames, and worker threads.
#[derive(Clone, Debug, Default)]
pub struct FramePipeline<'a> {
    /// Factory-guided hot-pixel interpolation settings.
    pub config: CorrectionConfig,
    /// Coordinate-free temperature/exposure/gain prior; `None` disables it.
    pub universal_hotpixel: Option<&'a UniversalHotpixelProfile>,
    /// Sensor-family corner-glow model; `None` disables it.
    pub thermal: Option<&'a ThermalProfile>,
    /// Optional camera-specific defect and line cleanup.
    pub cleanup: CleanupStage<'a>,
    /// Worker threads for the per-frame kernels; `0` uses every core. Work is
    /// split across the rows of one frame, so memory stays at one frame
    /// regardless of the thread count.
    pub threads: usize,
}

/// Result of correcting one camera frame.
#[derive(Clone, Debug)]
pub struct CorrectedFrame {
    pub width: usize,
    pub height: usize,
    pub pattern: SensorPattern,
    /// Linear samples in Q6 RAW codes (`raw << 6`) in decoded stream order
    /// (mosaic or grayscale). Untouched RAW10 samples remain exact multiples
    /// of [`Q6_SCALE`]. `pattern` describes the calibration raster; see
    /// [`Self::stream_pattern`].
    pub samples_q6: Vec<u16>,
    pub hotpixel: CorrectionStats,
    pub universal_hotpixel: UniversalHotpixelStats,
    pub thermal: ThermalCorrectionStats,
    pub cleanup: CleanupCorrectionStats,
}

impl CorrectedFrame {
    /// Fraction of all samples replaced by the factory-guided correction.
    pub fn corrected_fraction(&self) -> f64 {
        if self.samples_q6.is_empty() {
            0.0
        } else {
            self.hotpixel.corrected as f64 / self.samples_q6.len() as f64
        }
    }

    /// Demosaic the corrected Bayer mosaic into interleaved linear 16-bit RGB.
    /// Fails for monochrome cameras, which need no demosaicing.
    pub fn to_rgb16(&self) -> Result<Vec<u16>> {
        self.to_rgb16_with_method(DemosaicMethod::default(), 0)
    }

    /// Demosaic using an explicit method and thread count (`0` = auto).
    pub fn to_rgb16_with_method(&self, method: DemosaicMethod, threads: usize) -> Result<Vec<u16>> {
        demosaic(
            &self.samples_q6,
            self.width,
            self.height,
            self.stream_pattern(),
            method,
            threads,
        )
    }

    /// CFA layout of `samples_q6`, which are in decoded stream order, while
    /// `pattern` is recorded for the calibration raster (rotated 180 degrees).
    pub fn stream_pattern(&self) -> SensorPattern {
        self.pattern.rotated_180()
    }

    /// Write the frame as a linear 16-bit PNG, atomically replacing `path`,
    /// using every core. Returns the PNG colour type that was written
    /// (`"RGB16"` or `"GRAY16"`).
    pub fn write_png(&self, path: &Path, mode: OutputMode) -> Result<&'static str> {
        self.write_png_with_threads(path, mode, 0)
    }

    /// [`Self::write_png`] with an explicit render thread count (`0` = auto).
    pub fn write_png_with_threads(
        &self,
        path: &Path,
        mode: OutputMode,
        threads: usize,
    ) -> Result<&'static str> {
        self.write_png_with_options(path, mode, threads, DEFAULT_DEFLATE_LEVEL)
    }

    /// [`Self::write_png`] with explicit threads (`0` = auto) and deflate
    /// level (`0` stores rows uncompressed, `2` is the default trade-off).
    ///
    /// Demosaicing happens per row band directly into the encoder's stream,
    /// so the full RGB frame is never materialised.
    pub fn write_png_with_options(
        &self,
        path: &Path,
        mode: OutputMode,
        threads: usize,
        deflate_level: u32,
    ) -> Result<&'static str> {
        self.write_png_with_demosaic_options(
            path,
            mode,
            DemosaicMethod::default(),
            threads,
            deflate_level,
        )
    }

    /// Write with an explicit Bayer reconstruction method.
    pub fn write_png_with_demosaic_options(
        &self,
        path: &Path,
        mode: OutputMode,
        demosaic_method: DemosaicMethod,
        threads: usize,
        deflate_level: u32,
    ) -> Result<&'static str> {
        let (width, height, pattern) = (self.width, self.height, self.pattern);
        if mode == OutputMode::Rgb && pattern != SensorPattern::Mono {
            let stream_pattern = self.stream_pattern();
            if demosaic_method == DemosaicMethod::Simple {
                write_png16_streaming_atomic_with_level(
                    path,
                    width,
                    height,
                    PngColor::Rgb16,
                    threads,
                    deflate_level,
                    |rows, bytes| {
                        let mut band = vec![0u16; rows.len() * width * 3];
                        demosaic_rows(
                            &self.samples_q6,
                            width,
                            height,
                            stream_pattern,
                            rows,
                            &mut band,
                        );
                        samples_to_be_bytes(&band, bytes);
                    },
                )?;
            } else {
                let rgb = demosaic(
                    &self.samples_q6,
                    width,
                    height,
                    stream_pattern,
                    demosaic_method,
                    threads,
                )?;
                write_png16_streaming_atomic_with_level(
                    path,
                    width,
                    height,
                    PngColor::Rgb16,
                    threads,
                    deflate_level,
                    |rows, bytes| {
                        samples_to_be_bytes(
                            &rgb[rows.start * width * 3..rows.end * width * 3],
                            bytes,
                        );
                    },
                )?;
            }
        } else {
            write_png16_streaming_atomic_with_level(
                path,
                width,
                height,
                PngColor::Gray16,
                threads,
                deflate_level,
                |rows, bytes| {
                    samples_to_be_bytes(
                        &self.samples_q6[rows.start * width..rows.end * width],
                        bytes,
                    );
                },
            )?;
        }
        Ok(mode.png_color_type(pattern))
    }
}

/// Unpack one camera's tightly packed RAW10 surface from a complete LRI.
pub fn extract_raw_plane(lri: &[u8], camera: &RawCamera) -> Result<Vec<u16>> {
    extract_raw_plane_threaded(lri, camera, 1)
}

/// [`extract_raw_plane`] with the unpacking split across `threads` row bands.
pub fn extract_raw_plane_threaded(
    lri: &[u8],
    camera: &RawCamera,
    threads: usize,
) -> Result<Vec<u16>> {
    let start = camera.absolute_offset;
    let end = start
        .checked_add(camera.byte_len)
        .context("RAW span overflows")?;
    if end > lri.len() {
        bail!("RAW span for {} lies outside the LRI", camera.name);
    }
    let expected_packed = camera
        .width
        .checked_mul(camera.height)
        .and_then(|samples| samples.checked_mul(5))
        .map(|bytes| bytes / 4)
        .context("RAW dimensions overflow")?;
    if camera.byte_len != expected_packed {
        bail!(
            "{} uses a padded RAW stride ({} bytes, expected {}); this version supports tightly packed RAW10",
            camera.name,
            camera.byte_len,
            expected_packed
        );
    }
    unpack_l16_10bit_threaded(&lri[start..end], camera.width, camera.height, threads)
}

impl FramePipeline<'_> {
    /// Extract and correct `camera` from a complete in-memory LRI.
    ///
    /// `severity_map` is the factory map already rotated into decoded RAW
    /// coordinates, as returned by `HotpixelRec::load_rotated_map`.
    pub fn correct_lri(
        &self,
        lri: &[u8],
        camera: &RawCamera,
        severity_map: &[u8],
    ) -> Result<CorrectedFrame> {
        let raw = extract_raw_plane_threaded(lri, camera, self.threads)?;
        self.correct_raw(camera, raw, severity_map)
    }

    /// Correct an already unpacked 10-bit RAW plane for `camera`.
    ///
    /// `raw` holds `camera.width * camera.height` samples in decoded RAW
    /// order; `severity_map` is the matching rotated factory map.
    pub fn correct_raw(
        &self,
        camera: &RawCamera,
        mut raw: Vec<u16>,
        severity_map: &[u8],
    ) -> Result<CorrectedFrame> {
        let sample_count = camera
            .width
            .checked_mul(camera.height)
            .context("RAW dimensions overflow")?;
        if raw.len() != sample_count {
            bail!(
                "{} RAW plane has {} samples; expected {}x{}",
                camera.name,
                raw.len(),
                camera.width,
                camera.height
            );
        }
        if severity_map.len() != sample_count {
            bail!("RAW and hotpixel map dimensions differ");
        }
        let config = &self.config;
        let activation_threshold = config.absolute_threshold as f32;

        // Stage 2: coordinate-free universal prior.
        let (mut forced_map, universal_hotpixel_stats) = match self.universal_hotpixel {
            Some(profile) => {
                let (active, stats) =
                    profile.active_map(camera, severity_map, activation_threshold);
                (stats.applied.then_some(active), stats)
            }
            None => (
                None,
                UniversalHotpixelStats {
                    reason: Some("bundled universal hotpixel model disabled".to_owned()),
                    requested_temperature_c: camera.sensor_temperature_c,
                    ..UniversalHotpixelStats::default()
                },
            ),
        };

        // Stages 3 and 4: optional camera-specific cleanup, then factory-guided
        // interpolation. Cleanup works in Q6 so that row/column bias keeps its
        // fractional precision; the interpolation then runs in the same domain.
        let mut cleanup_stats = CleanupCorrectionStats::default();
        let (hotpixel_stats, mut samples_q6) = if let Some(profile) = self.cleanup.profile() {
            let mut samples_q6 = promote_q6(&raw);
            cleanup_stats = profile.correct_q6(camera, &mut samples_q6, activation_threshold);
            if let Ok(personal_map) = profile.temperature_active_map(camera, activation_threshold) {
                let combined = forced_map.get_or_insert_with(|| vec![false; personal_map.len()]);
                for (active, personal) in combined.iter_mut().zip(personal_map) {
                    *active |= personal;
                }
            }
            let mut q6_config = config.clone();
            q6_config.absolute_threshold = config
                .absolute_threshold
                .saturating_mul(i32::from(Q6_SCALE));
            let mut stats = correct_hot_pixels_threaded(
                &mut samples_q6,
                camera.width,
                camera.height,
                camera.pattern,
                severity_map,
                forced_map.as_deref(),
                &q6_config,
                self.threads,
            )?;
            // Report the interpolation statistics in whole RAW codes.
            stats.mean_absolute_change /= f64::from(Q6_SCALE);
            stats.maximum_absolute_change = ((u32::from(stats.maximum_absolute_change)
                + u32::from(Q6_SCALE) / 2)
                / u32::from(Q6_SCALE)) as u16;
            (stats, samples_q6)
        } else {
            let stats = correct_hot_pixels_threaded(
                &mut raw,
                camera.width,
                camera.height,
                camera.pattern,
                severity_map,
                forced_map.as_deref(),
                config,
                self.threads,
            )?;
            // Promote RAW10 to Q6 before glow subtraction. The original samples
            // remain exact multiples of 64, while the smooth model can retain
            // sub-code detail.
            (stats, promote_q6(&raw))
        };
        if matches!(self.cleanup, CleanupStage::NotCalibrated) {
            cleanup_stats.reason = Some("cleanup profile has no entry for this camera".to_owned());
        }

        // Stage 5: sensor-family corner glow. The glow model is expressed in
        // calibrated sensor orientation, which is the decoded RAW rotated by
        // 180 degrees; the RAW-plane entry point folds that rotation into its
        // coordinate mapping.
        let mut thermal_stats = ThermalCorrectionStats::default();
        if let Some(profile) = self.thermal {
            thermal_stats = profile.correct_raw_plane_q6(camera, &mut samples_q6, self.threads)?;
        }

        Ok(CorrectedFrame {
            width: camera.width,
            height: camera.height,
            pattern: camera.pattern,
            samples_q6,
            hotpixel: hotpixel_stats,
            universal_hotpixel: universal_hotpixel_stats,
            thermal: thermal_stats,
            cleanup: cleanup_stats,
        })
    }
}

fn promote_q6(raw: &[u16]) -> Vec<u16> {
    raw.iter().map(|sample| sample << 6).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw10::pack_l16_10bit;

    fn camera(width: usize, height: usize, pattern: SensorPattern) -> RawCamera {
        RawCamera {
            id: 0,
            name: "A1".to_owned(),
            width,
            height,
            row_stride: 0,
            absolute_offset: 0,
            byte_len: width * height * 5 / 4,
            pattern,
            sensor_temperature_c: None,
            analog_gain: 1.0,
            digital_gain: 1.0,
            exposure_ns: 0,
            black_level: 0.0,
            white_level: 1023.0,
        }
    }

    #[test]
    fn extracts_raw_plane_from_lri_bytes() {
        let samples: Vec<u16> = (0..32).map(|index| (index * 29 + 3) & 1023).collect();
        let packed = pack_l16_10bit(&samples).unwrap();
        let mut lri = vec![0xAAu8; 7];
        let mut camera = camera(8, 4, SensorPattern::Rggb);
        camera.absolute_offset = lri.len();
        lri.extend_from_slice(&packed);
        lri.extend_from_slice(&[0x55; 3]);

        assert_eq!(extract_raw_plane(&lri, &camera).unwrap(), samples);

        camera.byte_len += 1;
        assert!(extract_raw_plane(&lri, &camera).is_err());
    }

    #[test]
    fn pipeline_without_models_interpolates_factory_outliers_in_q6() {
        let width = 8;
        let height = 8;
        let camera = camera(width, height, SensorPattern::Mono);
        let mut raw = vec![100u16; width * height];
        let hot = 3 * width + 4;
        raw[hot] = 900;
        let mut severity = vec![0u8; raw.len()];
        severity[hot] = 200;
        // A listed pixel that is not an outlier is left untouched in adaptive mode.
        severity[5 * width + 2] = 200;

        let pipeline = FramePipeline::default();
        let frame = pipeline.correct_raw(&camera, raw, &severity).unwrap();

        assert_eq!(frame.hotpixel.candidates, 2);
        assert_eq!(frame.hotpixel.corrected, 1);
        assert_eq!(frame.samples_q6[hot], 100 << 6);
        assert!(frame.samples_q6.iter().all(|&value| value == 100 << 6));
        assert!(!frame.universal_hotpixel.applied);
        assert!(!frame.thermal.applied);
        assert!(!frame.cleanup.applied);
        assert!(frame.cleanup.reason.is_none());
        assert_eq!(frame.corrected_fraction(), 1.0 / 64.0);
        assert_eq!(OutputMode::Rgb.png_color_type(frame.pattern), "GRAY16");
        assert_eq!(OutputMode::Rgb.png_color_type(SensorPattern::Rggb), "RGB16");
        assert_eq!(
            OutputMode::Mosaic.png_color_type(SensorPattern::Rggb),
            "GRAY16"
        );
    }

    #[test]
    fn missing_cleanup_entry_is_reported() {
        let camera = camera(4, 4, SensorPattern::Mono);
        let pipeline = FramePipeline {
            cleanup: CleanupStage::NotCalibrated,
            ..FramePipeline::default()
        };
        let frame = pipeline
            .correct_raw(&camera, vec![10; 16], &[0; 16])
            .unwrap();
        assert_eq!(
            frame.cleanup.reason.as_deref(),
            Some("cleanup profile has no entry for this camera")
        );
    }

    #[test]
    fn writes_rgb_for_bayer_and_gray_for_mosaic_or_mono() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = FramePipeline::default();

        let bayer = camera(4, 4, SensorPattern::Rggb);
        let frame = pipeline
            .correct_raw(&bayer, vec![100; 16], &[0; 16])
            .unwrap();
        let rgb_path = dir.path().join("rgb.png");
        assert_eq!(
            frame.write_png(&rgb_path, OutputMode::Rgb).unwrap(),
            "RGB16"
        );
        let mosaic_path = dir.path().join("mosaic.png");
        assert_eq!(
            frame.write_png(&mosaic_path, OutputMode::Mosaic).unwrap(),
            "GRAY16"
        );
        assert_eq!(frame.to_rgb16().unwrap().len(), 16 * 3);

        let mono = camera(4, 4, SensorPattern::Mono);
        let frame = pipeline
            .correct_raw(&mono, vec![100; 16], &[0; 16])
            .unwrap();
        let mono_path = dir.path().join("mono.png");
        assert_eq!(
            frame.write_png(&mono_path, OutputMode::Rgb).unwrap(),
            "GRAY16"
        );
        assert!(frame.to_rgb16().is_err());

        for path in [rgb_path, mosaic_path, mono_path] {
            assert!(path.is_file());
            assert!(!path.with_extension("png.part").exists());
        }
    }

    #[test]
    fn rejects_mismatched_dimensions() {
        let camera = camera(4, 4, SensorPattern::Mono);
        let pipeline = FramePipeline::default();
        assert!(
            pipeline
                .correct_raw(&camera, vec![0; 15], &[0; 16])
                .is_err()
        );
        assert!(
            pipeline
                .correct_raw(&camera, vec![0; 16], &[0; 15])
                .is_err()
        );
    }
}
