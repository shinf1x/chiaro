//! Factory geometric calibration, resolved from the sparse protobuf fragments
//! found in a capture and in device calibration files.
//!
//! Captures embed most of the device's `calibration.lri` byte for byte, but
//! not all of it (`zoom_calib_v0.lri` adds mirror aiming data, and some
//! modules carry extra focus bundles only in the capture). Resolution is
//! therefore field-oriented: intrinsics are the union of every distinct
//! `(focus Hall code, K)` pair, while poses and mirror systems take the first
//! record that carries them, with the capture's own headers consulted first.
//!
//! The conventions implemented here are the ones validated on real captures in
//! the companion research notes (`docs/coordinate-conventions.md`): canonical
//! extrinsics are world-to-camera, the mirror matrix is camera-to-world, and
//! `flip_img_around_x` selects which image axis a mirror reflects.

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
};

use anyhow::{Context, Result, bail};
use chiaro::lri::{SensorNoiseProfile, inspect_lelr_block_header, parse_sensor_noise_profile};
use chiaro_proto::{
    Message,
    geometric_calibration::geometric_calibration::MirrorType,
    lightheader::LightHeader,
    matrix3x3f::Matrix3x3F,
    mirror_system::{
        MirrorActuatorMapping, MirrorSystem, mirror_actuator_mapping::TransformationType,
    },
    point2f::Point2F,
    point3f::Point3F,
    view_preferences::ViewPreferences,
};
use serde::Serialize;

use crate::math::{Mat3, Vec2, Vec3};

const LELR_HEADER_SIZE: usize = 32;
pub const CAMERA_NAMES: [&str; 16] = [
    "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "C6",
];

/// Every `LightHeader` (block type 0) and `ViewPreferences` (type 1) message in
/// an LRI, in file order.
pub struct LriMessages {
    pub headers: Vec<LightHeader>,
    pub view_preferences: Vec<ViewPreferences>,
}

/// Stable identity of the physical L16 that produced an LRI. Device-specific
/// calibration must never cross this boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub low: u64,
    pub high: u64,
}

impl DeviceId {
    /// Filesystem-safe fixed-width representation.
    pub fn cache_key(self) -> String {
        format!("{:016x}{:016x}", self.high, self.low)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.cache_key())
    }
}

impl LriMessages {
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut headers = Vec::new();
        let mut view_preferences = Vec::new();
        let mut offset = 0usize;
        while offset < data.len() {
            let header = data
                .get(offset..offset + LELR_HEADER_SIZE)
                .context("truncated LELR block header")?;
            let block =
                inspect_lelr_block_header(header, offset).map_err(|e| anyhow::anyhow!("{e}"))?;
            let message = &data[block.message_start..block.message_start + block.message_length];
            match block.message_type {
                0 => headers
                    .push(LightHeader::parse_from_bytes(message).context("decode LightHeader")?),
                1 => view_preferences.push(
                    ViewPreferences::parse_from_bytes(message).context("decode ViewPreferences")?,
                ),
                _ => {}
            }
            offset = block.block_offset + block.block_length;
        }
        if headers.is_empty() {
            bail!("LRI contains no LightHeader block");
        }
        Ok(Self {
            headers,
            view_preferences,
        })
    }

    /// Physical device recorded in this file, if both halves are present.
    pub fn device_id(&self) -> Option<DeviceId> {
        self.headers.iter().find_map(|header| {
            Some(DeviceId {
                low: header.device_unique_id_low?,
                high: header.device_unique_id_high?,
            })
        })
    }
}

fn matrix3(message: &Matrix3x3F) -> Mat3 {
    let f = |v: Option<f32>| f64::from(v.unwrap_or(0.0));
    [
        [f(message.x00), f(message.x01), f(message.x02)],
        [f(message.x10), f(message.x11), f(message.x12)],
        [f(message.x20), f(message.x21), f(message.x22)],
    ]
}

fn point3(message: &Point3F) -> Vec3 {
    [
        f64::from(message.x.unwrap_or(0.0)),
        f64::from(message.y.unwrap_or(0.0)),
        f64::from(message.z.unwrap_or(0.0)),
    ]
}

fn point2(message: &Point2F) -> Vec2 {
    [
        f64::from(message.x.unwrap_or(0.0)),
        f64::from(message.y.unwrap_or(0.0)),
    ]
}

/// One calibrated focus state.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicsBundle {
    pub hall_code: Option<f64>,
    pub focus_distance: f64,
    pub k: Mat3,
}

/// World-to-camera extrinsics: `X_c = R X_w + t`.
#[derive(Clone, Debug)]
pub struct CanonicalPose {
    pub rotation_wc: Mat3,
    pub translation_wc: Vec3,
}

impl CanonicalPose {
    pub fn center_world(&self) -> Vec3 {
        let r_cw = crate::math::transpose(&self.rotation_wc);
        crate::math::scale(crate::math::mul_vec(&r_cw, self.translation_wc), -1.0)
    }
}

/// Hall code to mirror angle, see `docs/calibration-model.md`.
#[derive(Clone, Debug)]
pub struct MirrorActuator {
    pub mean_std_normalize: bool,
    pub actuator_length_offset: f64,
    pub actuator_length_scale: f64,
    pub mirror_angle_offset: f64,
    pub mirror_angle_scale: f64,
    /// Measured `(hall_code, angle_degrees)` pairs.
    pub hall_angle_pairs: Vec<(f64, f64)>,
    /// Six coefficients forming two candidate quadratic branches.
    pub quadratic_coeffs: Vec<f64>,
}

impl MirrorActuator {
    fn normalized(&self, hall_code: f64) -> f64 {
        (self.actuator_length_offset - hall_code) / self.actuator_length_scale
    }

    fn branch_rmse(&self, coeffs: &[f64]) -> f64 {
        if self.hall_angle_pairs.is_empty() || self.mirror_angle_scale == 0.0 {
            return f64::INFINITY;
        }
        let sum: f64 = self
            .hall_angle_pairs
            .iter()
            .map(|&(hall, angle)| {
                let x = self.normalized(hall);
                let observed = (angle - self.mirror_angle_offset) / self.mirror_angle_scale;
                let predicted = coeffs[0] * x * x + coeffs[1] * x + coeffs[2];
                (predicted - observed).powi(2)
            })
            .sum();
        (sum / self.hall_angle_pairs.len() as f64).sqrt()
    }

    /// The quadratic branch that reproduces the measured pairs best.
    pub fn selected_branch(&self) -> Option<usize> {
        if !self.mean_std_normalize
            || self.quadratic_coeffs.len() != 6
            || self.actuator_length_scale == 0.0
        {
            return None;
        }
        let first = self.branch_rmse(&self.quadratic_coeffs[..3]);
        let second = self.branch_rmse(&self.quadratic_coeffs[3..]);
        Some(if second < first { 1 } else { 0 })
    }

    /// Linear interpolation through the measured pairs.
    fn interpolated_angle(&self, hall_code: f64) -> Result<f64> {
        if self.hall_angle_pairs.len() < 2 {
            bail!("mirror calibration has fewer than two Hall/angle pairs");
        }
        let mut pairs = self.hall_angle_pairs.clone();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        if hall_code <= pairs[0].0 {
            return Ok(pairs[0].1);
        }
        if hall_code >= pairs[pairs.len() - 1].0 {
            return Ok(pairs[pairs.len() - 1].1);
        }
        for window in pairs.windows(2) {
            let (h0, a0) = window[0];
            let (h1, a1) = window[1];
            if hall_code >= h0 && hall_code <= h1 {
                let t = if h1 == h0 {
                    0.0
                } else {
                    (hall_code - h0) / (h1 - h0)
                };
                return Ok(a0 + t * (a1 - a0));
            }
        }
        Ok(pairs[pairs.len() - 1].1)
    }

    /// Mirror angle in degrees for a Hall code.
    pub fn angle_for_hall(&self, hall_code: f64) -> Result<f64> {
        let Some(branch) = self.selected_branch() else {
            return self.interpolated_angle(hall_code);
        };
        let c = &self.quadratic_coeffs[branch * 3..branch * 3 + 3];
        let x = self.normalized(hall_code);
        let normalized_angle = c[0] * x * x + c[1] * x + c[2];
        Ok(normalized_angle * self.mirror_angle_scale + self.mirror_angle_offset)
    }
}

/// Movable-mirror optical system.
#[derive(Clone, Debug)]
pub struct MirrorModel {
    pub real_camera_location: Vec3,
    /// Camera-to-world orientation of the physical module.
    pub real_camera_orientation_cw: Mat3,
    pub rotation_axis: Vec3,
    pub point_on_rotation_axis: Vec3,
    pub mirror_plane_distance: f64,
    pub mirror_normal_zero: Vec3,
    pub flip_img_around_x: bool,
    pub actuator: MirrorActuator,
}

/// Brown/OpenCV polynomial distortion in Light's explicit normalised frame.
#[derive(Clone, Debug)]
pub struct PolynomialDistortion {
    pub center: Vec2,
    pub normalization: Vec2,
    /// `k1, k2, p1, p2, k3`.
    pub coeffs: Vec<f64>,
}

/// Colour calibration of one module for one illuminant.
#[derive(Clone, Debug)]
pub struct ColorProfile {
    pub illuminant: i32,
    pub forward_matrix: Mat3,
    pub rg_ratio: f64,
    pub bg_ratio: f64,
}

/// Flat-field gain mesh: nodes span the sensor inclusively (`columns` across
/// `0..=width`), value 1 at the optical centre and up to ~3.4 in the corners
/// of the wide modules. Stored in calibration-raster orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct VignettingMesh {
    pub columns: usize,
    pub rows: usize,
    pub gains: Vec<f32>,
}

impl VignettingMesh {
    /// Gain at a calibration-raster position of a `width x height` sensor.
    pub fn gain(&self, x: f32, y: f32, width: usize, height: usize) -> f32 {
        if self.columns < 2 || self.rows < 2 {
            return 1.0;
        }
        let fx =
            (x / (width as f32) * (self.columns - 1) as f32).clamp(0.0, (self.columns - 1) as f32);
        let fy = (y / (height as f32) * (self.rows - 1) as f32).clamp(0.0, (self.rows - 1) as f32);
        let c0 = fx.floor() as usize;
        let r0 = fy.floor() as usize;
        let c1 = (c0 + 1).min(self.columns - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f32;
        let ty = fy - r0 as f32;
        let g = |c: usize, r: usize| self.gains[r * self.columns + c];
        let top = g(c0, r0) * (1.0 - tx) + g(c1, r0) * tx;
        let bottom = g(c0, r1) * (1.0 - tx) + g(c1, r1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// Colour-crosstalk mesh: one 4x4 matrix per node acting on
/// `[R, G(red row), G(blue row), B]`, near identity at the centre and mixing a
/// few percent in the corners. Same node layout as [`VignettingMesh`].
#[derive(Clone, Debug, PartialEq)]
pub struct CrosstalkMesh {
    pub columns: usize,
    pub rows: usize,
    /// `columns * rows * 16` row-major coefficients.
    pub matrices: Vec<f32>,
}

impl CrosstalkMesh {
    /// Bilinearly interpolated matrix at a calibration-raster position.
    pub fn matrix(&self, x: f32, y: f32, width: usize, height: usize) -> [f32; 16] {
        let fx =
            (x / (width as f32) * (self.columns - 1) as f32).clamp(0.0, (self.columns - 1) as f32);
        let fy = (y / (height as f32) * (self.rows - 1) as f32).clamp(0.0, (self.rows - 1) as f32);
        let c0 = fx.floor() as usize;
        let r0 = fy.floor() as usize;
        let c1 = (c0 + 1).min(self.columns - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f32;
        let ty = fy - r0 as f32;
        let node = |c: usize, r: usize| &self.matrices[(r * self.columns + c) * 16..][..16];
        let (a, b, c, d) = (node(c0, r0), node(c1, r0), node(c0, r1), node(c1, r1));
        let mut out = [0.0f32; 16];
        for (i, value) in out.iter_mut().enumerate() {
            let top = a[i] * (1.0 - tx) + b[i] * tx;
            let bottom = c[i] * (1.0 - tx) + d[i] * tx;
            *value = top * (1.0 - ty) + bottom * ty;
        }
        out
    }
}

/// Vignetting calibration: one mesh per mirror Hall code (a single entry for
/// fixed modules), the colour-crosstalk mesh, and the module's transmission
/// relative to the A modules.
#[derive(Clone, Debug, Default)]
pub struct Vignetting {
    pub meshes: Vec<(f64, VignettingMesh)>,
    pub crosstalk: Option<CrosstalkMesh>,
    pub relative_brightness: f32,
}

impl Vignetting {
    /// Mesh for the capture's mirror position. Nodes are interpolated in Hall
    /// space; positions outside the measured interval use the nearest mesh.
    pub fn mesh_for_hall(&self, mirror_hall: f64) -> Option<VignettingMesh> {
        let mut meshes = self.meshes.iter().collect::<Vec<_>>();
        meshes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let first = *meshes.first()?;
        let last = *meshes.last()?;
        if mirror_hall <= first.0 {
            return Some(first.1.clone());
        }
        if mirror_hall >= last.0 {
            return Some(last.1.clone());
        }
        let pair = meshes
            .windows(2)
            .find(|pair| pair[0].0 <= mirror_hall && mirror_hall <= pair[1].0)
            .expect("mirror Hall position lies inside the sorted mesh interval");
        let (left, right) = (pair[0], pair[1]);
        if left.1.columns != right.1.columns
            || left.1.rows != right.1.rows
            || left.1.gains.len() != right.1.gains.len()
            || right.0 == left.0
        {
            return Some(
                if (mirror_hall - left.0).abs() <= (right.0 - mirror_hall).abs() {
                    &left.1
                } else {
                    &right.1
                }
                .clone(),
            );
        }
        let t = ((mirror_hall - left.0) / (right.0 - left.0)) as f32;
        Some(VignettingMesh {
            columns: left.1.columns,
            rows: left.1.rows,
            gains: left
                .1
                .gains
                .iter()
                .zip(&right.1.gains)
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct CameraCalibration {
    pub name: String,
    pub mirror_type: Option<MirrorType>,
    pub intrinsics: Vec<IntrinsicsBundle>,
    pub canonical_pose: Option<CanonicalPose>,
    pub mirror: Option<MirrorModel>,
    pub distortion: Option<PolynomialDistortion>,
    pub color: Vec<ColorProfile>,
    pub vignetting: Option<Vignetting>,
}

/// How focus-dependent intrinsics behave outside the calibrated Hall range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IntrinsicsMode {
    /// Freeze at the nearest calibrated bundle outside the measured range.
    Clamp,
    /// Continue the nearest segment linearly in Hall space. Real captures
    /// commonly focus just outside the two factory samples, and the validated
    /// reconstruction model uses this continuation.
    #[default]
    LinearHall,
}

impl CameraCalibration {
    /// Focus-dependent intrinsics for a lens Hall code.
    pub fn k_for_hall(&self, lens_hall: f64, mode: IntrinsicsMode) -> Result<Mat3> {
        if self.intrinsics.is_empty() {
            bail!("no intrinsics for {}", self.name);
        }
        let mut points = self
            .intrinsics
            .iter()
            .filter_map(|bundle| bundle.hall_code.map(|hall| (hall, bundle.k)))
            .collect::<Vec<_>>();
        if points.len() < 2 {
            return Ok(self.intrinsics[0].k);
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (left, right) = if lens_hall <= points[0].0 {
            if mode == IntrinsicsMode::Clamp {
                return Ok(points[0].1);
            }
            (points[0], points[1])
        } else if lens_hall >= points[points.len() - 1].0 {
            if mode == IntrinsicsMode::Clamp {
                return Ok(points[points.len() - 1].1);
            }
            (points[points.len() - 2], points[points.len() - 1])
        } else {
            let mut pair = (points[0], points[1]);
            for window in points.windows(2) {
                if window[0].0 <= lens_hall && lens_hall <= window[1].0 {
                    pair = (window[0], window[1]);
                    break;
                }
            }
            pair
        };
        let span = right.0 - left.0;
        if span == 0.0 {
            bail!("duplicate focus Hall codes for {}", self.name);
        }
        let t = (lens_hall - left.0) / span;
        let mut k = [[0.0; 3]; 3];
        for (r, row) in k.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = (1.0 - t) * left.1[r][c] + t * right.1[r][c];
            }
        }
        Ok(k)
    }

    /// Calibrated object-space focus distance for a capture-time lens Hall
    /// code. This follows the same interpolation/extrapolation policy as the
    /// focus-dependent intrinsic matrix.
    pub fn focus_distance_for_hall(&self, lens_hall: f64, mode: IntrinsicsMode) -> Option<f64> {
        let mut points = self
            .intrinsics
            .iter()
            .filter_map(|bundle| {
                (bundle.focus_distance.is_finite() && bundle.focus_distance > 0.0)
                    .then_some((bundle.hall_code?, bundle.focus_distance))
            })
            .collect::<Vec<_>>();
        if points.is_empty() {
            return None;
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if points.len() == 1 {
            return Some(points[0].1);
        }
        let (left, right) = if lens_hall <= points[0].0 {
            if mode == IntrinsicsMode::Clamp {
                return Some(points[0].1);
            }
            (points[0], points[1])
        } else if lens_hall >= points[points.len() - 1].0 {
            if mode == IntrinsicsMode::Clamp {
                return Some(points[points.len() - 1].1);
            }
            (points[points.len() - 2], points[points.len() - 1])
        } else {
            points
                .windows(2)
                .find(|window| window[0].0 <= lens_hall && lens_hall <= window[1].0)
                .map(|window| (window[0], window[1]))?
        };
        let span = right.0 - left.0;
        if span == 0.0 {
            return None;
        }
        let t = (lens_hall - left.0) / span;
        Some(((1.0 - t) * left.1 + t * right.1).max(1.0))
    }
}

/// All calibrated modules of one device.
#[derive(Clone, Debug, Default)]
pub struct CalibrationDatabase {
    pub cameras: BTreeMap<String, CameraCalibration>,
    /// Sensor type enum value per camera from `hw_info`.
    pub sensor_types: HashMap<String, i32>,
    /// Gain-indexed sensor noise profiles. Earlier headers take priority at a
    /// duplicate gain; device-matched overlays supplement missing points.
    pub sensor_noise_profiles: HashMap<u64, SensorNoiseProfile>,
}

impl CalibrationDatabase {
    /// Resolve from headers in priority order (the capture's own first).
    pub fn from_headers<'a>(headers: impl IntoIterator<Item = &'a LightHeader>) -> Self {
        let mut db = Self::default();
        let mut seen_k = HashMap::<String, Vec<(Option<f64>, Mat3)>>::new();
        for header in headers {
            for sensor in &header.sensor_data {
                let Some(sensor_type) = sensor.type_.map(|value| value.value()) else {
                    continue;
                };
                let Ok(sensor_type) = u64::try_from(sensor_type) else {
                    continue;
                };
                let Some(profile) = sensor
                    .data
                    .as_ref()
                    .and_then(|data| parse_sensor_noise_profile(sensor_type, data))
                else {
                    continue;
                };
                match db.sensor_noise_profiles.entry(sensor_type) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().merge_missing(&profile);
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(profile);
                    }
                }
            }
            if let Some(hw) = header.hw_info.as_ref() {
                for item in &hw.camera {
                    if let (Some(id), Some(sensor)) = (item.id, item.sensor) {
                        db.sensor_types
                            .entry(camera_name(id.value()))
                            .or_insert(sensor.value());
                    }
                }
            }
            for item in &header.module_calibration {
                let Some(id) = item.camera_id else { continue };
                let name = camera_name(id.value());
                let camera = db
                    .cameras
                    .entry(name.clone())
                    .or_insert_with(|| CameraCalibration {
                        name: name.clone(),
                        ..Default::default()
                    });
                for color in &item.color {
                    if let (Some(kind), Some(matrix), Some(rg), Some(bg)) = (
                        color.type_,
                        color.forward_matrix.as_ref(),
                        color.rg_ratio,
                        color.bg_ratio,
                    ) {
                        let profile = ColorProfile {
                            illuminant: kind.value(),
                            forward_matrix: matrix3(matrix),
                            rg_ratio: f64::from(rg),
                            bg_ratio: f64::from(bg),
                        };
                        if !camera.color.iter().any(|existing| {
                            existing.illuminant == profile.illuminant
                                && existing.forward_matrix == profile.forward_matrix
                        }) {
                            camera.color.push(profile);
                        }
                    }
                }
                if let Some(characterization) = item.vignetting.as_ref() {
                    let meshes = characterization
                        .vignetting
                        .iter()
                        .filter_map(|entry| {
                            let model = entry.vignetting.as_ref()?;
                            let (columns, rows) = (model.width? as usize, model.height? as usize);
                            (model.data.len() == columns * rows && columns >= 2 && rows >= 2).then(
                                || {
                                    (
                                        f64::from(entry.hall_code.unwrap_or(0)),
                                        VignettingMesh {
                                            columns,
                                            rows,
                                            gains: model.data.clone(),
                                        },
                                    )
                                },
                            )
                        })
                        .collect::<Vec<_>>();
                    let richer = camera
                        .vignetting
                        .as_ref()
                        .is_none_or(|existing| existing.meshes.len() < meshes.len());
                    if !meshes.is_empty() && richer {
                        let crosstalk = characterization.crosstalk.as_ref().and_then(|model| {
                            let (columns, rows) = (model.width? as usize, model.height? as usize);
                            (model.data_packed.len() == columns * rows * 16
                                && columns >= 2
                                && rows >= 2)
                                .then(|| CrosstalkMesh {
                                    columns,
                                    rows,
                                    matrices: model.data_packed.clone(),
                                })
                        });
                        camera.vignetting = Some(Vignetting {
                            meshes,
                            crosstalk,
                            relative_brightness: characterization
                                .relative_brightness
                                .unwrap_or(1.0),
                        });
                    }
                }
                let Some(geometry) = item.geometry.as_ref() else {
                    continue;
                };
                if camera.mirror_type.is_none() {
                    camera.mirror_type = geometry.mirror_type.and_then(|t| t.enum_value().ok());
                }
                let seen = seen_k.entry(name.clone()).or_default();
                for bundle in &geometry.per_focus_calibration {
                    if let Some(k_mat) = bundle.intrinsics.as_ref().and_then(|i| i.k_mat.as_ref()) {
                        let hall = bundle.focus_hall_code.map(f64::from);
                        let k = matrix3(k_mat);
                        if !seen.iter().any(|(h, m)| *h == hall && *m == k) {
                            seen.push((hall, k));
                            camera.intrinsics.push(IntrinsicsBundle {
                                hall_code: hall,
                                focus_distance: f64::from(bundle.focus_distance.unwrap_or(0.0)),
                                k,
                            });
                        }
                    }
                    if let Some(extrinsics) = bundle.extrinsics.as_ref() {
                        if camera.canonical_pose.is_none()
                            && let Some(canonical) = extrinsics.canonical.as_ref()
                            && let (Some(rotation), Some(translation)) =
                                (canonical.rotation.as_ref(), canonical.translation.as_ref())
                        {
                            camera.canonical_pose = Some(CanonicalPose {
                                rotation_wc: matrix3(rotation),
                                translation_wc: point3(translation),
                            });
                        }
                        if camera.mirror.is_none()
                            && let Some(movable) = extrinsics.moveable_mirror.as_ref()
                            && let (Some(system), Some(mapping)) = (
                                movable.mirror_system.as_ref(),
                                movable.mirror_actuator_mapping.as_ref(),
                            )
                        {
                            camera.mirror = Some(mirror_model(system, mapping));
                        }
                    }
                }
                if camera.distortion.is_none()
                    && let Some(polynomial) = geometry
                        .distortion
                        .as_ref()
                        .and_then(|d| d.polynomial.as_ref())
                    && let (Some(center), Some(normalization)) = (
                        polynomial.distortion_center.as_ref(),
                        polynomial.normalization.as_ref(),
                    )
                {
                    camera.distortion = Some(PolynomialDistortion {
                        center: point2(center),
                        normalization: point2(normalization),
                        coeffs: polynomial.coeffs.iter().map(|&c| f64::from(c)).collect(),
                    });
                }
            }
        }
        db
    }

    /// Resolve from a capture plus optional overlay files (`calibration.lri`,
    /// `zoom_calib_v0.lri`). The capture's headers take priority. Overlays are
    /// accepted only when their physical-device id exactly matches the
    /// capture, preventing accidental cross-camera calibration.
    pub fn from_capture_and_overlays(capture: &LriMessages, overlays: &[LriMessages]) -> Self {
        let device_id = capture.device_id();
        let headers = capture.headers.iter().chain(
            overlays
                .iter()
                .filter(|overlay| device_id.is_some() && overlay.device_id() == device_id)
                .flat_map(|overlay| overlay.headers.iter()),
        );
        Self::from_headers(headers)
    }

    pub fn camera(&self, name: &str) -> Result<&CameraCalibration> {
        self.cameras
            .get(&name.to_ascii_uppercase())
            .with_context(|| format!("no calibration for camera {name}"))
    }
}

fn mirror_model(system: &MirrorSystem, mapping: &MirrorActuatorMapping) -> MirrorModel {
    let f = |v: Option<f32>| f64::from(v.unwrap_or(0.0));
    MirrorModel {
        real_camera_location: system
            .real_camera_location
            .as_ref()
            .map(point3)
            .unwrap_or_default(),
        real_camera_orientation_cw: system
            .real_camera_orientation
            .as_ref()
            .map(matrix3)
            .unwrap_or(crate::math::IDENTITY),
        rotation_axis: system
            .rotation_axis
            .as_ref()
            .map(point3)
            .unwrap_or([0.0, 0.0, 1.0]),
        point_on_rotation_axis: system
            .point_on_rotation_axis
            .as_ref()
            .map(point3)
            .unwrap_or_default(),
        mirror_plane_distance: f(system.distance_mirror_plane_to_point_on_rotation_axis),
        mirror_normal_zero: system
            .mirror_normal_at_zero_degrees
            .as_ref()
            .map(point3)
            .unwrap_or([0.0, 0.0, 1.0]),
        flip_img_around_x: system.flip_img_around_x.unwrap_or(false),
        actuator: MirrorActuator {
            mean_std_normalize: mapping
                .transformation_type
                .and_then(|t| t.enum_value().ok())
                .is_none_or(|t| t == TransformationType::MEAN_STD_NORMALIZE),
            actuator_length_offset: f(mapping.actuator_length_offset),
            actuator_length_scale: f(mapping.actuator_length_scale),
            mirror_angle_offset: f(mapping.mirror_angle_offset),
            mirror_angle_scale: f(mapping.mirror_angle_scale),
            hall_angle_pairs: mapping
                .actuator_angle_pair_vec
                .iter()
                .map(|pair| (f64::from(pair.hall_code.unwrap_or(0)), f(pair.angle)))
                .collect(),
            quadratic_coeffs: mapping
                .quadratic_model
                .as_ref()
                .map(|q| q.model_coeffs.iter().map(|&c| f64::from(c)).collect())
                .unwrap_or_default(),
        },
    }
}

pub fn camera_name(id: i32) -> String {
    CAMERA_NAMES
        .get(id as usize)
        .map(|name| (*name).to_owned())
        .unwrap_or_else(|| format!("camera{id}"))
}

/// Autofocus observations recorded for one captured module.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ModuleFocusState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub achieved: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disparity_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roi: Option<Vec2>,
    pub lens_timeout: bool,
    pub mirror_timeout: bool,
}

/// Capture-time state of one module that the geometry depends on.
#[derive(Clone, Debug)]
pub struct ModuleState {
    pub name: String,
    pub lens_hall: f64,
    pub mirror_hall: f64,
    pub width: usize,
    pub height: usize,
    /// Analogue x digital gain, for photometric normalisation.
    pub gain: f64,
    pub exposure_ns: u64,
    pub focus: ModuleFocusState,
}

/// Read per-module capture state from the capture's headers (first frame of
/// each enabled RAW module).
pub fn module_states(messages: &LriMessages) -> Vec<ModuleState> {
    let mut states = BTreeMap::<String, ModuleState>::new();
    for header in &messages.headers {
        for module in &header.modules {
            let Some(id) = module.id else { continue };
            if !module.is_enabled.unwrap_or(true) || module.frame_index.unwrap_or(0) != 0 {
                continue;
            }
            let Some(surface) = module.sensor_data_surface.as_ref() else {
                continue;
            };
            let Some(size) = surface.size.as_ref() else {
                continue;
            };
            let name = camera_name(id.value());
            let autofocus = module.af_info.as_ref();
            states.entry(name.clone()).or_insert(ModuleState {
                name,
                lens_hall: f64::from(module.lens_position.unwrap_or(0)),
                mirror_hall: f64::from(module.mirror_position.unwrap_or(0)),
                width: size.x.unwrap_or(0).max(0) as usize,
                height: size.y.unwrap_or(0).max(0) as usize,
                gain: f64::from(module.sensor_analog_gain.unwrap_or(1.0))
                    * f64::from(module.sensor_digital_gain.unwrap_or(1.0)),
                exposure_ns: module.sensor_exposure.unwrap_or(0),
                focus: ModuleFocusState {
                    achieved: header.af_info.as_ref().and_then(|info| info.focus_achieved),
                    disparity_distance: autofocus
                        .and_then(|info| info.disparity_focus_distance)
                        .map(f64::from),
                    contrast_distance: autofocus
                        .and_then(|info| info.contrast_focus_distance)
                        .map(f64::from),
                    roi: autofocus
                        .and_then(|info| info.roi_center.as_ref())
                        .map(point2),
                    lens_timeout: autofocus
                        .and_then(|info| info.lens_timeout)
                        .unwrap_or(false),
                    mirror_timeout: autofocus
                        .and_then(|info| info.mirror_timeout)
                        .unwrap_or(false),
                },
            });
        }
    }
    states.into_values().collect()
}

/// 35 mm-equivalent focal length the photographer framed with, if recorded.
pub fn image_focal_length_mm(messages: &LriMessages) -> Option<i32> {
    messages.headers.iter().find_map(|h| h.image_focal_length)
}

/// White-balance gains `(r, g, b)` recorded by the camera, if any.
pub fn awb_gains(messages: &LriMessages) -> Option<[f64; 3]> {
    let prefs = messages.view_preferences.iter().chain(
        messages
            .headers
            .iter()
            .filter_map(|h| h.view_preferences.as_ref()),
    );
    for pref in prefs {
        if let Some(gains) = pref.awb_gains.as_ref()
            && let (Some(r), Some(gr), Some(gb), Some(b)) = (gains.r, gains.g_r, gains.g_b, gains.b)
        {
            return Some([f64::from(r), f64::from(gr + gb) * 0.5, f64::from(b)]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intrinsic(focal: f64) -> Mat3 {
        [
            [focal, 0.0, 2_080.0],
            [0.0, focal, 1_560.0],
            [0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn focus_intrinsics_interpolate_and_optionally_continue() {
        let camera = CameraCalibration {
            name: "B1".to_owned(),
            intrinsics: vec![
                IntrinsicsBundle {
                    hall_code: Some(100.0),
                    focus_distance: 1_000.0,
                    k: intrinsic(1_000.0),
                },
                IntrinsicsBundle {
                    hall_code: Some(200.0),
                    focus_distance: 2_000.0,
                    k: intrinsic(900.0),
                },
            ],
            ..CameraCalibration::default()
        };
        assert_eq!(
            camera.k_for_hall(150.0, IntrinsicsMode::Clamp).unwrap()[0][0],
            950.0
        );
        assert_eq!(
            camera.k_for_hall(50.0, IntrinsicsMode::Clamp).unwrap()[0][0],
            1_000.0
        );
        assert_eq!(
            camera.k_for_hall(50.0, IntrinsicsMode::LinearHall).unwrap()[0][0],
            1_050.0
        );
        assert_eq!(
            camera.focus_distance_for_hall(150.0, IntrinsicsMode::Clamp),
            Some(1_500.0)
        );
        assert_eq!(
            camera.focus_distance_for_hall(50.0, IntrinsicsMode::Clamp),
            Some(1_000.0)
        );
        assert_eq!(
            camera.focus_distance_for_hall(50.0, IntrinsicsMode::LinearHall),
            Some(500.0)
        );
    }

    #[test]
    fn vignetting_interpolates_mesh_nodes_in_mirror_hall_space() {
        let mesh = |gain| VignettingMesh {
            columns: 2,
            rows: 2,
            gains: vec![gain; 4],
        };
        let calibration = Vignetting {
            meshes: vec![(100.0, mesh(1.0)), (200.0, mesh(3.0))],
            ..Vignetting::default()
        };
        assert_eq!(calibration.mesh_for_hall(50.0).unwrap().gains, vec![1.0; 4]);
        assert_eq!(
            calibration.mesh_for_hall(150.0).unwrap().gains,
            vec![2.0; 4]
        );
        assert_eq!(
            calibration.mesh_for_hall(250.0).unwrap().gains,
            vec![3.0; 4]
        );
    }
}
