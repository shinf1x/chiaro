//! Forward camera model for one captured module: pixel -> world ray and
//! world point -> pixel, including focus-dependent intrinsics, polynomial
//! distortion, canonical extrinsics, and movable/glued mirrors modelled as a
//! reflected virtual camera.
//!
//! Pixel coordinates are *calibration raster* coordinates: the decoded RAW
//! stream rotated by 180 degrees (`x_cal = width - 1 - x_stream`). `x` grows
//! right, `y` grows down, camera `z` points forward. These semantics were
//! validated against real captures in the companion research (see the crate
//! README); the Rust port is checked numerically against that reference
//! implementation in `tests/geometry_fixture.rs`.

use anyhow::{Context, Result, bail};

use crate::calibration::{CameraCalibration, IntrinsicsMode, ModuleState, PolynomialDistortion};
use crate::math::{
    self, IDENTITY, Mat3, Vec2, Vec3, add, mul, mul_vec, normalize, reflection, scale, sub,
    transpose,
};

/// Small per-capture corrections applied on top of the factory model.
#[derive(Clone, Debug, Default)]
pub struct CameraRefinement {
    /// Additive mirror angle, degrees (movable modules).
    pub mirror_angle_offset_degrees: f64,
    /// World-space axis-angle rotation of the bearing frame, degrees.
    pub orientation_offset_degrees: Option<Vec3>,
    /// Offset of the resolved optical centre in world calibration units. For
    /// mirrored modules this intentionally corrects the effective virtual
    /// viewpoint rather than pretending one capture can identify every
    /// physical mirror-system component independently.
    pub center_offset_world: Option<Vec3>,
    /// Translation of the calibration raster relative to the captured sensor
    /// raster, in pixels. Both K's principal point and the distortion centre
    /// move together, which models a crop/active-area origin error without
    /// changing the calibrated distortion coefficients.
    pub sensor_offset_px: Option<Vec2>,
    /// Small capture-specific common multiplicative correction to focal
    /// length, expressed as a fractional delta from factory (0.001 = +0.1%).
    pub focal_scale_delta: Option<f64>,
    /// Additional capture-specific focal anisotropy. Positive values increase
    /// fx and decrease fy by the same fractional amount around the common
    /// focal scale. This adds one aspect-ratio DOF without duplicating the
    /// isotropic scale parameter.
    pub focal_aspect_delta: Option<f64>,
    /// Additive Brown/OpenCV distortion corrections `[dk1, dk2, dp1, dp2]` in
    /// the calibration's normalized distortion frame. They are intentionally
    /// capture-local nuisance terms and never overwrite factory calibration.
    pub distortion_delta: Option<[f64; 4]>,
    /// Additional displacement of the Brown/OpenCV distortion centre relative
    /// to the principal-point/raster shift, in native sensor pixels. Keeping
    /// this separate from `sensor_offset_px` lets a capture correct a genuine
    /// lens/distortion-centre error without pretending that the active raster
    /// origin moved by the same amount.
    pub distortion_center_offset_px: Option<Vec2>,
}

#[derive(Clone, Debug)]
pub struct ResolvedCameraTemplate {
    name: String,
    width: usize,
    height: usize,
    base_k: Mat3,
    distortion: Option<PolynomialDistortion>,
    flip_around_x: Option<bool>,
    pose: PoseTemplate,
    focus_distance: Option<f64>,
    angle_optical_center_reference: Option<&'static str>,
    angle_optical_center_reference_pixel: Option<Vec2>,
}

#[derive(Clone, Debug)]
enum PoseTemplate {
    Canonical {
        rotation_wc: Mat3,
        rotation_cw: Mat3,
        translation_wc: Vec3,
        center: Vec3,
    },
    Mirror {
        real_cw: Mat3,
        rotation_axis: Vec3,
        factory_angle_degrees: f64,
        mirror_normal_zero: Vec3,
        point_on_rotation_axis: Vec3,
        mirror_plane_distance: f64,
        real_camera_location: Vec3,
    },
}

impl ResolvedCameraTemplate {
    pub fn new(
        calibration: &CameraCalibration,
        state: &ModuleState,
        mode: IntrinsicsMode,
    ) -> Result<Self> {
        let base_k = calibration.k_for_hall(state.lens_hall, mode)?;
        let mut factory_mirror_angle = None;
        let pose = if let Some(canonical) = calibration.canonical_pose.as_ref() {
            let rotation_cw = transpose(&canonical.rotation_wc);
            PoseTemplate::Canonical {
                rotation_wc: canonical.rotation_wc,
                rotation_cw,
                translation_wc: canonical.translation_wc,
                center: canonical.center_world(),
            }
        } else if let Some(mirror) = calibration.mirror.as_ref() {
            let angle = mirror.actuator.angle_for_hall(state.mirror_hall)?;
            factory_mirror_angle = Some(angle);
            PoseTemplate::Mirror {
                real_cw: mirror.real_camera_orientation_cw,
                rotation_axis: mirror.rotation_axis,
                factory_angle_degrees: angle,
                mirror_normal_zero: mirror.mirror_normal_zero,
                point_on_rotation_axis: mirror.point_on_rotation_axis,
                mirror_plane_distance: mirror.mirror_plane_distance,
                real_camera_location: mirror.real_camera_location,
            }
        } else {
            bail!(
                "{} has neither a canonical pose nor a mirror model",
                calibration.name
            );
        };
        Ok(Self {
            name: calibration.name.clone(),
            width: state.width,
            height: state.height,
            base_k,
            distortion: calibration.distortion.clone(),
            flip_around_x: calibration.mirror.as_ref().map(|m| m.flip_img_around_x),
            pose,
            focus_distance: calibration.focus_distance_for_hall(state.lens_hall, mode),
            angle_optical_center_reference: optical_center_reference(&calibration.name),
            angle_optical_center_reference_pixel: factory_mirror_angle.and_then(|angle| {
                calibration
                    .angle_optical_center_mapping
                    .as_ref()?
                    .center_for_angle(angle)
            }),
        })
    }

    pub fn resolve(&self, refinement: &CameraRefinement) -> Result<ResolvedCamera> {
        let sensor_offset = refinement.sensor_offset_px.unwrap_or([0.0; 2]);
        let focal_scale = 1.0 + refinement.focal_scale_delta.unwrap_or(0.0);
        let focal_aspect = refinement.focal_aspect_delta.unwrap_or(0.0);
        let focal_x_scale = focal_scale * (1.0 + focal_aspect);
        let focal_y_scale = focal_scale * (1.0 - focal_aspect);
        if !focal_x_scale.is_finite()
            || !focal_y_scale.is_finite()
            || focal_x_scale <= 0.0
            || focal_y_scale <= 0.0
        {
            bail!(
                "{} has invalid focal refinement scales ({focal_x_scale}, {focal_y_scale})",
                self.name
            );
        }
        let mut k = self.base_k;
        k[0][0] *= focal_x_scale;
        k[1][1] *= focal_y_scale;
        k[0][2] += sensor_offset[0];
        k[1][2] += sensor_offset[1];
        let k_inverse = math::inverse(&k).context("singular intrinsic matrix")?;
        let center_offset = refinement.center_offset_world.unwrap_or([0.0; 3]);
        let pose = match &self.pose {
            PoseTemplate::Canonical {
                rotation_wc,
                rotation_cw,
                translation_wc,
                center,
            } => Pose::Canonical {
                rotation_wc: *rotation_wc,
                rotation_cw: *rotation_cw,
                translation_wc: sub(*translation_wc, mul_vec(rotation_wc, center_offset)),
                center: add(*center, center_offset),
            },
            PoseTemplate::Mirror {
                real_cw,
                rotation_axis,
                factory_angle_degrees,
                mirror_normal_zero,
                point_on_rotation_axis,
                mirror_plane_distance,
                real_camera_location,
            } => {
                let angle = factory_angle_degrees + refinement.mirror_angle_offset_degrees;
                let rotation = math::rotation_about_axis(*rotation_axis, angle.to_radians());
                let normal = normalize(mul_vec(&rotation, *mirror_normal_zero));
                let plane_point = add(
                    *point_on_rotation_axis,
                    scale(normal, *mirror_plane_distance),
                );
                let reflect = reflection(normal);
                let distance = math::dot(normal, sub(*real_camera_location, plane_point));
                let virtual_center = add(
                    sub(*real_camera_location, scale(normal, 2.0 * distance)),
                    center_offset,
                );
                Pose::Mirror {
                    real_cw: *real_cw,
                    reflect,
                    virtual_center,
                }
            }
        };
        let orientation_correction = refinement
            .orientation_offset_degrees
            .map(|v| math::rotation_from_axis_angle(scale(v, std::f64::consts::PI / 180.0)))
            .unwrap_or(IDENTITY);
        let mut distortion = self.distortion.clone();
        if let Some(distortion) = distortion.as_mut() {
            distortion.center[0] += sensor_offset[0];
            distortion.center[1] += sensor_offset[1];
            if let Some(offset) = refinement.distortion_center_offset_px {
                distortion.center[0] += offset[0];
                distortion.center[1] += offset[1];
            }
            if let Some(delta) = refinement.distortion_delta {
                if distortion.coeffs.len() < 4 {
                    distortion.coeffs.resize(4, 0.0);
                }
                distortion.coeffs[0] += delta[0];
                distortion.coeffs[1] += delta[1];
                distortion.coeffs[2] += delta[2];
                distortion.coeffs[3] += delta[3];
            }
        }
        Ok(ResolvedCamera {
            name: self.name.clone(),
            width: self.width,
            height: self.height,
            k,
            k_inverse,
            distortion,
            flip_around_x: self.flip_around_x,
            pose,
            orientation_correction,
            focal_px: k[0][0],
            focus_distance: self.focus_distance,
            angle_optical_center_reference: self.angle_optical_center_reference,
            angle_optical_center_reference_pixel: self.angle_optical_center_reference_pixel,
        })
    }
}

/// A module with its calibration resolved for one capture.
#[derive(Clone, Debug)]
pub struct ResolvedCamera {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub k: Mat3,
    k_inverse: Mat3,
    distortion: Option<PolynomialDistortion>,
    flip_around_x: Option<bool>,
    pose: Pose,
    orientation_correction: Mat3,
    /// Calibrated focal length in pixels, for resolution weighting.
    pub focal_px: f64,
    /// Object-space focus distance interpolated from the capture-time lens
    /// Hall code, in the calibration's distance units (millimetres on L16).
    pub focus_distance: Option<f64>,
    angle_optical_center_reference: Option<&'static str>,
    angle_optical_center_reference_pixel: Option<Vec2>,
}

#[derive(Clone, Debug)]
enum Pose {
    Canonical {
        rotation_wc: Mat3,
        rotation_cw: Mat3,
        translation_wc: Vec3,
        center: Vec3,
    },
    Mirror {
        /// Camera-to-world orientation of the physical module.
        real_cw: Mat3,
        reflect: Mat3,
        virtual_center: Vec3,
    },
}

/// A world ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl ResolvedCamera {
    pub fn new(
        calibration: &CameraCalibration,
        state: &ModuleState,
        mode: IntrinsicsMode,
        refinement: &CameraRefinement,
    ) -> Result<Self> {
        ResolvedCameraTemplate::new(calibration, state, mode)?.resolve(refinement)
    }

    /// Optical centre in world (calibration) coordinates.
    pub fn center(&self) -> Vec3 {
        match &self.pose {
            Pose::Canonical { center, .. } => *center,
            Pose::Mirror { virtual_center, .. } => *virtual_center,
        }
    }

    /// Capture-resolved factory optical-axis point when `reference_name` is
    /// the mapping's adjacent wider L16 reference camera.
    pub fn angle_optical_center_prior(&self, reference_name: &str) -> Option<Vec2> {
        self.angle_optical_center_reference
            .is_some_and(|name| name.eq_ignore_ascii_case(reference_name))
            .then_some(self.angle_optical_center_reference_pixel?)
    }

    /// Observed target-raster pixel of the camera-frame optical axis. This is
    /// normally very close to K's principal point but includes the calibrated
    /// Brown distortion convention used by the rest of the forward model.
    pub fn optical_axis_pixel(&self) -> Vec2 {
        let ideal = [self.k[0][2], self.k[1][2]];
        self.distortion
            .as_ref()
            .map_or(ideal, |distortion| distort(distortion, ideal))
    }

    /// Restore right-handedness of a mirrored image by reflecting one axis.
    fn reflect_image_axis(&self, normalized: Vec2) -> Vec2 {
        match self.flip_around_x {
            Some(true) => [normalized[0], -normalized[1]],
            Some(false) => [-normalized[0], normalized[1]],
            None => normalized,
        }
    }

    /// Distortion-corrected, right-handed camera-frame bearing for a raster
    /// pixel. Unlike [`Self::pixel_to_ray`], this deliberately excludes the
    /// factory extrinsic/mirror pose. It is therefore suitable for estimating
    /// capture-specific epipolar geometry from image correspondences without
    /// baking the pose we are trying to refine into the measurements.
    pub fn pixel_to_camera_direction(&self, pixel: Vec2) -> Vec3 {
        let ideal = match &self.distortion {
            Some(distortion) => undistort(distortion, pixel),
            None => pixel,
        };
        let mut direction = normalize(mul_vec(&self.k_inverse, [ideal[0], ideal[1], 1.0]));
        if self.flip_around_x.is_some() {
            let xy = self.reflect_image_axis([direction[0], direction[1]]);
            direction = normalize([xy[0], xy[1], direction[2]]);
        }
        direction
    }

    /// Distortion-corrected normalized image coordinate `(x/z, y/z)` in the
    /// right-handed camera frame.
    pub fn pixel_to_normalized_camera(&self, pixel: Vec2) -> Option<Vec2> {
        let direction = self.pixel_to_camera_direction(pixel);
        if !direction[2].is_finite() || direction[2].abs() <= 1.0e-12 {
            None
        } else {
            Some([direction[0] / direction[2], direction[1] / direction[2]])
        }
    }

    /// Geometric-mean focal scale of the resolved intrinsic matrix, in sensor
    /// pixels. This is used only to express normalized epipolar residuals in
    /// an intuitive pixel-equivalent unit.
    pub fn focal_scale_px(&self) -> f64 {
        (self.k[0][0].abs() * self.k[1][1].abs()).sqrt().max(1.0)
    }

    /// Pixel (calibration raster) to world ray.
    pub fn pixel_to_ray(&self, pixel: Vec2) -> Ray {
        let direction = self.pixel_to_camera_direction(pixel);
        let (origin, world) = match &self.pose {
            Pose::Canonical {
                rotation_cw,
                center,
                ..
            } => (*center, mul_vec(rotation_cw, direction)),
            Pose::Mirror {
                real_cw,
                reflect,
                virtual_center,
            } => (
                *virtual_center,
                mul_vec(reflect, mul_vec(real_cw, direction)),
            ),
        };
        Ray {
            origin,
            direction: normalize(mul_vec(&self.orientation_correction, normalize(world))),
        }
    }

    /// World point to pixel. Returns `None` when the point is behind the
    /// camera or so far outside the field that the distortion polynomial is
    /// meaningless (beyond half a frame outside the sensor, where the
    /// polynomial folds back and could land inside the image).
    pub fn project(&self, point: Vec3) -> Option<Vec2> {
        self.project_impl(point, true)
    }

    /// [`Self::project`] without the field-of-view guard: the raw model
    /// evaluated anywhere in front of the camera.
    pub fn project_unbounded(&self, point: Vec3) -> Option<Vec2> {
        self.project_impl(point, false)
    }

    fn project_impl(&self, point: Vec3, bounded: bool) -> Option<Vec2> {
        let mut point = point;
        if self.orientation_correction != IDENTITY {
            let center = self.center();
            point = add(
                center,
                mul_vec(&transpose(&self.orientation_correction), sub(point, center)),
            );
        }
        let camera = match &self.pose {
            Pose::Canonical {
                rotation_wc,
                translation_wc,
                ..
            } => add(mul_vec(rotation_wc, point), *translation_wc),
            Pose::Mirror {
                real_cw,
                reflect,
                virtual_center,
            } => mul_vec(
                &transpose(real_cw),
                mul_vec(reflect, sub(point, *virtual_center)),
            ),
        };
        if camera[2] <= 0.0 {
            return None;
        }
        let normalized = self.reflect_image_axis([camera[0] / camera[2], camera[1] / camera[2]]);
        let k = &self.k;
        let ideal = [
            k[0][0] * normalized[0] + k[0][1] * normalized[1] + k[0][2],
            k[1][1] * normalized[1] + k[1][2],
        ];
        // The distortion polynomial is only meaningful near the calibrated
        // field; far outside it folds back and can land inside the sensor.
        // Reject ideal positions beyond half a frame outside the image.
        let margin_x = self.width as f64 * 0.5;
        let margin_y = self.height as f64 * 0.5;
        if bounded
            && (ideal[0] < -margin_x
                || ideal[1] < -margin_y
                || ideal[0] > self.width as f64 + margin_x
                || ideal[1] > self.height as f64 + margin_y)
        {
            return None;
        }
        Some(match &self.distortion {
            Some(distortion) => distort(distortion, ideal),
            None => ideal,
        })
    }

    /// Where a pixel of `reference` lands in this camera for a scene point at
    /// distance `depth` along the reference ray (calibration distance units,
    /// believed to be millimetres). Use a very large depth for the infinity
    /// (pure-rotation) mapping.
    pub fn map_from(&self, reference: &ResolvedCamera, pixel: Vec2, depth: f64) -> Option<Vec2> {
        let ray = reference.pixel_to_ray(pixel);
        self.project(add(ray.origin, scale(ray.direction, depth)))
    }

    /// `true` when a pixel lies inside the sensor.
    pub fn contains(&self, pixel: Vec2) -> bool {
        pixel[0] >= 0.0
            && pixel[1] >= 0.0
            && pixel[0] <= (self.width - 1) as f64
            && pixel[1] <= (self.height - 1) as f64
    }

    /// Rotation-only homography mapping `reference` pixels to this camera's
    /// ideal (undistorted) pixels for points at infinity: `K_t R K_r^-1`.
    /// Useful as a compact summary; `map_from` is exact (includes distortion).
    pub fn infinity_homography_from(&self, reference: &ResolvedCamera) -> Mat3 {
        // Build from three ray correspondences through the exact model.
        let rotation_ref_to_this = {
            let basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let mut columns = [[0.0; 3]; 3];
            for (i, axis) in basis.iter().enumerate() {
                // world direction from reference camera axis, then into this camera frame
                let world = reference.camera_to_world_direction(*axis);
                let local = self.world_to_camera_direction(world);
                for r in 0..3 {
                    columns[r][i] = local[r];
                }
            }
            columns
        };
        mul(&mul(&self.k, &rotation_ref_to_this), &reference.k_inverse)
    }

    fn camera_to_world_direction(&self, direction: Vec3) -> Vec3 {
        let direction = if self.flip_around_x.is_some() {
            let xy = self.reflect_image_axis([direction[0], direction[1]]);
            [xy[0], xy[1], direction[2]]
        } else {
            direction
        };
        let world = match &self.pose {
            Pose::Canonical { rotation_cw, .. } => mul_vec(rotation_cw, direction),
            Pose::Mirror {
                real_cw, reflect, ..
            } => mul_vec(reflect, mul_vec(real_cw, direction)),
        };
        mul_vec(&self.orientation_correction, world)
    }

    fn world_to_camera_direction(&self, world: Vec3) -> Vec3 {
        let world = mul_vec(&transpose(&self.orientation_correction), world);
        let camera = match &self.pose {
            Pose::Canonical { rotation_wc, .. } => mul_vec(rotation_wc, world),
            Pose::Mirror {
                real_cw, reflect, ..
            } => mul_vec(&transpose(real_cw), mul_vec(reflect, world)),
        };
        if self.flip_around_x.is_some() {
            let xy = self.reflect_image_axis([camera[0], camera[1]]);
            [xy[0], xy[1], camera[2]]
        } else {
            camera
        }
    }
}

fn optical_center_reference(camera: &str) -> Option<&'static str> {
    match camera {
        "B1" | "B2" | "B3" | "B5" => Some("A1"),
        "C1" | "C2" | "C3" | "C4" => Some("B4"),
        _ => None,
    }
}

fn to_normalized(distortion: &PolynomialDistortion, pixel: Vec2) -> Vec2 {
    [
        (pixel[0] - distortion.center[0]) / distortion.normalization[0],
        (pixel[1] - distortion.center[1]) / distortion.normalization[1],
    ]
}

fn from_normalized(distortion: &PolynomialDistortion, normalized: Vec2) -> Vec2 {
    [
        normalized[0] * distortion.normalization[0] + distortion.center[0],
        normalized[1] * distortion.normalization[1] + distortion.center[1],
    ]
}

/// Brown model `k1, k2, p1, p2, k3` applied to normalised coordinates.
fn distort_normalized(coeffs: &[f64], xy: Vec2) -> Vec2 {
    let c = |i: usize| coeffs.get(i).copied().unwrap_or(0.0);
    let (k1, k2, p1, p2, k3) = (c(0), c(1), c(2), c(3), c(4));
    let [x, y] = xy;
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    [
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    ]
}

pub(crate) fn distort(distortion: &PolynomialDistortion, ideal: Vec2) -> Vec2 {
    from_normalized(
        distortion,
        distort_normalized(&distortion.coeffs, to_normalized(distortion, ideal)),
    )
}

/// Inverse of [`distort`] by fixed-point iteration, as OpenCV's
/// `undistortPoints` does.
pub(crate) fn undistort(distortion: &PolynomialDistortion, pixel: Vec2) -> Vec2 {
    let observed = to_normalized(distortion, pixel);
    let mut estimate = observed;
    for _ in 0..20 {
        let distorted = distort_normalized(&distortion.coeffs, estimate);
        let next = [
            estimate[0] + (observed[0] - distorted[0]),
            estimate[1] + (observed[1] - distorted[1]),
        ];
        let delta = (next[0] - estimate[0])
            .abs()
            .max((next[1] - estimate[1]).abs());
        estimate = next;
        if delta < 1e-12 {
            break;
        }
    }
    from_normalized(distortion, estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CanonicalPose, IntrinsicsBundle};
    use crate::math::norm;

    fn refinement_test_camera(distortion: Option<PolynomialDistortion>) -> CameraCalibration {
        CameraCalibration {
            name: "B1".to_owned(),
            intrinsics: vec![IntrinsicsBundle {
                hall_code: Some(0.0),
                focus_distance: 1_000.0,
                k: [[800.0, 0.0, 500.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]],
            }],
            canonical_pose: Some(CanonicalPose {
                rotation_wc: IDENTITY,
                translation_wc: [-10.0, -20.0, -30.0],
            }),
            distortion,
            ..Default::default()
        }
    }

    fn refinement_test_state() -> ModuleState {
        ModuleState {
            name: "B1".to_owned(),
            lens_hall: 0.0,
            mirror_hall: 0.0,
            width: 1_000,
            height: 800,
            gain: 1.0,
            exposure_ns: 1,
            focus: Default::default(),
        }
    }

    #[test]
    fn distortion_round_trips() {
        let distortion = PolynomialDistortion {
            center: [2080.0, 1560.0],
            normalization: [3380.0, 3380.0],
            coeffs: vec![0.1138, -0.3495, 0.0, 0.0, 0.0934],
        };
        for pixel in [
            [100.0, 100.0],
            [2080.0, 1560.0],
            [4100.0, 3000.0],
            [0.0, 3119.0],
        ] {
            let ideal = undistort(&distortion, pixel);
            let back = distort(&distortion, ideal);
            assert!((back[0] - pixel[0]).abs() < 1e-6 && (back[1] - pixel[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn focal_scale_changes_only_radial_image_magnification() {
        let calibration = refinement_test_camera(None);
        let state = refinement_test_state();
        let factory = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement::default(),
        )
        .unwrap();
        let refined = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement {
                focal_scale_delta: Some(0.01),
                ..Default::default()
            },
        )
        .unwrap();
        let point = add(factory.center(), [200.0, -100.0, 2_000.0]);
        let factory_pixel = factory.project(point).unwrap();
        let refined_pixel = refined.project(point).unwrap();
        let principal = [500.0, 400.0];
        for axis in 0..2 {
            let factory_offset = factory_pixel[axis] - principal[axis];
            let refined_offset = refined_pixel[axis] - principal[axis];
            assert!((refined_offset - 1.01 * factory_offset).abs() < 1.0e-9);
        }
        // The inverse ray must remain exactly consistent with the refined
        // projection rather than changing the camera's physical centre.
        let ray = refined.pixel_to_ray(refined_pixel);
        assert!(norm(sub(ray.origin, factory.center())) < 1.0e-12);
    }

    #[test]
    fn focal_aspect_changes_x_and_y_in_opposite_directions() {
        let calibration = refinement_test_camera(None);
        let state = refinement_test_state();
        let factory = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement::default(),
        )
        .unwrap();
        let refined = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement {
                focal_aspect_delta: Some(0.01),
                ..Default::default()
            },
        )
        .unwrap();
        let point = add(factory.center(), [200.0, -100.0, 2_000.0]);
        let factory_pixel = factory.project(point).unwrap();
        let refined_pixel = refined.project(point).unwrap();
        let principal = [500.0, 400.0];
        assert!(
            ((refined_pixel[0] - principal[0]) - 1.01 * (factory_pixel[0] - principal[0])).abs()
                < 1.0e-9
        );
        assert!(
            ((refined_pixel[1] - principal[1]) - 0.99 * (factory_pixel[1] - principal[1])).abs()
                < 1.0e-9
        );
    }

    #[test]
    fn distortion_delta_is_applied_and_inverse_projection_remains_consistent() {
        let distortion = PolynomialDistortion {
            center: [500.0, 400.0],
            normalization: [800.0, 800.0],
            coeffs: vec![0.08, -0.02, 0.001, -0.002, 0.0],
        };
        let calibration = refinement_test_camera(Some(distortion));
        let state = refinement_test_state();
        let factory = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement::default(),
        )
        .unwrap();
        let refined = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement {
                distortion_delta: Some([0.01, -0.015, 0.0005, -0.0007]),
                ..Default::default()
            },
        )
        .unwrap();
        let point = add(factory.center(), [500.0, -300.0, 1_500.0]);
        let factory_pixel = factory.project(point).unwrap();
        let refined_pixel = refined.project(point).unwrap();
        assert!(
            (factory_pixel[0] - refined_pixel[0]).abs()
                + (factory_pixel[1] - refined_pixel[1]).abs()
                > 0.05
        );
        let ray = refined.pixel_to_ray(refined_pixel);
        let expected = normalize(sub(point, refined.center()));
        assert!(norm(sub(ray.direction, expected)) < 1.0e-6);
    }

    #[test]
    fn center_and_sensor_offsets_modify_the_resolved_camera_in_their_own_frames() {
        let distortion = PolynomialDistortion {
            center: [500.0, 400.0],
            normalization: [800.0, 800.0],
            coeffs: vec![0.08, -0.02, 0.001, -0.002, 0.0],
        };
        let calibration = refinement_test_camera(Some(distortion));
        let state = refinement_test_state();
        let factory = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement::default(),
        )
        .unwrap();
        let refinement = CameraRefinement {
            center_offset_world: Some([1.5, -2.0, 0.75]),
            sensor_offset_px: Some([13.0, -7.0]),
            ..Default::default()
        };
        let refined =
            ResolvedCamera::new(&calibration, &state, IntrinsicsMode::Clamp, &refinement).unwrap();
        assert_eq!(refined.center(), [11.5, 18.0, 30.75]);

        let point = add(factory.center(), [120.0, -80.0, 2_000.0]);
        let factory_pixel = factory.project(point).unwrap();
        let sensor_only = ResolvedCamera::new(
            &calibration,
            &state,
            IntrinsicsMode::Clamp,
            &CameraRefinement {
                sensor_offset_px: Some([13.0, -7.0]),
                ..Default::default()
            },
        )
        .unwrap();
        let shifted_pixel = sensor_only.project(point).unwrap();
        assert!((shifted_pixel[0] - factory_pixel[0] - 13.0).abs() < 1.0e-9);
        assert!((shifted_pixel[1] - factory_pixel[1] + 7.0).abs() < 1.0e-9);
        let shifted_ray = sensor_only.pixel_to_ray(shifted_pixel);
        let factory_ray = factory.pixel_to_ray(factory_pixel);
        assert!(norm(sub(shifted_ray.direction, factory_ray.direction)) < 1.0e-10);
    }


}
