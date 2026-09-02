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
        let k = calibration.k_for_hall(state.lens_hall, mode)?;
        let k_inverse = math::inverse(&k).context("singular intrinsic matrix")?;
        let pose = if let Some(canonical) = calibration.canonical_pose.as_ref() {
            Pose::Canonical {
                rotation_wc: canonical.rotation_wc,
                rotation_cw: transpose(&canonical.rotation_wc),
                translation_wc: canonical.translation_wc,
                center: canonical.center_world(),
            }
        } else if let Some(mirror) = calibration.mirror.as_ref() {
            let angle = mirror.actuator.angle_for_hall(state.mirror_hall)?
                + refinement.mirror_angle_offset_degrees;
            let rotation = math::rotation_about_axis(mirror.rotation_axis, angle.to_radians());
            let normal = normalize(mul_vec(&rotation, mirror.mirror_normal_zero));
            let plane_point = add(
                mirror.point_on_rotation_axis,
                scale(normal, mirror.mirror_plane_distance),
            );
            let reflect = reflection(normal);
            let distance = math::dot(normal, sub(mirror.real_camera_location, plane_point));
            let virtual_center = sub(mirror.real_camera_location, scale(normal, 2.0 * distance));
            Pose::Mirror {
                real_cw: mirror.real_camera_orientation_cw,
                reflect,
                virtual_center,
            }
        } else {
            bail!(
                "{} has neither a canonical pose nor a mirror model",
                calibration.name
            );
        };
        let orientation_correction = refinement
            .orientation_offset_degrees
            .map(|v| math::rotation_from_axis_angle(scale(v, std::f64::consts::PI / 180.0)))
            .unwrap_or(IDENTITY);
        Ok(Self {
            name: calibration.name.clone(),
            width: state.width,
            height: state.height,
            k,
            k_inverse,
            distortion: calibration.distortion.clone(),
            flip_around_x: calibration.mirror.as_ref().map(|m| m.flip_img_around_x),
            pose,
            orientation_correction,
            focal_px: k[0][0],
            focus_distance: calibration.focus_distance_for_hall(state.lens_hall, mode),
        })
    }

    /// Optical centre in world (calibration) coordinates.
    pub fn center(&self) -> Vec3 {
        match &self.pose {
            Pose::Canonical { center, .. } => *center,
            Pose::Mirror { virtual_center, .. } => *virtual_center,
        }
    }

    /// Restore right-handedness of a mirrored image by reflecting one axis.
    fn reflect_image_axis(&self, normalized: Vec2) -> Vec2 {
        match self.flip_around_x {
            Some(true) => [normalized[0], -normalized[1]],
            Some(false) => [-normalized[0], normalized[1]],
            None => normalized,
        }
    }

    /// Pixel (calibration raster) to world ray.
    pub fn pixel_to_ray(&self, pixel: Vec2) -> Ray {
        let ideal = match &self.distortion {
            Some(distortion) => undistort(distortion, pixel),
            None => pixel,
        };
        let mut direction = normalize(mul_vec(&self.k_inverse, [ideal[0], ideal[1], 1.0]));
        if self.flip_around_x.is_some() {
            let xy = self.reflect_image_axis([direction[0], direction[1]]);
            direction = normalize([xy[0], xy[1], direction[2]]);
        }
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
}
