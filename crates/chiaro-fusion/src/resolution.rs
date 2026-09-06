//! Bounded-memory multi-camera resolution reconstruction primitives.
//!
//! Unlike pull resampling, the reconstruction path retains the location of
//! physical sensor samples in the output lattice. Nearby samples are projected
//! through the local inverse warp Jacobian and accumulated with a compact Hann
//! window. Multiple cameras therefore contribute their distinct subpixel
//! phases before normalization.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{
    align::Warp,
    image::{Plane, match_patch},
};

const LOCAL_WARP_STEP: usize = 32;
const LOCAL_MATCH_PATCH: usize = 12;
const LOCAL_MATCH_RADIUS: usize = 2;
const MAGNIFIED_MATCH_RADIUS: usize = 5;
const LOCAL_MINIMUM_SCORE: f32 = 0.72;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResolutionReconstruction {
    /// Existing per-camera bilinear pull resampling and robust blending.
    Resample,
    /// Project physical samples from multiple cameras onto the output lattice.
    MultiCamera,
    /// Joint solve from projected physical Bayer measurements, with production
    /// fallback wherever the local solve lacks independent support.
    #[default]
    JointCfa,
}

impl ResolutionReconstruction {
    pub const ALL: [Self; 3] = [Self::JointCfa, Self::MultiCamera, Self::Resample];

    pub fn uses_resolution_warps(self) -> bool {
        matches!(self, Self::MultiCamera | Self::JointCfa)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Resample => "Resample",
            Self::MultiCamera => "Multi-camera",
            Self::JointCfa => "Joint CFA",
        }
    }
}

impl fmt::Display for ResolutionReconstruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Resample => "resample",
            Self::MultiCamera => "multi-camera",
            Self::JointCfa => "joint-cfa",
        })
    }
}

impl FromStr for ResolutionReconstruction {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "resample" => Ok(Self::Resample),
            "multi-camera" | "multicamera" => Ok(Self::MultiCamera),
            "joint-cfa" | "jointcfa" => Ok(Self::JointCfa),
            _ => Err(format!(
                "unknown resolution reconstruction {value:?}; expected resample, multi-camera, or joint-cfa"
            )),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ResolutionReconstructionReport {
    pub mode: ResolutionReconstruction,
    /// Fraction of covered pixels with at least two locally registered cameras.
    pub candidate_fraction: f32,
    /// Fraction retaining useful subpixel phase diversity.
    pub phase_supported_fraction: f32,
    /// Fraction of covered output pixels receiving a multi-camera estimate.
    pub reconstructed_fraction: f32,
    /// Mean number of independently projected cameras at reconstructed pixels.
    pub mean_cameras: f32,
    /// Mean RMS displacement of camera sampling phases, in output pixels.
    pub mean_phase_spread: f32,
    /// Mean blend confidence applied to the reconstructed luminance.
    pub mean_confidence: f32,
    /// Compact support radius used by the normalized overlap-add kernel.
    pub hann_radius_px: f32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ResolutionAlignmentReport {
    /// Fraction of local control nodes supported by texture and a spatially
    /// coherent correlation peak.
    pub verified_fraction: f32,
    /// Direct support plus short-range completion from coherent verified
    /// neighbours. Completion never crosses an inconsistent correction field.
    pub supported_fraction: f32,
    /// Median local correction in reference-raster pixels.
    pub median_correction_px: f32,
    /// Mean confidence of the verified local nodes.
    pub mean_confidence: f32,
}

#[derive(Clone, Debug)]
pub struct ResolutionWarp {
    pub warp: Warp,
    pub report: ResolutionAlignmentReport,
}

/// Local linearization of reference-raster coordinates into a module raster.
/// The inverse maps a module-pixel displacement back to reference pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InverseWarpJacobian {
    pub xx: f32,
    pub xy: f32,
    pub yx: f32,
    pub yy: f32,
}

impl InverseWarpJacobian {
    #[inline]
    pub fn map(self, dx: f32, dy: f32) -> [f32; 2] {
        [self.xx * dx + self.xy * dy, self.yx * dx + self.yy * dy]
    }

    /// Conservative source-coordinate radius containing an output-space disc.
    pub fn source_radius(self, output_radius: f32, scale: f32) -> usize {
        let inverse_scale = scale.max(1.0e-4).recip();
        // Invert the inverse Jacobian again only for a support bound. Its
        // column L1 norms safely enclose the transformed square.
        let determinant = self.xx * self.yy - self.xy * self.yx;
        if !determinant.is_finite() || determinant.abs() < 1.0e-8 {
            return 0;
        }
        let jxx = self.yy / determinant;
        let jxy = -self.xy / determinant;
        let jyx = -self.yx / determinant;
        let jyy = self.xx / determinant;
        let extent =
            (jxx.abs() + jxy.abs()).max(jyx.abs() + jyy.abs()) * output_radius * inverse_scale;
        extent.ceil().clamp(1.0, 8.0) as usize
    }
}

/// Central finite-difference Jacobian. This follows the final dense/depth warp,
/// so distortion and local parallax are included rather than approximated by a
/// single capture-wide homography.
pub(crate) fn inverse_warp_jacobian(warp: &Warp, x: f32, y: f32) -> Option<InverseWarpJacobian> {
    const STEP: f32 = 0.5;
    let left = warp.map(x - STEP, y)?;
    let right = warp.map(x + STEP, y)?;
    let above = warp.map(x, y - STEP)?;
    let below = warp.map(x, y + STEP)?;
    let jxx = (right[0] - left[0]) / (2.0 * STEP);
    let jyx = (right[1] - left[1]) / (2.0 * STEP);
    let jxy = (below[0] - above[0]) / (2.0 * STEP);
    let jyy = (below[1] - above[1]) / (2.0 * STEP);
    let determinant = jxx * jyy - jxy * jyx;
    if !determinant.is_finite() || determinant.abs() < 1.0e-6 {
        return None;
    }
    Some(InverseWarpJacobian {
        xx: jyy / determinant,
        xy: -jxy / determinant,
        yx: -jyx / determinant,
        yy: jxx / determinant,
    })
}

/// Radial compact Hann window. Positive-only weights avoid ringing and make
/// the accumulated support meaningful as a confidence measure.
#[inline]
pub(crate) fn hann_weight(dx: f32, dy: f32, radius: f32) -> f32 {
    let distance = dx.hypot(dy);
    if !distance.is_finite() || distance >= radius || radius <= 0.0 {
        return 0.0;
    }
    0.5 + 0.5 * (std::f32::consts::PI * distance / radius).cos()
}

/// Compact Hann window steered by the reference image's local edge. Near a
/// strong edge, support is narrow in the gradient direction and wider along
/// the edge. This gathers more agreeing samples without averaging across a
/// branch, cable, or other thin boundary. Flat regions remain isotropic.
#[inline]
pub(crate) fn edge_aligned_hann_weight(
    dx: f32,
    dy: f32,
    radius: f32,
    reference_structure: Option<[f32; 3]>,
) -> f32 {
    let Some([gx, gy, _]) = reference_structure else {
        return hann_weight(dx, dy, radius);
    };
    let gradient = gx.hypot(gy);
    if !gradient.is_finite() || gradient <= 1.0e-6 {
        return hann_weight(dx, dy, radius);
    }

    // Fade in steering so noise in nominally flat areas cannot pick an
    // arbitrary kernel direction. The structure signal is log-luminance per
    // reference pixel; 0.02 begins steering and 0.10 is a decisive edge.
    let steering = smoothstep((gradient - 0.02) / 0.08);
    let across_radius = radius * (1.0 - 0.28 * steering);
    let along_radius = radius * (1.0 + 0.50 * steering);
    let normal = [gx / gradient, gy / gradient];
    let across = dx * normal[0] + dy * normal[1];
    let along = -dx * normal[1] + dy * normal[0];
    let normalized = (across / across_radius).hypot(along / along_radius);
    if !normalized.is_finite() || normalized >= 1.0 {
        return 0.0;
    }
    0.5 + 0.5 * (std::f32::consts::PI * normalized).cos()
}

/// Confidence from both independent camera count and sampling-phase spread.
/// Coincident grids improve noise but cannot resolve new spatial frequencies.
pub(crate) fn reconstruction_confidence(cameras: usize, phase_spread: f32) -> f32 {
    if cameras < 2 {
        return 0.0;
    }
    let camera_support = smoothstep((cameras as f32 - 1.0) / 2.0);
    let phase_support = smoothstep((phase_spread - 0.04) / 0.24);
    camera_support * phase_support
}

/// Refine a geometry/depth warp on a sparse grid specifically for resolution
/// reconstruction. Matching happens after the target has been brought to the
/// reference sampling domain; the returned warp still addresses the original
/// target raster, from which synthesis retrieves its sharp samples.
///
/// Unsupported nodes retain the base mapping but get zero confidence, so the
/// normal fusion path remains available while unverified offsets cannot add
/// high-frequency coefficients.
pub fn refine_resolution_warp(
    reference: &Plane,
    target: &Plane,
    base: &Warp,
    width: usize,
    height: usize,
) -> ResolutionWarp {
    let centre = [width as f32 * 0.5, height as f32 * 0.5];
    let target_magnification = base
        .magnification(centre[0], centre[1])
        .unwrap_or(1.0)
        .clamp(0.5, 4.0);
    // A tele module's residual disparity is expressed on the reference grid.
    // The old two-pixel half-resolution search could correct at most four
    // reference pixels, exactly where several C modules in difficult captures
    // reached the boundary. Give magnified contributors enough room without
    // multiplying the cost for same-focal-length modules.
    let match_radius = if target_magnification > 1.35 {
        MAGNIFIED_MATCH_RADIUS
    } else {
        LOCAL_MATCH_RADIUS
    };
    let mut rendered = Plane::new(reference.width, reference.height);
    for y in 0..rendered.height {
        for x in 0..rendered.width {
            let rx = x as f32 * 2.0 + 0.5;
            let ry = y as f32 * 2.0 + 0.5;
            rendered.data[y * rendered.width + x] = base
                .map(rx, ry)
                .and_then(|q| {
                    sample_at_reference_bandwidth(
                        target,
                        (q[0] + 0.5) * 0.5 - 0.5,
                        (q[1] + 0.5) * 0.5 - 0.5,
                        target_magnification,
                    )
                })
                .unwrap_or(f32::NAN);
        }
    }

    let columns = width.div_ceil(LOCAL_WARP_STEP) + 1;
    let rows = height.div_ceil(LOCAL_WARP_STEP) + 1;
    let mut measured = vec![None::<([f32; 2], f32)>; columns * rows];
    for row in 0..rows {
        for column in 0..columns {
            let rx = (column * LOCAL_WARP_STEP) as f32;
            let ry = (row * LOCAL_WARP_STEP) as f32;
            if rx >= width as f32 || ry >= height as f32 {
                continue;
            }
            let plane_x = ((rx - 0.5) * 0.5).round() as isize;
            let plane_y = ((ry - 0.5) * 0.5).round() as isize;
            let x = plane_x - LOCAL_MATCH_PATCH as isize / 2;
            let y = plane_y - LOCAL_MATCH_PATCH as isize / 2;
            if x < match_radius as isize
                || y < match_radius as isize
                || x + LOCAL_MATCH_PATCH as isize + match_radius as isize > reference.width as isize
                || y + LOCAL_MATCH_PATCH as isize + match_radius as isize
                    > reference.height as isize
            {
                continue;
            }
            let (x, y) = (x as usize, y as usize);
            if rendered_window_is_covered(&rendered, x, y, LOCAL_MATCH_PATCH, match_radius)
                && let Some(found) =
                    match_patch(reference, &rendered, x, y, LOCAL_MATCH_PATCH, match_radius)
                && found.score >= LOCAL_MINIMUM_SCORE
            {
                measured[row * columns + column] =
                    Some(([found.shift[0] * 2.0, found.shift[1] * 2.0], found.score));
            }
        }
    }

    // A real calibration residual varies smoothly over a 32 px cell. Reject
    // isolated correlation peaks and use the neighbourhood median to keep
    // periodic textures from producing discontinuous one-cell offsets.
    let mut points = Vec::with_capacity(columns * rows);
    let mut confidence = Vec::with_capacity(columns * rows);
    let mut correction_grid = Vec::with_capacity(columns * rows);
    let mut corrections = Vec::new();
    let mut confidence_sum = 0.0f32;
    for row in 0..rows {
        for column in 0..columns {
            let rx = (column * LOCAL_WARP_STEP) as f32;
            let ry = (row * LOCAL_WARP_STEP) as f32;
            let index = row * columns + column;
            let refined = measured[index].and_then(|(centre, centre_score)| {
                let mut xs = Vec::with_capacity(9);
                let mut ys = Vec::with_capacity(9);
                let first_row = row.saturating_sub(1);
                let last_row = (row + 1).min(rows - 1);
                let first_column = column.saturating_sub(1);
                let last_column = (column + 1).min(columns - 1);
                for neighbour_row in first_row..=last_row {
                    for neighbour_column in first_column..=last_column {
                        if let Some((shift, _)) =
                            measured[neighbour_row * columns + neighbour_column]
                        {
                            xs.push(shift[0]);
                            ys.push(shift[1]);
                        }
                    }
                }
                if xs.len() < 3 {
                    return None;
                }
                xs.sort_by(f32::total_cmp);
                ys.sort_by(f32::total_cmp);
                let median = [xs[xs.len() / 2], ys[ys.len() / 2]];
                if (centre[0] - median[0]).hypot(centre[1] - median[1]) > 1.5 {
                    return None;
                }
                let q = base.map(rx + median[0], ry + median[1])?;
                let score_confidence = smoothstep((centre_score - LOCAL_MINIMUM_SCORE) / 0.20);
                let base_confidence = base.confidence(rx, ry);
                Some((q, score_confidence * base_confidence, median))
            });
            if let Some((point, local_confidence, correction)) = refined {
                points.push(point);
                confidence.push(local_confidence);
                correction_grid.push(Some(correction));
                corrections.push(correction[0].hypot(correction[1]));
                confidence_sum += local_confidence;
            } else {
                points.push(base.map(rx, ry).unwrap_or([f32::NAN; 2]));
                confidence.push(0.0);
                correction_grid.push(None);
            }
        }
    }
    // Complete only short holes inside a coherent local correction surface.
    // This recovers textureless cells between measurements without turning a
    // foreground/background discontinuity into a smooth but wrong warp.
    for _ in 0..2 {
        let correction_snapshot = correction_grid.clone();
        let confidence_snapshot = confidence.clone();
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                if correction_snapshot[index].is_some() {
                    continue;
                }
                let mut xs = Vec::with_capacity(8);
                let mut ys = Vec::with_capacity(8);
                let mut confidence_sum = 0.0f32;
                for neighbour_row in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                    for neighbour_column in column.saturating_sub(1)..=(column + 1).min(columns - 1)
                    {
                        let neighbour = neighbour_row * columns + neighbour_column;
                        if let Some(correction) = correction_snapshot[neighbour] {
                            xs.push(correction[0]);
                            ys.push(correction[1]);
                            confidence_sum += confidence_snapshot[neighbour];
                        }
                    }
                }
                if xs.len() < 3 {
                    continue;
                }
                xs.sort_by(f32::total_cmp);
                ys.sort_by(f32::total_cmp);
                let correction = [xs[xs.len() / 2], ys[ys.len() / 2]];
                let coherent = xs
                    .iter()
                    .zip(&ys)
                    .all(|(&x, &y)| (x - correction[0]).hypot(y - correction[1]) <= 1.5);
                if !coherent {
                    continue;
                }
                let rx = (column * LOCAL_WARP_STEP) as f32;
                let ry = (row * LOCAL_WARP_STEP) as f32;
                if let Some(point) = base.map(rx + correction[0], ry + correction[1]) {
                    points[index] = point;
                    // Two completion passes decay naturally because inferred
                    // neighbours already carry this reduced confidence.
                    confidence[index] = 0.45 * confidence_sum / xs.len() as f32;
                    correction_grid[index] = Some(correction);
                }
            }
        }
    }
    corrections.sort_by(f32::total_cmp);
    let verified = corrections.len();
    let supported = confidence.iter().filter(|&&value| value > 0.0).count();
    ResolutionWarp {
        warp: Warp {
            step: LOCAL_WARP_STEP,
            columns,
            rows,
            points,
            confidence,
        },
        report: ResolutionAlignmentReport {
            verified_fraction: verified as f32 / (columns * rows).max(1) as f32,
            supported_fraction: supported as f32 / (columns * rows).max(1) as f32,
            median_correction_px: corrections.get(verified / 2).copied().unwrap_or_default(),
            mean_confidence: confidence_sum / verified.max(1) as f32,
        },
    }
}

/// Bring a sharper target into the reference camera's matching bandwidth.
/// Sampling one tele pixel aliases fine texture into the NCC surface and makes
/// the correct peak look inconsistent with the wider reference module. A
/// compact Gaussian footprint approximates the area represented by one
/// reference-plane sample; synthesis still retrieves the original unblurred
/// target samples after the correspondence has been verified.
fn sample_at_reference_bandwidth(
    target: &Plane,
    x: f32,
    y: f32,
    magnification: f32,
) -> Option<f32> {
    if magnification <= 1.15 {
        return target.sample(x, y);
    }
    let sigma = (0.42 * magnification).max(0.55);
    let radius = (2.0 * sigma).ceil().clamp(1.0, 4.0) as isize;
    let mut sum = 0.0f32;
    let mut weight = 0.0f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let distance_squared = (dx * dx + dy * dy) as f32;
            let kernel = (-0.5 * distance_squared / (sigma * sigma)).exp();
            if let Some(value) = target.sample(x + dx as f32, y + dy as f32) {
                sum += kernel * value;
                weight += kernel;
            }
        }
    }
    (weight > 1.0e-8).then_some(sum / weight)
}

fn rendered_window_is_covered(
    rendered: &Plane,
    x: usize,
    y: usize,
    size: usize,
    radius: usize,
) -> bool {
    let x0 = x - radius;
    let y0 = y - radius;
    let x1 = x + size + radius;
    let y1 = y + size + radius;
    (y0..y1).all(|row| {
        rendered.data[row * rendered.width + x0..row * rendered.width + x1]
            .iter()
            .all(|value| value.is_finite())
    })
}

#[inline]
fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_cfa_is_the_resolution_default() {
        assert_eq!(
            ResolutionReconstruction::default(),
            ResolutionReconstruction::JointCfa
        );
    }

    #[test]
    fn hann_is_compact_smooth_and_positive() {
        assert_eq!(hann_weight(0.0, 0.0, 1.5), 1.0);
        assert!(hann_weight(0.5, 0.0, 1.5) > hann_weight(1.0, 0.0, 1.5));
        assert_eq!(hann_weight(1.5, 0.0, 1.5), 0.0);
        assert_eq!(hann_weight(2.0, 0.0, 1.5), 0.0);
    }

    #[test]
    fn edge_aligned_hann_is_narrow_across_and_wide_along_an_edge() {
        let vertical_edge = Some([0.2, 0.0, 0.0]);
        assert_eq!(edge_aligned_hann_weight(0.8, 0.0, 1.0, vertical_edge), 0.0);
        assert!(edge_aligned_hann_weight(0.0, 1.2, 1.0, vertical_edge) > 0.0);
        assert_eq!(edge_aligned_hann_weight(0.0, 1.5, 1.0, vertical_edge), 0.0);
    }

    #[test]
    fn distinct_phases_are_required_for_resolution_confidence() {
        assert_eq!(reconstruction_confidence(1, 0.5), 0.0);
        assert_eq!(reconstruction_confidence(4, 0.0), 0.0);
        assert!(reconstruction_confidence(2, 0.15) > 0.0);
        assert!(reconstruction_confidence(3, 0.30) > reconstruction_confidence(2, 0.15));
    }

    #[test]
    fn inverse_jacobian_maps_a_known_affine_warp() {
        let warp = Warp::from_fn(100, 100, 4, |p| {
            Some([2.0 * p[0] + 0.25 * p[1], -0.5 * p[0] + 1.5 * p[1]])
        });
        let inverse = inverse_warp_jacobian(&warp, 50.0, 50.0).unwrap();
        let source_delta = [2.25, 1.0];
        let reference_delta = inverse.map(source_delta[0], source_delta[1]);
        assert!((reference_delta[0] - 1.0).abs() < 1.0e-4);
        assert!((reference_delta[1] - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn local_resolution_warp_recovers_a_subpixel_plane_shift() {
        let signal = |x: f32, y: f32| {
            (x * 0.27).sin() * (y * 0.19).cos()
                + 0.35 * (x * 0.073 + y * 0.11).sin()
                + 0.12 * (x * 0.41 - y * 0.037).cos()
        };
        let (plane_width, plane_height) = (256, 192);
        let expected_plane_shift = [0.6, -0.35];
        let mut reference = Plane::new(plane_width, plane_height);
        let mut target = Plane::new(plane_width, plane_height);
        for y in 0..plane_height {
            for x in 0..plane_width {
                reference.data[y * plane_width + x] = signal(x as f32, y as f32);
                target.data[y * plane_width + x] = signal(
                    x as f32 - expected_plane_shift[0],
                    y as f32 - expected_plane_shift[1],
                );
            }
        }
        let (width, height) = (plane_width * 2, plane_height * 2);
        let base = Warp::from_fn(width, height, 32, Some);
        let refined = refine_resolution_warp(&reference, &target, &base, width, height);
        let p = [256.0, 192.0];
        let q = refined.warp.map(p[0], p[1]).unwrap();
        assert!((q[0] - p[0] - expected_plane_shift[0] * 2.0).abs() < 0.35);
        assert!((q[1] - p[1] - expected_plane_shift[1] * 2.0).abs() < 0.35);
        assert!(refined.warp.confidence(p[0], p[1]) > 0.5);
        assert!(refined.report.verified_fraction > 0.5);
        assert!(refined.report.supported_fraction >= refined.report.verified_fraction);
    }
}
