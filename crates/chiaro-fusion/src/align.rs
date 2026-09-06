//! Stage 2: align every module to the reference module.
//!
//! Output is one [`Warp`] per module: a function from reference-raster pixels
//! to the module's raster pixels, represented as a coarse grid of sampled
//! target coordinates that synthesis interpolates bilinearly. Grids are model
//! agnostic, so a later depth-aware warp can replace the homography without
//! touching synthesis.
//!
//! The warp is built in two steps:
//!
//! 1. **Initialisation.** With calibration, the factory camera model maps each
//!    reference pixel to the module at a very large depth (pure rotation, the
//!    correct model for distant scenes; it includes lens distortion). Without
//!    calibration a nominal focal-group scale about the image centres is used
//!    and a coarse global search finds the translation.
//! 2. **Refinement.** Measured on real captures, the factory model lands
//!    20-60 px off with 10-50 px of spread, so a coarse-to-fine normalised
//!    cross-correlation search measures local shifts on a pyramid of the
//!    log-luminance images, and a RANSAC homography of the reference raster
//!    (`p -> C(p)`) is fitted so that `M'(p) = M(C(p))`. Confidence comes from
//!    the correlation peak, which discards textureless sky automatically.
//!
//! The report records per-module statistics (initial offset, inliers, residual
//! quantiles) so alignment quality can be inspected after every export.

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::depth::{DepthAlignmentReport, DepthOptions};
use crate::geometry::ResolvedCamera;
use crate::image::{Plane, match_patch};
use crate::math::{Mat3, Vec2, apply_homography};

/// Reference-raster pixel -> module-raster pixel, sampled on a regular grid.
#[derive(Clone, Debug)]
pub struct Warp {
    pub step: usize,
    pub columns: usize,
    pub rows: usize,
    /// `columns * rows` target coordinates; NaN where the mapping is undefined.
    pub points: Vec<[f32; 2]>,
    /// Local synthesis confidence at every grid node. Alignment-only warps use
    /// one; depth ambiguity and occlusion may lower it towards zero.
    pub confidence: Vec<f32>,
}

impl Warp {
    /// Sample the mapping by evaluating `map` on a `step`-spaced grid that
    /// covers `0..=width` x `0..=height` of the reference raster.
    pub fn from_fn(
        width: usize,
        height: usize,
        step: usize,
        map: impl Fn(Vec2) -> Option<Vec2>,
    ) -> Self {
        let columns = width.div_ceil(step) + 1;
        let rows = height.div_ceil(step) + 1;
        let mut points = Vec::with_capacity(columns * rows);
        let mut confidence = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for column in 0..columns {
                let p = [(column * step) as f64, (row * step) as f64];
                match map(p) {
                    Some(q) => {
                        points.push([q[0] as f32, q[1] as f32]);
                        confidence.push(1.0);
                    }
                    None => {
                        points.push([f32::NAN, f32::NAN]);
                        confidence.push(0.0);
                    }
                }
            }
        }
        Self {
            step,
            columns,
            rows,
            points,
            confidence,
        }
    }

    /// Target coordinates for a reference pixel; `None` where undefined.
    #[inline]
    pub fn map(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let fx = x / self.step as f32;
        let fy = y / self.step as f32;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let c0 = (fx.floor() as usize).min(self.columns - 1);
        let r0 = (fy.floor() as usize).min(self.rows - 1);
        let c1 = (c0 + 1).min(self.columns - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f32;
        let ty = fy - r0 as f32;
        let p = |c: usize, r: usize| self.points[r * self.columns + c];
        let (a, b, c, d) = (p(c0, r0), p(c1, r0), p(c0, r1), p(c1, r1));
        let mut out = [0.0f32; 2];
        for k in 0..2 {
            let top = a[k] * (1.0 - tx) + b[k] * tx;
            let bottom = c[k] * (1.0 - tx) + d[k] * tx;
            out[k] = top * (1.0 - ty) + bottom * ty;
        }
        if out[0].is_nan() || out[1].is_nan() {
            None
        } else {
            Some(out)
        }
    }

    /// Bilinearly interpolated local confidence for synthesis.
    #[inline]
    pub fn confidence(&self, x: f32, y: f32) -> f32 {
        let fx = x / self.step as f32;
        let fy = y / self.step as f32;
        if fx < 0.0 || fy < 0.0 || self.confidence.len() != self.points.len() {
            return 0.0;
        }
        let c0 = (fx.floor() as usize).min(self.columns - 1);
        let r0 = (fy.floor() as usize).min(self.rows - 1);
        let c1 = (c0 + 1).min(self.columns - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f32;
        let ty = fy - r0 as f32;
        let value = |column: usize, row: usize| self.confidence[row * self.columns + column];
        let top = value(c0, r0) * (1.0 - tx) + value(c1, r0) * tx;
        let bottom = value(c0, r1) * (1.0 - tx) + value(c1, r1) * tx;
        (top * (1.0 - ty) + bottom * ty).clamp(0.0, 1.0)
    }

    /// Local magnification (target pixels per reference pixel) at a point,
    /// from finite differences of the grid.
    pub fn magnification(&self, x: f32, y: f32) -> Option<f32> {
        let h = self.step as f32;
        let a = self.map(x, y)?;
        let b = self.map(x + h, y)?;
        let c = self.map(x, y + h)?;
        let dx = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / h;
        let dy = ((c[0] - a[0]).powi(2) + (c[1] - a[1]).powi(2)).sqrt() / h;
        Some(((dx * dy).sqrt()).max(1e-6))
    }
}

/// Alignment of one module to the reference.
#[derive(Clone, Debug)]
pub struct ModuleAlignment {
    pub name: String,
    pub warp: Warp,
    /// Finest-level image observations retained in physical raster
    /// coordinates for capture-specific rig refinement. These are not a warp:
    /// each entry is one independently matched patch centre.
    pub correspondences: Vec<AlignmentCorrespondence>,
    /// Luminance match to the reference, applied as `gain * (sample - offset)`
    /// to every channel (filled in by the pipeline's photometric step; a rough
    /// estimate from the alignment planes until then).
    pub gain: f32,
    pub offset: f32,
    pub report: AlignmentReport,
}

/// One cross-camera patch observation exposed to the physical rig optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignmentCorrespondence {
    pub reference_pixel: Vec2,
    pub target_pixel: Vec2,
    /// Normalized cross-correlation peak.
    pub confidence: f32,
    /// Target pixels per reference pixel at this location.
    pub local_scale: f32,
    /// Standard deviation of the reference log-luminance match window.
    pub structure: f32,
    /// Filled by later depth-aware matchers when available. The initial
    /// physical solve deliberately works without requiring dense depth.
    pub depth_reliability: Option<f32>,
}

/// Diagnostics written next to the fused output.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AlignmentReport {
    pub camera: String,
    pub initialised_from: &'static str,
    /// Capture autofocus result for the exposure group containing this module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_achieved: Option<bool>,
    /// Object-space focus distance interpolated from factory calibration and
    /// the captured lens Hall position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_focus_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disparity_focus_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast_focus_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_roi: Option<[f64; 2]>,
    pub lens_timeout: bool,
    pub mirror_timeout: bool,
    /// Fraction of the reference frame this module covers.
    pub coverage: f32,
    /// Median correction applied to the factory model, reference pixels.
    pub correction_median_px: [f32; 2],
    /// Per-level refinement: (level scale, patches tried, inliers, median residual px).
    pub levels: Vec<LevelReport>,
    pub inliers: usize,
    pub patches: usize,
    pub residual_median_px: f32,
    pub residual_p90_px: f32,
    /// Fraction of the finest-level patch matches consistent with the robust
    /// model. Low consensus usually means competing scene depths or a false
    /// correlation, even when the inlier residual itself is small.
    pub inlier_ratio: f32,
    /// Local calibrated inverse-depth refinement, when both camera models were
    /// available and depth-aware alignment was enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<DepthAlignmentReport>,
    /// Whether synthesis may use this module.
    pub accepted: bool,
    pub status: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LevelReport {
    pub scale: usize,
    pub patches: usize,
    pub inliers: usize,
    pub median_residual_px: f32,
}

/// Tunables of the refinement search.
#[derive(Clone, Debug)]
pub struct AlignOptions {
    /// Spacing of the output warp grid in reference pixels.
    pub grid_step: usize,
    /// Correlation window, pixels of the (half-resolution) luminance plane.
    pub patch: usize,
    /// Search radius at the coarsest level, plane pixels.
    pub coarse_radius: usize,
    /// Minimum NCC peak for a patch to vote.
    pub min_score: f32,
    /// RANSAC inlier threshold in reference pixels (full resolution).
    pub inlier_px: f32,
    /// Minimum finest-level correspondence consensus required for synthesis.
    pub min_inlier_ratio: f32,
    /// Skip refinement and keep the factory model (for diagnostics).
    pub refine: bool,
    /// Conservative local parallax refinement after the global homography.
    pub depth: DepthOptions,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            grid_step: 32,
            patch: 32,
            coarse_radius: 12,
            min_score: 0.5,
            inlier_px: 3.0,
            min_inlier_ratio: 0.45,
            refine: true,
            depth: DepthOptions::default(),
        }
    }
}

/// One module's inputs to alignment.
pub struct AlignInput<'a> {
    pub name: &'a str,
    /// Half-resolution log luminance of the module.
    pub luminance: &'a Plane,
    pub width: usize,
    pub height: usize,
    /// Resolved camera model, if calibration is available.
    pub camera: Option<&'a ResolvedCamera>,
    /// Nominal focal length in pixels (used when `camera` is `None`).
    pub nominal_focal_px: f64,
}

#[derive(Clone, Copy)]
pub struct AlignmentSeed<'a> {
    pub warp: &'a Warp,
    pub name: &'static str,
}

/// Depth used for the rotation-only initialisation (calibration units).
const FAR_DEPTH: f64 = 1.0e8;

/// Align `target` to `reference` (same structure; the reference aligns to
/// itself with an identity warp).
pub fn align_module(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    options: &AlignOptions,
) -> Result<ModuleAlignment> {
    align_module_seeded(reference, target, options, None)
}

/// [`align_module`] with an optional externally predicted reference-to-target
/// warp. Temporal burst processing uses an IMU rotation here; correlation
/// still refines the seed and decides whether the result is trustworthy.
pub fn align_module_seeded(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    options: &AlignOptions,
    seed: Option<AlignmentSeed<'_>>,
) -> Result<ModuleAlignment> {
    let (width, height) = (reference.width, reference.height);
    let mut report = AlignmentReport {
        camera: target.name.to_owned(),
        ..Default::default()
    };
    if target.name == reference.name {
        let warp = Warp::from_fn(width, height, options.grid_step, Some);
        report.initialised_from = "reference";
        report.coverage = 1.0;
        report.inlier_ratio = 1.0;
        report.accepted = true;
        report.status = "reference".to_owned();
        return Ok(ModuleAlignment {
            name: target.name.to_owned(),
            warp,
            correspondences: Vec::new(),
            gain: 1.0,
            offset: 0.0,
            report,
        });
    }

    // Step 1: initial mapping, tabulated on a fine grid. The camera model
    // (iterative undistortion) is too slow to evaluate per pixel, and the
    // mapping is smooth, so bilinear interpolation of an 8 px grid is exact
    // to well under 0.01 px.
    let initial_grid = match (seed, reference.camera, target.camera) {
        (Some(seed), _, _) => {
            report.initialised_from = seed.name;
            seed.warp.clone()
        }
        (None, Some(reference_camera), Some(target_camera)) => {
            report.initialised_from = "calibration";
            Warp::from_fn(width, height, 8, |p| {
                target_camera
                    .map_from(reference_camera, p, FAR_DEPTH)
                    .filter(|q| q[0].is_finite() && q[1].is_finite())
            })
        }
        (None, _, _) => {
            report.initialised_from = "nominal focal group";
            let scale = target.nominal_focal_px / reference.nominal_focal_px;
            let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
            let (tx, ty) = (
                (target.width as f64 - 1.0) / 2.0,
                (target.height as f64 - 1.0) / 2.0,
            );
            let shift = coarse_global_shift(reference, target, scale, options)?;
            Warp::from_fn(width, height, 8, |p| {
                Some([
                    (p[0] - cx) * scale + tx + shift[0],
                    (p[1] - cy) * scale + ty + shift[1],
                ])
            })
        }
    };
    let initial = |p: Vec2| -> Option<Vec2> {
        let q = initial_grid.map(p[0] as f32, p[1] as f32)?;
        Some([f64::from(q[0]), f64::from(q[1])])
    };

    // Step 2: coarse-to-fine refinement of a reference-space homography C.
    let mut correction: Mat3 = crate::math::IDENTITY;
    let mut finest_residuals: Vec<f32> = Vec::new();
    let mut correspondences = Vec::new();
    if options.refine {
        let reference_pyramid = reference.luminance.pyramid(96);
        let target_pyramid = target.luminance.pyramid(48);
        // Magnification of the module relative to the reference: how many
        // target luminance pixels one reference luminance pixel covers.
        let magnification = {
            let c = [(width as f64) / 2.0, (height as f64) / 2.0];
            let h = 64.0;
            match (
                initial(c),
                initial([c[0] + h, c[1]]),
                initial([c[0], c[1] + h]),
            ) {
                (Some(a), Some(b), Some(d)) => {
                    let dx = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() / h;
                    let dy = ((d[0] - a[0]).powi(2) + (d[1] - a[1]).powi(2)).sqrt() / h;
                    (dx * dy).sqrt()
                }
                _ => target.nominal_focal_px / reference.nominal_focal_px,
            }
        };
        for level in (0..reference_pyramid.len()).rev() {
            let scale = 1usize << level; // reference luminance pixels per level pixel
            let reference_plane = &reference_pyramid[level];
            // Render the target through the current mapping at this level's
            // sampling density, taking the target pyramid level whose pixel
            // size best matches (accounts for the magnification).
            let target_level = ((scale as f64 * magnification).log2().round().max(0.0) as usize)
                .min(target_pyramid.len() - 1);
            let target_plane = &target_pyramid[target_level];
            let target_scale = (1usize << target_level) as f64;
            let rendered = render_through(
                reference_plane,
                scale,
                target_plane,
                target_scale,
                &initial,
                &correction,
            );
            let radius = if level + 1 == reference_pyramid.len() {
                options.coarse_radius
            } else {
                3
            };
            let patch = options.patch.min(reference_plane.width / 4).max(8);
            let mut pairs = Vec::new();
            let mut pair_structure = Vec::new();
            let stride = patch / 2;
            let mut y = radius;
            while y + patch + radius <= reference_plane.height {
                let mut x = radius;
                while x + patch + radius <= reference_plane.width {
                    if rendered_covered(&rendered, x, y, patch)
                        && let Some(found) =
                            match_patch(reference_plane, &rendered, x, y, patch, radius)
                        && found.score >= options.min_score
                    {
                        // Reference-raster coordinates, same pixel-centre
                        // convention as `render_through`.
                        let to_reference = |px: f32, py: f32| -> Vec2 {
                            let convert =
                                |v: f32| f64::from(v) * scale as f64 * 2.0 + scale as f64 - 0.5;
                            [convert(px), convert(py)]
                        };
                        let centre = to_reference(
                            x as f32 + patch as f32 / 2.0,
                            y as f32 + patch as f32 / 2.0,
                        );
                        let shifted = to_reference(
                            x as f32 + patch as f32 / 2.0 + found.shift[0],
                            y as f32 + patch as f32 / 2.0 + found.shift[1],
                        );
                        // The rendered image at `shifted` shows what the
                        // reference shows at `centre`: C maps centre -> shifted
                        // (in the current corrected frame).
                        pairs.push((centre, shifted, found.score));
                        pair_structure.push(reference_plane.window_std(x, y, patch));
                    }
                    x += stride;
                }
                y += stride;
            }
            let threshold = (options.inlier_px * scale as f32 * 2.0).max(options.inlier_px);
            if let Some((update, inliers, residuals)) = fit_homography_ransac(&pairs, threshold) {
                if level == 0 {
                    correspondences = inliers
                        .iter()
                        .filter_map(|&index| {
                            let (reference_pixel, shifted, confidence) = pairs[index];
                            let corrected = apply_homography(&correction, shifted)?;
                            let target_pixel = initial(corrected)?;
                            Some(AlignmentCorrespondence {
                                reference_pixel,
                                target_pixel,
                                confidence,
                                local_scale: magnification as f32,
                                structure: pair_structure[index],
                                depth_reliability: None,
                            })
                        })
                        .collect();
                }
                correction = crate::math::mul(&correction, &update);
                report.levels.push(LevelReport {
                    scale: scale * 2,
                    patches: pairs.len(),
                    inliers: inliers.len(),
                    median_residual_px: residuals[residuals.len() / 2],
                });
                if level == 0 {
                    finest_residuals = residuals;
                    report.patches = pairs.len();
                    report.inliers = inliers.len();
                }
            } else {
                report.levels.push(LevelReport {
                    scale: scale * 2,
                    patches: pairs.len(),
                    inliers: 0,
                    median_residual_px: f32::NAN,
                });
            }
        }
    }

    // Final warp: M'(p) = M(C(p)).
    let warp = Warp::from_fn(width, height, options.grid_step, |p| {
        let corrected = apply_homography(&correction, p)?;
        initial(corrected)
    });
    let covered = warp
        .points
        .iter()
        .filter(|q| {
            q[0].is_finite()
                && q[0] >= 0.0
                && q[1] >= 0.0
                && q[0] <= (target.width - 1) as f32
                && q[1] <= (target.height - 1) as f32
        })
        .count();
    report.coverage = covered as f32 / warp.points.len() as f32;

    // Residuals of the finest-level fit (sorted) and the total correction at
    // the frame centre.
    if !finest_residuals.is_empty() {
        let residuals = &finest_residuals;
        report.residual_median_px = residuals[residuals.len() / 2];
        report.residual_p90_px = residuals[(residuals.len() * 9 / 10).min(residuals.len() - 1)];
    }
    {
        let centre = [(width / 2) as f64, (height / 2) as f64];
        let moved = apply_homography(&correction, centre).unwrap_or(centre);
        report.correction_median_px =
            [(moved[0] - centre[0]) as f32, (moved[1] - centre[1]) as f32];
    }

    report.inlier_ratio = if report.patches == 0 {
        0.0
    } else {
        report.inliers as f32 / report.patches as f32
    };
    report.accepted = !options.refine || report.inlier_ratio >= options.min_inlier_ratio;
    report.status = if !report.accepted {
        format!(
            "rejected: {:.0}% correspondence consensus",
            report.inlier_ratio * 100.0
        )
    } else if !options.refine {
        "factory model only".to_owned()
    } else if report.inliers >= 12 && report.residual_median_px < options.inlier_px {
        "refined".to_owned()
    } else if report.inliers > 0 {
        "weak refinement".to_owned()
    } else {
        "no correlation support; factory model kept".to_owned()
    };

    // Photometric gain: median ratio of reference to rendered target luminance
    // (the planes are log luminance, so the ratio is exp of the difference).
    let gain = photometric_gain(reference.luminance, target.luminance, &warp).unwrap_or(1.0);

    Ok(ModuleAlignment {
        name: target.name.to_owned(),
        warp,
        correspondences,
        gain,
        offset: 0.0,
        report,
    })
}

/// Render the target luminance into the reference grid of one pyramid level
/// through `initial(correction(p))`. NaN marks uncovered pixels.
fn render_through(
    reference_plane: &Plane,
    reference_scale: usize,
    target_plane: &Plane,
    target_scale: f64,
    initial: &dyn Fn(Vec2) -> Option<Vec2>,
    correction: &Mat3,
) -> Plane {
    let mut out = Plane::new(reference_plane.width, reference_plane.height);
    // Luminance planes are half resolution: plane pixel i covers raster 2i..2i+2,
    // centred at 2i + 0.5 (raster pixel centres at integer coordinates).
    let to_raster =
        |v: f32| (f64::from(v) * reference_scale as f64 * 2.0) + (reference_scale as f64) - 0.5;
    let from_raster = |v: f64| ((v + 0.5) / (2.0 * target_scale) - 0.5) as f32;
    for y in 0..out.height {
        for x in 0..out.width {
            let p = [to_raster(x as f32), to_raster(y as f32)];
            let value = apply_homography(correction, p)
                .and_then(initial)
                .and_then(|q| target_plane.sample(from_raster(q[0]), from_raster(q[1])))
                .unwrap_or(f32::NAN);
            out.data[y * out.width + x] = value;
        }
    }
    out
}

fn rendered_covered(rendered: &Plane, x: usize, y: usize, size: usize) -> bool {
    for row in y..y + size {
        if rendered.data[row * rendered.width + x..row * rendered.width + x + size]
            .iter()
            .any(|v| v.is_nan())
        {
            return false;
        }
    }
    true
}

/// Exhaustive translation search at a coarse level for the no-calibration
/// fallback. Returns the shift in target raster pixels.
fn coarse_global_shift(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    scale: f64,
    options: &AlignOptions,
) -> Result<Vec2> {
    // Downsample the reference to ~64 px wide and the target to the same
    // angular density, then slide the smaller over the larger.
    let reference_pyramid = reference.luminance.pyramid(32);
    let reference_plane = reference_pyramid.last().unwrap();
    let reference_scale = (1usize << (reference_pyramid.len() - 1)) as f64;
    let target_pyramid = target.luminance.pyramid(16);
    let wanted = reference_scale * scale;
    let target_level = (wanted.log2().round().max(0.0) as usize).min(target_pyramid.len() - 1);
    let target_plane = &target_pyramid[target_level];
    let target_scale = (1usize << target_level) as f64;
    // Resample the target to the reference density.
    let ratio = (target_scale / wanted) as f32;
    let rw = ((target_plane.width as f32) * ratio).floor().max(4.0) as usize;
    let rh = ((target_plane.height as f32) * ratio).floor().max(4.0) as usize;
    let mut resampled = Plane::new(rw, rh);
    for y in 0..rh {
        for x in 0..rw {
            resampled.data[y * rw + x] = target_plane
                .sample(x as f32 / ratio, y as f32 / ratio)
                .unwrap_or(0.0);
        }
    }
    if rw + 2 > reference_plane.width || rh + 2 > reference_plane.height {
        // Target is wider than the reference at this density: assume centred.
        return Ok([0.0, 0.0]);
    }
    let patch = rw
        .min(rh)
        .min(reference_plane.width / 2)
        .min(reference_plane.height / 2);
    let radius = ((reference_plane.width.min(reference_plane.height) - patch) / 2).max(1);
    let rx = (reference_plane.width - patch) / 2;
    let ry = (reference_plane.height - patch) / 2;
    let tx = (rw - patch) / 2;
    let ty = (rh - patch) / 2;
    // Embed the resampled target in a plane the size of the reference so the
    // search can slide it.
    let mut embedded = Plane::new(reference_plane.width, reference_plane.height);
    embedded.data.fill(f32::NAN);
    for y in 0..patch {
        for x in 0..patch {
            embedded.data[(ry + y) * embedded.width + rx + x] = resampled.at(tx + x, ty + y);
        }
    }
    let found = match_patch(
        reference_plane,
        &embedded,
        rx,
        ry,
        patch,
        radius.min(rx).min(ry),
    )
    .context("no correlation between the module and the reference")?;
    if found.score < options.min_score * 0.6 {
        bail!(
            "module does not correlate with the reference ({:.2})",
            found.score
        );
    }
    // The target patch appears at reference (rx + shift): the target centre
    // maps to reference centre + shift (in coarse pixels) -> convert to target
    // raster pixels via the nominal scale.
    Ok([
        -f64::from(found.shift[0]) * reference_scale * 2.0 * scale,
        -f64::from(found.shift[1]) * reference_scale * 2.0 * scale,
    ])
}

/// `(C, inlier indices, sorted inlier residuals)` of a robust homography fit.
type HomographyFit = (Mat3, Vec<usize>, Vec<f32>);

/// Fit `C` with `b ~= C a` for weighted pairs `(a, b, score)` by RANSAC over
/// four-point DLT samples followed by a least-squares refit on the inliers.
/// Returns `(C, inliers, sorted inlier residuals)`.
fn fit_homography_ransac(pairs: &[(Vec2, Vec2, f32)], threshold: f32) -> Option<HomographyFit> {
    if pairs.len() < 6 {
        return None;
    }
    let mut best_inliers: Vec<usize> = Vec::new();
    // Deterministic pseudo-random sampling (LCG) keeps runs reproducible.
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ pairs.len() as u64;
    let mut next = move |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % n
    };
    let iterations = 300;
    for _ in 0..iterations {
        let mut sample = [0usize; 4];
        for slot in 0..4 {
            loop {
                let candidate = next(pairs.len());
                if !sample[..slot].contains(&candidate) {
                    sample[slot] = candidate;
                    break;
                }
            }
        }
        let subset = sample.iter().map(|&i| pairs[i]).collect::<Vec<_>>();
        let Some(h) = fit_homography_least_squares(&subset) else {
            continue;
        };
        let inliers = pairs
            .iter()
            .enumerate()
            .filter(|(_, (a, b, _))| {
                apply_homography(&h, *a).is_some_and(|p| {
                    ((p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2)).sqrt() <= f64::from(threshold)
                })
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }
    if best_inliers.len() < 6 {
        return None;
    }
    let inlier_pairs = best_inliers.iter().map(|&i| pairs[i]).collect::<Vec<_>>();
    let h = fit_homography_least_squares(&inlier_pairs)?;
    // Re-select inliers under the refit and compute residuals.
    let mut final_inliers = Vec::new();
    let mut residuals = Vec::new();
    for (index, pair) in pairs.iter().enumerate() {
        if let Some(p) = apply_homography(&h, pair.0) {
            let r = ((p[0] - pair.1[0]).powi(2) + (p[1] - pair.1[1]).powi(2)).sqrt() as f32;
            if r <= threshold {
                final_inliers.push(index);
                residuals.push(r);
            }
        }
    }
    if final_inliers.len() < 6 {
        return None;
    }
    residuals.sort_by(f32::total_cmp);
    Some((h, final_inliers, residuals))
}

/// Weighted DLT: solve `b = H a` for `H` (with `h33 = 1`) in least squares.
/// Coordinates are normalised about their centroid for conditioning.
fn fit_homography_least_squares(pairs: &[(Vec2, Vec2, f32)]) -> Option<Mat3> {
    if pairs.len() < 4 {
        return None;
    }
    let n = pairs.len() as f64;
    let mean = |f: &dyn Fn(&(Vec2, Vec2, f32)) -> f64| pairs.iter().map(f).sum::<f64>() / n;
    let (ax, ay) = (mean(&|p| p.0[0]), mean(&|p| p.0[1]));
    let (bx, by) = (mean(&|p| p.1[0]), mean(&|p| p.1[1]));
    let sa = mean(&|p| ((p.0[0] - ax).powi(2) + (p.0[1] - ay).powi(2)).sqrt()).max(1e-9);
    let sb = mean(&|p| ((p.1[0] - bx).powi(2) + (p.1[1] - by).powi(2)).sqrt()).max(1e-9);
    // Normal equations for the 8 unknowns (h33 = 1).
    let mut ata = [[0.0f64; 8]; 8];
    let mut atb = [0.0f64; 8];
    for (a, b, score) in pairs {
        let w = f64::from(*score).max(0.05);
        let (x, y) = ((a[0] - ax) / sa, (a[1] - ay) / sa);
        let (u, v) = ((b[0] - bx) / sb, (b[1] - by) / sb);
        let rows = [
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ];
        for (row, rhs) in rows {
            for i in 0..8 {
                atb[i] += w * row[i] * rhs;
                for j in 0..8 {
                    ata[i][j] += w * row[i] * row[j];
                }
            }
        }
    }
    let solution = solve_linear(ata, atb)?;
    let normalized = [
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], 1.0],
    ];
    // Undo normalisation: H = T_b^-1 * Hn * T_a.
    let t_a = [
        [1.0 / sa, 0.0, -ax / sa],
        [0.0, 1.0 / sa, -ay / sa],
        [0.0, 0.0, 1.0],
    ];
    let t_b_inv = [[sb, 0.0, bx], [0.0, sb, by], [0.0, 0.0, 1.0]];
    let h = crate::math::mul(&crate::math::mul(&t_b_inv, &normalized), &t_a);
    let scale = h[2][2];
    if scale.abs() < 1e-12 || !scale.is_finite() {
        return None;
    }
    Some(h.map(|row| row.map(|v| v / scale)))
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve_linear(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for column in 0..8 {
        let pivot =
            (column..8).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() < 1e-12 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        let pivot_row = a[column];
        for row in column + 1..8 {
            let factor = a[row][column] / pivot_row[column];
            for (value, pivot_value) in a[row].iter_mut().zip(pivot_row).skip(column) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.0; 8];
    for row in (0..8).rev() {
        let mut sum = b[row];
        for (k, value) in x.iter().enumerate().skip(row + 1) {
            sum -= a[row][k] * value;
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Median luminance ratio reference / target over the covered area.
fn photometric_gain(reference: &Plane, target: &Plane, warp: &Warp) -> Option<f32> {
    let mut ratios = Vec::new();
    let step = 16;
    let mut y = step;
    while y + step < reference.height {
        let mut x = step;
        while x + step < reference.width {
            let rx = (x as f32) * 2.0 + 0.5;
            let ry = (y as f32) * 2.0 + 0.5;
            if let Some(q) = warp.map(rx, ry)
                && let Some(t) = target.sample((q[0] - 0.5) / 2.0, (q[1] - 0.5) / 2.0)
            {
                let r = reference.at(x, y);
                // Planes hold ln(1 + 1000 L); invert to linear before the ratio.
                let lr = (r.exp() - 1.0) / 1000.0;
                let lt = (t.exp() - 1.0) / 1000.0;
                if lr > 0.01 && lt > 0.01 && lr < 0.95 && lt < 0.95 {
                    ratios.push(lr / lt);
                }
            }
            x += step;
        }
        y += step;
    }
    if ratios.len() < 16 {
        return None;
    }
    ratios.sort_by(f32::total_cmp);
    Some(ratios[ratios.len() / 2].clamp(0.25, 4.0))
}

/// Checkerboard composite of the reference luminance and the module warped
/// into the reference frame, for visual alignment checks: with a good warp the
/// tile boundaries are invisible; misalignment shows as broken edges. Both
/// planes are half-resolution log luminance; `tile` is the tile size in plane
/// pixels. Returns 16-bit grayscale samples and the plane size.
pub fn debug_checkerboard(
    reference: &Plane,
    target: &Plane,
    warp: &Warp,
    tile: usize,
) -> (Vec<u16>, usize, usize) {
    let tile = tile.max(1);
    let mut out = vec![0u16; reference.width * reference.height];
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &v in &reference.data {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let range = (hi - lo).max(1e-6);
    for y in 0..reference.height {
        for x in 0..reference.width {
            let use_target = ((x / tile) + (y / tile)) % 2 == 1;
            let value = if use_target {
                let rx = x as f32 * 2.0 + 0.5;
                let ry = y as f32 * 2.0 + 0.5;
                warp.map(rx, ry)
                    .and_then(|q| target.sample((q[0] - 0.5) / 2.0, (q[1] - 0.5) / 2.0))
            } else {
                Some(reference.at(x, y))
            };
            out[y * reference.width + x] = match value {
                Some(v) => (((v - lo) / range).clamp(0.0, 1.0) * 65535.0) as u16,
                None => 0,
            };
        }
    }
    (out, reference.width, reference.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_fit_recovers_a_known_transform() {
        let truth = [[1.02, 0.01, 5.0], [-0.015, 0.99, -3.0], [1e-6, -2e-6, 1.0]];
        let mut pairs = Vec::new();
        for y in (0..3000).step_by(250) {
            for x in (0..4000).step_by(250) {
                let a = [x as f64, y as f64];
                let b = apply_homography(&truth, a).unwrap();
                pairs.push((a, b, 0.9));
            }
        }
        // Plant gross outliers.
        for i in 0..20 {
            let a = pairs[i * 7].0;
            pairs.push((a, [a[0] + 80.0, a[1] - 120.0], 0.9));
        }
        let (h, inliers, residuals) = fit_homography_ransac(&pairs, 1.0).unwrap();
        assert!(inliers.len() >= 190, "{}", inliers.len());
        assert!(residuals[residuals.len() / 2] < 1e-3);
        for a in [[100.0, 100.0], [3900.0, 2900.0]] {
            let want = apply_homography(&truth, a).unwrap();
            let got = apply_homography(&h, a).unwrap();
            assert!((got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3);
        }
    }

    #[test]
    fn warp_grid_interpolates_and_reports_magnification() {
        let warp = Warp::from_fn(100, 60, 10, |p| Some([p[0] * 2.0 + 1.0, p[1] * 2.0 - 1.0]));
        assert_eq!((warp.columns, warp.rows), (11, 7));
        let q = warp.map(25.0, 13.0).unwrap();
        assert!((q[0] - 51.0).abs() < 1e-4 && (q[1] - 25.0).abs() < 1e-4);
        assert!((warp.magnification(20.0, 20.0).unwrap() - 2.0).abs() < 1e-4);
        let partial = Warp::from_fn(100, 60, 10, |p| (p[0] < 50.0).then_some(p));
        assert!(partial.map(10.0, 10.0).is_some());
        assert!(partial.map(90.0, 10.0).is_none());
    }
}
