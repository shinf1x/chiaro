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
//! quantiles) so alignment quality can be inspected after every export. Rig
//! calibration does not reuse the arbitrary grid-window centres above as 3-D
//! points: after the final warp is known, a separate sparse Shi-Tomasi/corner
//! matcher produces point observations with uniqueness, mutual-match and
//! localization-covariance diagnostics.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::depth::{DepthAlignmentReport, DepthOptions};
use crate::geometry::ResolvedCamera;
use crate::image::{Plane, match_patch, match_patch_at};
use crate::math::{Mat3, Vec2, apply_homography};

/// View-dependent visibility of one reference-space warp location.
///
/// Alignment-only warps cannot infer visibility and therefore remain
/// [`Unknown`](Self::Unknown). Dense calibrated depth refinement upgrades
/// locations to [`Visible`](Self::Visible), [`Occluded`](Self::Occluded), or
/// [`Boundary`](Self::Boundary). Downstream reconstruction must never bridge
/// `Occluded`/`Boundary` regions merely because a later image-space matcher
/// finds a plausible texture there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WarpVisibility {
    #[default]
    Unknown,
    Visible,
    Occluded,
    Boundary,
}

impl WarpVisibility {
    /// Whether this state is explicit geometric evidence that the source must
    /// not contribute the requested surface sample.
    #[inline]
    pub fn blocks_sampling(self) -> bool {
        matches!(self, Self::Occluded | Self::Boundary)
    }
}

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
    /// Categorical visibility carried separately from scalar confidence.
    /// Keeping the two distinct prevents a later local-registration stage
    /// from accidentally turning an explicitly occluded surface back into a
    /// usable source merely by raising confidence.
    pub visibility: Vec<WarpVisibility>,
}

/// All synthesis-facing values from one warp grid cell. Computing these
/// together avoids repeating the same grid coordinate, floor, and index work
/// for map/confidence/visibility queries at the same output location.
#[derive(Clone, Copy, Debug)]
pub struct WarpSample {
    pub mapped: Option<[f32; 2]>,
    pub confidence: f32,
    pub visibility: WarpVisibility,
}

#[derive(Clone, Copy)]
struct WarpCell {
    c0: usize,
    r0: usize,
    c1: usize,
    r1: usize,
    tx: f32,
    ty: f32,
}

impl Warp {
    #[inline]
    fn cell(&self, x: f32, y: f32) -> Option<WarpCell> {
        if self.step == 0 || self.columns == 0 || self.rows == 0 {
            return None;
        }
        let fx = x / self.step as f32;
        let fy = y / self.step as f32;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let c0 = (fx.floor() as usize).min(self.columns - 1);
        let r0 = (fy.floor() as usize).min(self.rows - 1);
        Some(WarpCell {
            c0,
            r0,
            c1: (c0 + 1).min(self.columns - 1),
            r1: (r0 + 1).min(self.rows - 1),
            tx: fx - c0 as f32,
            ty: fy - r0 as f32,
        })
    }
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
        let mut visibility = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for column in 0..columns {
                let p = [(column * step) as f64, (row * step) as f64];
                match map(p) {
                    Some(q) => {
                        points.push([q[0] as f32, q[1] as f32]);
                        confidence.push(1.0);
                        // A generic image-space/calibration warp says where a
                        // point maps, but not whether that scene surface is
                        // visible from the target optical centre.
                        visibility.push(WarpVisibility::Unknown);
                    }
                    None => {
                        points.push([f32::NAN, f32::NAN]);
                        confidence.push(0.0);
                        visibility.push(WarpVisibility::Unknown);
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
            visibility,
        }
    }

    /// Sample mapping, confidence, and visibility using one shared grid-cell lookup.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> WarpSample {
        let Some(cell) = self.cell(x, y) else {
            return WarpSample {
                mapped: None,
                confidence: 0.0,
                visibility: WarpVisibility::Unknown,
            };
        };
        let WarpCell {
            c0,
            r0,
            c1,
            r1,
            tx,
            ty,
        } = cell;
        let p = |c: usize, r: usize| self.points[r * self.columns + c];
        let (a, b, c, d) = (p(c0, r0), p(c1, r0), p(c0, r1), p(c1, r1));
        let mut mapped = [0.0f32; 2];
        for k in 0..2 {
            let top = a[k] * (1.0 - tx) + b[k] * tx;
            let bottom = c[k] * (1.0 - tx) + d[k] * tx;
            mapped[k] = top * (1.0 - ty) + bottom * ty;
        }
        let mapped = (!mapped[0].is_nan() && !mapped[1].is_nan()).then_some(mapped);

        let confidence = if self.confidence.len() == self.points.len() {
            let value = |column: usize, row: usize| self.confidence[row * self.columns + column];
            let top = value(c0, r0) * (1.0 - tx) + value(c1, r0) * tx;
            let bottom = value(c0, r1) * (1.0 - tx) + value(c1, r1) * tx;
            (top * (1.0 - ty) + bottom * ty).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let visibility = if self.visibility.len() == self.points.len() {
            let values = [
                self.visibility[r0 * self.columns + c0],
                self.visibility[r0 * self.columns + c1],
                self.visibility[r1 * self.columns + c0],
                self.visibility[r1 * self.columns + c1],
            ];
            if values.contains(&WarpVisibility::Boundary) {
                WarpVisibility::Boundary
            } else {
                let occluded = values.contains(&WarpVisibility::Occluded);
                let visible = values.contains(&WarpVisibility::Visible);
                let unknown = values.contains(&WarpVisibility::Unknown);
                if occluded {
                    if visible || unknown {
                        WarpVisibility::Boundary
                    } else {
                        WarpVisibility::Occluded
                    }
                } else if unknown {
                    WarpVisibility::Unknown
                } else {
                    WarpVisibility::Visible
                }
            }
        } else {
            WarpVisibility::Unknown
        };

        WarpSample {
            mapped,
            confidence,
            visibility,
        }
    }

    /// Target coordinates for a reference pixel; `None` where undefined.
    #[inline]
    pub fn map(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let cell = self.cell(x, y)?;
        let p = |c: usize, r: usize| self.points[r * self.columns + c];
        let (a, b, c, d) = (
            p(cell.c0, cell.r0),
            p(cell.c1, cell.r0),
            p(cell.c0, cell.r1),
            p(cell.c1, cell.r1),
        );
        let mut out = [0.0f32; 2];
        for k in 0..2 {
            let top = a[k] * (1.0 - cell.tx) + b[k] * cell.tx;
            let bottom = c[k] * (1.0 - cell.tx) + d[k] * cell.tx;
            out[k] = top * (1.0 - cell.ty) + bottom * cell.ty;
        }
        (!out[0].is_nan() && !out[1].is_nan()).then_some(out)
    }

    /// Analytic local Jacobian of the bilinear warp, but only when the full
    /// requested stencil remains inside the same grid cell. Callers that need
    /// exact behaviour across cell boundaries can fall back to finite
    /// differences there. Returning the two derivative columns avoids four
    /// repeated grid-cell lookups in the common interior case.
    #[inline]
    pub(crate) fn local_jacobian_same_cell(
        &self,
        x: f32,
        y: f32,
        stencil_radius: f32,
    ) -> Option<([f32; 2], [f32; 2])> {
        let cell = self.cell(x, y)?;
        let same_cell = |sample: WarpCell| {
            sample.c0 == cell.c0
                && sample.c1 == cell.c1
                && sample.r0 == cell.r0
                && sample.r1 == cell.r1
        };
        for (sx, sy) in [
            (x - stencil_radius, y),
            (x + stencil_radius, y),
            (x, y - stencil_radius),
            (x, y + stencil_radius),
        ] {
            let Some(sample) = self.cell(sx, sy) else {
                return None;
            };
            if !same_cell(sample) {
                return None;
            }
        }

        let p = |c: usize, r: usize| self.points[r * self.columns + c];
        let (a, b, c, d) = (
            p(cell.c0, cell.r0),
            p(cell.c1, cell.r0),
            p(cell.c0, cell.r1),
            p(cell.c1, cell.r1),
        );
        let inverse_step = (self.step as f32).recip();
        let mut dx = [0.0f32; 2];
        let mut dy = [0.0f32; 2];
        for k in 0..2 {
            dx[k] = ((b[k] - a[k]) * (1.0 - cell.ty) + (d[k] - c[k]) * cell.ty) * inverse_step;
            dy[k] = ((c[k] - a[k]) * (1.0 - cell.tx) + (d[k] - b[k]) * cell.tx) * inverse_step;
        }
        dx.into_iter()
            .chain(dy)
            .all(f32::is_finite)
            .then_some((dx, dy))
    }

    /// Bilinearly interpolated local confidence for synthesis.
    #[inline]
    pub fn confidence(&self, x: f32, y: f32) -> f32 {
        let Some(cell) = self.cell(x, y) else {
            return 0.0;
        };
        if self.confidence.len() != self.points.len() {
            return 0.0;
        }
        let value = |column: usize, row: usize| self.confidence[row * self.columns + column];
        let top = value(cell.c0, cell.r0) * (1.0 - cell.tx) + value(cell.c1, cell.r0) * cell.tx;
        let bottom = value(cell.c0, cell.r1) * (1.0 - cell.tx) + value(cell.c1, cell.r1) * cell.tx;
        (top * (1.0 - cell.ty) + bottom * cell.ty).clamp(0.0, 1.0)
    }

    /// Conservative categorical visibility at a reference-space position.
    #[inline]
    pub fn visibility(&self, x: f32, y: f32) -> WarpVisibility {
        let Some(cell) = self.cell(x, y) else {
            return WarpVisibility::Unknown;
        };
        if self.visibility.len() != self.points.len() {
            return WarpVisibility::Unknown;
        }
        let values = [
            self.visibility[cell.r0 * self.columns + cell.c0],
            self.visibility[cell.r0 * self.columns + cell.c1],
            self.visibility[cell.r1 * self.columns + cell.c0],
            self.visibility[cell.r1 * self.columns + cell.c1],
        ];
        if values.contains(&WarpVisibility::Boundary) {
            return WarpVisibility::Boundary;
        }
        let occluded = values.contains(&WarpVisibility::Occluded);
        let visible = values.contains(&WarpVisibility::Visible);
        let unknown = values.contains(&WarpVisibility::Unknown);
        if occluded {
            if visible || unknown {
                WarpVisibility::Boundary
            } else {
                WarpVisibility::Occluded
            }
        } else if unknown {
            WarpVisibility::Unknown
        } else {
            WarpVisibility::Visible
        }
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

impl ModuleAlignment {
    /// Downstream geometric admission. Once a physical depth warp exists, the
    /// old single-homography acceptance bit is no longer authoritative: a
    /// camera may be globally non-homographic yet contain a coherent visible
    /// subset under the calibrated 3-D rig. The 24-node floor matches the
    /// minimum coherent direct-depth component used by the depth stage.
    pub fn geometry_accepted(&self) -> bool {
        self.report
            .geometry_accepted
            .unwrap_or(self.report.accepted)
    }
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
    /// Unit-determinant localization covariance of the reference feature in
    /// reference sensor axes. This is derived from the local structure tensor:
    /// corners are close to isotropic while edge-like observations are
    /// anisotropic and therefore receive less weight along the weak direction.
    pub reference_localization_covariance: [[f32; 2]; 2],
    /// Unit-determinant localization covariance of the target feature in
    /// target sensor axes after propagating the rendered-space structure
    /// tensor through the current geometric proposal.
    pub target_localization_covariance: [[f32; 2]; 2],
    /// Separation between the winning NCC peak and the best non-neighbouring
    /// peak. Small values indicate repeated structure or a broad edge ridge.
    pub peak_margin: f32,
    /// Forward/backward sparse-match closure error, in reference luminance
    /// plane pixels (one plane pixel is two sensor pixels).
    pub forward_backward_error_px: f32,
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
    /// Factory mirror-angle mapping point in the parent reference raster.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_optical_center_prior_reference_px: Option<Vec2>,
    /// Translation applied to the factory reference-to-target warp so that
    /// the mapped point lands on the target optical axis. Image matching is
    /// subsequently free to override this initializer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_optical_center_prior_shift_target_px: Option<Vec2>,
    /// Whether the mapping was applied. `None` means no applicable mapping
    /// existed; the mapped initializer is not raced against a legacy seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_optical_center_prior_selected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle_optical_center_prior_quality: Option<f32>,
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
    /// Sparse point-feature proposals considered for physical rig fitting.
    /// These are deliberately separate from the dense/grid warp patches.
    pub rig_feature_candidates: usize,
    /// Sparse point-feature matches that passed 2-D corner conditioning, NCC
    /// peak uniqueness, and mutual forward/backward verification.
    pub rig_feature_matches: usize,
    pub rig_feature_rejected_ambiguous: usize,
    pub rig_feature_rejected_forward_backward: usize,
    /// Local calibrated inverse-depth refinement, when both camera models were
    /// available and depth-aware alignment was enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<DepthAlignmentReport>,
    /// Whether the legacy global image-space alignment passed its own
    /// homography-consensus gate. Once physical depth is available this is a
    /// diagnostic, not the downstream source-admission authority.
    pub accepted: bool,
    /// Physical downstream admission after joint depth/visibility, when that
    /// path ran. `None` means the compatibility path still uses `accepted`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_accepted: Option<bool>,
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
#[derive(Clone, Copy)]
pub struct AlignInput<'a> {
    pub name: &'a str,
    /// Half-resolution log luminance of the module.
    pub luminance: &'a Plane,
    pub width: usize,
    pub height: usize,
    /// Resolved camera model, if calibration is available.
    pub camera: Option<&'a ResolvedCamera>,
    /// Whether this image may contribute photometric evidence to dense depth.
    /// Held-out CFA validation disables this while still retaining the camera
    /// model for projection/evaluation, preventing target-radiance leakage.
    pub depth_evidence_enabled: bool,
    /// Multiplicative evidence prior. V8 keeps this neutral (`1.0`) for every
    /// enabled camera: sparse rig difficulty is not a camera-wide quality
    /// judgement. Local match uniqueness/geometry decide authority instead.
    /// Held-out validation sets it to zero together with
    /// `depth_evidence_enabled = false`.
    pub depth_evidence_reliability: f32,
    /// Optional factory optical-axis point expressed in this alignment's
    /// reference raster. It is a bootstrap prior only.
    pub angle_optical_center_prior_reference_px: Option<Vec2>,
    /// Nominal focal length in pixels (used when `camera` is `None`).
    pub nominal_focal_px: f64,
}

#[derive(Clone, Copy)]
pub struct AlignmentSeed<'a> {
    pub warp: &'a Warp,
    pub name: &'static str,
}

/// Reusable luminance pyramid for repeated alignment passes. Building down to
/// the smallest alignment scale once lets each call borrow the prefix it
/// needs instead of repeatedly cloning/downsampling the same module.
#[derive(Clone, Debug)]
pub struct AlignPyramidCache {
    levels: Vec<Plane>,
    // Sparse rig features are detected only on the reference luminance plane.
    // The same reference cache is reused for every target camera, so memoising
    // them here avoids re-running the structure-tensor scan ten times.
    rig_corners: OnceLock<Vec<RigCorner>>,
}

impl AlignPyramidCache {
    pub fn new(luminance: &Plane) -> Self {
        Self {
            levels: luminance.pyramid(16),
            rig_corners: OnceLock::new(),
        }
    }

    fn rig_corners(&self) -> &[RigCorner] {
        self.rig_corners
            .get_or_init(|| detect_rig_corners(&self.levels[0]))
    }

    fn levels_for(&self, min_size: usize) -> &[Plane] {
        let minimum = min_size.max(8);
        let count = self
            .levels
            .iter()
            .take_while(|plane| plane.width.min(plane.height) >= minimum)
            .count()
            .max(1);
        &self.levels[..count.min(self.levels.len())]
    }
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
    let reference_pyramid = AlignPyramidCache::new(reference.luminance);
    let target_pyramid = AlignPyramidCache::new(target.luminance);
    align_module_seeded_cached(
        reference,
        target,
        options,
        seed,
        &reference_pyramid,
        &target_pyramid,
    )
}

pub fn align_module_seeded_cached(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    options: &AlignOptions,
    seed: Option<AlignmentSeed<'_>>,
    reference_pyramid: &AlignPyramidCache,
    target_pyramid: &AlignPyramidCache,
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
            let mut grid = Warp::from_fn(width, height, 8, |p| {
                target_camera
                    .map_from(reference_camera, p, FAR_DEPTH)
                    .filter(|q| q[0].is_finite() && q[1].is_finite())
            });
            if let Some(reference_pixel) = target.angle_optical_center_prior_reference_px {
                let Some(predicted) =
                    grid.map(reference_pixel[0] as f32, reference_pixel[1] as f32)
                else {
                    bail!(
                        "{} angle optical-center mapping point {:.3},{:.3} is outside the usable {} reference warp",
                        target.name,
                        reference_pixel[0],
                        reference_pixel[1],
                        reference.name,
                    );
                };
                let optical_axis = target_camera.optical_axis_pixel();
                let shift = [
                    optical_axis[0] - f64::from(predicted[0]),
                    optical_axis[1] - f64::from(predicted[1]),
                ];
                if !shift.iter().all(|value| value.is_finite()) {
                    bail!(
                        "{} angle optical-center mapping produced a non-finite target shift",
                        target.name,
                    );
                }
                for point in &mut grid.points {
                    if point[0].is_finite() && point[1].is_finite() {
                        point[0] += shift[0] as f32;
                        point[1] += shift[1] as f32;
                    }
                }
                report.initialised_from = "calibration + angle optical-center mapping";
                report.angle_optical_center_prior_reference_px = Some(reference_pixel);
                report.angle_optical_center_prior_shift_target_px = Some(shift);
            } else {
                report.initialised_from = "calibration";
            }
            grid
        }
        (None, _, _) => {
            report.initialised_from = "nominal focal group";
            let scale = target.nominal_focal_px / reference.nominal_focal_px;
            let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
            let (tx, ty) = (
                (target.width as f64 - 1.0) / 2.0,
                (target.height as f64 - 1.0) / 2.0,
            );
            let shift = coarse_global_shift(
                reference,
                target,
                scale,
                options,
                reference_pyramid,
                target_pyramid,
            )?;
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
    let mut alignment_magnification = target.nominal_focal_px / reference.nominal_focal_px;
    if options.refine {
        let reference_pyramid = reference_pyramid.levels_for(96);
        let target_pyramid = target_pyramid.levels_for(48);
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
        alignment_magnification = magnification;
        let mut acquired_correction = false;
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
            // A narrow residual search is valid only after some pyramid level
            // has actually established a homography. Narrow-FOV cameras can
            // have fewer than the six required patches at the coarsest level;
            // in that case retain the bootstrap radius at finer levels instead
            // of silently stranding an otherwise useful factory prior.
            let radius = if !acquired_correction {
                options.coarse_radius
            } else {
                3
            };
            let patch = options.patch.min(reference_plane.width / 4).max(8);
            let mut pairs = Vec::new();
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
                    }
                    x += stride;
                }
                y += stride;
            }
            // The grid patches above exist only to estimate the image warp. Rig
            // correspondences are generated separately from true point-like
            // features after the final homography has been fitted; an arbitrary
            // grid-window centre is not a valid 3-D observation.

            let threshold = (options.inlier_px * scale as f32 * 2.0).max(options.inlier_px);
            if let Some((update, inliers, residuals)) = fit_homography_ransac(&pairs, threshold) {
                correction = crate::math::mul(&correction, &update);
                acquired_correction = true;
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

    if options.refine {
        // Re-render at the native half-resolution luminance density through the
        // *final* 2-D alignment proposal. This image is only a search frame:
        // sparse correspondences remain explicit reference/target sensor
        // pixels and are independently verified by the rig stage.
        let rendered = render_through(
            reference.luminance,
            1,
            target.luminance,
            1.0,
            &initial,
            &correction,
        );
        correspondences = rig_feature_correspondences(
            reference.luminance,
            &rendered,
            reference_pyramid.rig_corners(),
            &initial,
            &correction,
            alignment_magnification as f32,
            options.min_score,
            false,
            &mut report,
        );
        if correspondences.len() < RIG_FEATURE_RETRY_BELOW_MATCHES
            && alignment_magnification >= RIG_FEATURE_RETRY_MIN_MAGNIFICATION
        {
            let mut retry_report = AlignmentReport::default();
            let retry = rig_feature_correspondences(
                reference.luminance,
                &rendered,
                reference_pyramid.rig_corners(),
                &initial,
                &correction,
                alignment_magnification as f32,
                options.min_score,
                true,
                &mut retry_report,
            );
            if retry.len() > correspondences.len() {
                correspondences = retry;
                report.rig_feature_candidates = retry_report.rig_feature_candidates;
                report.rig_feature_matches = retry_report.rig_feature_matches;
                report.rig_feature_rejected_ambiguous = retry_report.rig_feature_rejected_ambiguous;
                report.rig_feature_rejected_forward_backward =
                    retry_report.rig_feature_rejected_forward_backward;
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

// Sparse point-feature matching used only by capture-specific rig fitting.
// The normal alignment warp deliberately remains grid/NCC based: a broad
// window is excellent for estimating a local image translation but its
// arbitrary window centre is not necessarily a physical 3-D point. The rig
// matcher below instead anchors every observation on a genuine 2-D corner.
const RIG_FEATURE_TENSOR_RADIUS: usize = 2;
const RIG_FEATURE_SCAN_STEP: usize = 2;
const RIG_FEATURE_NMS_RADIUS: usize = 3;
const RIG_FEATURE_MAX_CANDIDATES: usize = 60_000;
const RIG_FEATURE_PAIR_MAX_CANDIDATES: usize = 15_000;
const RIG_FEATURE_MAX_MATCHES: usize = 15_000;
const RIG_FEATURE_PATCH: usize = 13;
const RIG_FEATURE_SEARCH_RADIUS: usize = 8;
const RIG_FEATURE_MIN_EIGEN_RATIO: f32 = 0.08;
const RIG_FEATURE_MIN_STRUCTURE: f32 = 0.010;
const RIG_FEATURE_MIN_SCORE: f32 = 0.65;
const RIG_FEATURE_MIN_PEAK_MARGIN: f32 = 0.025;
const RIG_FEATURE_MAX_FORWARD_BACKWARD_ERROR: f32 = 0.50;
const RIG_FEATURE_MAX_CORNER_DISAGREEMENT: f32 = 0.75;
// Narrow-FOV modules can have excellent geometric overlap but noticeably
// different local appearance after resampling.  A strict patch gate can then
// leave only a dozen correspondences (C3 on the real 75/150-mm capture),
// which is much worse than carrying a larger, noisier population into the
// calibrated epipolar RANSAC.  Retry only sparse high-magnification cases
// with a wider/high-recall search; RANSAC remains the geometric authority.
const RIG_FEATURE_RETRY_BELOW_MATCHES: usize = 1_500;
const RIG_FEATURE_RETRY_MIN_MAGNIFICATION: f64 = 1.35;
const RIG_FEATURE_RELAXED_SEARCH_RADIUS: usize = 20;
const RIG_FEATURE_RELAXED_MIN_SCORE: f32 = 0.54;
const RIG_FEATURE_RELAXED_MIN_PEAK_MARGIN: f32 = 0.010;
const RIG_FEATURE_RELAXED_MAX_FORWARD_BACKWARD_ERROR: f32 = 1.25;
const RIG_FEATURE_RELAXED_MAX_CORNER_DISAGREEMENT: f32 = 1.50;
const RIG_FEATURE_RELAXED_MAX_MATCHES: usize = 15_000;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RigCorner {
    pub(crate) integer: [usize; 2],
    pub(crate) subpixel: [f32; 2],
    pub(crate) score: f32,
    pub(crate) structure: f32,
    pub(crate) covariance: [[f32; 2]; 2],
}

/// Shi-Tomasi structure tensor at one luminance-plane pixel. The returned
/// covariance is proportional to `tensor^-1` and normalized to determinant 1;
/// absolute uncertainty continues to come from NCC/contrast in the rig loss.
fn rig_corner_metric(plane: &Plane, x: usize, y: usize) -> Option<(f32, f32, [[f32; 2]; 2])> {
    let radius = RIG_FEATURE_TENSOR_RADIUS;
    if x <= radius || y <= radius || x + radius + 1 >= plane.width || y + radius + 1 >= plane.height
    {
        return None;
    }
    let mut xx = 0.0f64;
    let mut xy = 0.0f64;
    let mut yy = 0.0f64;
    let mut samples = 0usize;
    for sy in y - radius..=y + radius {
        for sx in x - radius..=x + radius {
            let left = plane.at(sx - 1, sy);
            let right = plane.at(sx + 1, sy);
            let above = plane.at(sx, sy - 1);
            let below = plane.at(sx, sy + 1);
            if !left.is_finite() || !right.is_finite() || !above.is_finite() || !below.is_finite() {
                return None;
            }
            let gx = 0.5 * f64::from(right - left);
            let gy = 0.5 * f64::from(below - above);
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
            samples += 1;
        }
    }
    if samples == 0 {
        return None;
    }
    let inv_n = 1.0 / samples as f64;
    xx *= inv_n;
    xy *= inv_n;
    yy *= inv_n;
    let trace = xx + yy;
    let discriminant = ((xx - yy) * (xx - yy) + 4.0 * xy * xy).max(0.0).sqrt();
    let lambda_min = 0.5 * (trace - discriminant);
    let lambda_max = 0.5 * (trace + discriminant);
    if !lambda_min.is_finite()
        || !lambda_max.is_finite()
        || lambda_min <= 1.0e-10
        || lambda_max <= 0.0
    {
        return None;
    }
    let ratio = (lambda_min / lambda_max) as f32;
    let determinant = xx * yy - xy * xy;
    if !determinant.is_finite() || determinant <= 1.0e-16 {
        return None;
    }
    // inv(T) * sqrt(det(T)) has unit determinant. Clamp entries only through
    // the eigen-ratio admission above, not per-axis, so orientation is kept.
    let scale = determinant.sqrt();
    let covariance = [
        [(yy / scale) as f32, (-xy / scale) as f32],
        [(-xy / scale) as f32, (xx / scale) as f32],
    ];
    Some((lambda_min as f32, ratio, covariance))
}

fn parabolic_corner_offset(minus: f32, centre: f32, plus: f32) -> f32 {
    let denominator = minus - 2.0 * centre + plus;
    if !minus.is_finite() || !plus.is_finite() || denominator.abs() < 1.0e-12 {
        0.0
    } else {
        (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
    }
}

pub(crate) fn refine_rig_corner(plane: &Plane, seed_x: usize, seed_y: usize) -> Option<RigCorner> {
    let half = RIG_FEATURE_PATCH / 2;
    if seed_x <= half + 2
        || seed_y <= half + 2
        || seed_x + half + 2 >= plane.width
        || seed_y + half + 2 >= plane.height
    {
        return None;
    }

    // The detector scan may be strided. Search the immediate 3x3 response
    // neighbourhood before sub-pixel interpolation so an actual corner on an
    // unscanned integer pixel is not biased by up to one luminance-plane pixel.
    let mut best: Option<RigCorner> = None;
    for y in seed_y - 1..=seed_y + 1 {
        for x in seed_x - 1..=seed_x + 1 {
            let Some((score, ratio, covariance)) = rig_corner_metric(plane, x, y) else {
                continue;
            };
            if ratio < RIG_FEATURE_MIN_EIGEN_RATIO {
                continue;
            }
            let structure = plane.window_std(x - half, y - half, RIG_FEATURE_PATCH);
            if structure < RIG_FEATURE_MIN_STRUCTURE {
                continue;
            }
            if best.as_ref().is_none_or(|current| score > current.score) {
                best = Some(RigCorner {
                    integer: [x, y],
                    subpixel: [x as f32, y as f32],
                    score,
                    structure,
                    covariance,
                });
            }
        }
    }
    let mut corner = best?;
    let [x, y] = corner.integer;
    let sx_minus = rig_corner_metric(plane, x - 1, y).map_or(corner.score, |v| v.0);
    let sx_plus = rig_corner_metric(plane, x + 1, y).map_or(corner.score, |v| v.0);
    let sy_minus = rig_corner_metric(plane, x, y - 1).map_or(corner.score, |v| v.0);
    let sy_plus = rig_corner_metric(plane, x, y + 1).map_or(corner.score, |v| v.0);
    corner.subpixel = [
        x as f32 + parabolic_corner_offset(sx_minus, corner.score, sx_plus),
        y as f32 + parabolic_corner_offset(sy_minus, corner.score, sy_plus),
    ];
    Some(corner)
}

pub(crate) fn detect_rig_corners(reference: &Plane) -> Vec<RigCorner> {
    let half = RIG_FEATURE_PATCH / 2;
    let margin = half + RIG_FEATURE_SEARCH_RADIUS + RIG_FEATURE_TENSOR_RADIUS + 2;
    if reference.width <= 2 * margin || reference.height <= 2 * margin {
        return Vec::new();
    }
    let mut candidates = Vec::<RigCorner>::new();
    let mut y = margin;
    while y + margin < reference.height {
        let mut x = margin;
        while x + margin < reference.width {
            if let Some((score, ratio, covariance)) = rig_corner_metric(reference, x, y)
                && ratio >= RIG_FEATURE_MIN_EIGEN_RATIO
            {
                let structure = reference.window_std(x - half, y - half, RIG_FEATURE_PATCH);
                if structure >= RIG_FEATURE_MIN_STRUCTURE {
                    candidates.push(RigCorner {
                        integer: [x, y],
                        subpixel: [x as f32, y as f32],
                        score,
                        structure,
                        covariance,
                    });
                }
            }
            x += RIG_FEATURE_SCAN_STEP;
        }
        y += RIG_FEATURE_SCAN_STEP;
    }
    candidates.sort_by(|first, second| second.score.total_cmp(&first.score));

    // Greedy non-maximum suppression after response sorting. A compact byte
    // mask is faster than O(N^2) distance checks and keeps features spatially
    // distributed instead of allowing one railing junction to dominate.
    let mut blocked = vec![false; reference.width * reference.height];
    let mut selected = Vec::with_capacity(RIG_FEATURE_MAX_CANDIDATES);
    for candidate in candidates {
        let Some(candidate) =
            refine_rig_corner(reference, candidate.integer[0], candidate.integer[1])
        else {
            continue;
        };
        let [x, y] = candidate.integer;
        if blocked[y * reference.width + x] {
            continue;
        }
        selected.push(candidate);
        if selected.len() >= RIG_FEATURE_MAX_CANDIDATES {
            break;
        }
        let x0 = x.saturating_sub(RIG_FEATURE_NMS_RADIUS);
        let y0 = y.saturating_sub(RIG_FEATURE_NMS_RADIUS);
        let x1 = (x + RIG_FEATURE_NMS_RADIUS).min(reference.width - 1);
        let y1 = (y + RIG_FEATURE_NMS_RADIUS).min(reference.height - 1);
        for by in y0..=y1 {
            blocked[by * reference.width + x0..=by * reference.width + x1].fill(true);
        }
    }
    selected
}

fn normalize_covariance(covariance: [[f64; 2]; 2]) -> [[f32; 2]; 2] {
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    if !determinant.is_finite() || determinant <= 1.0e-12 {
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let scale = determinant.sqrt();
    [
        [
            (covariance[0][0] / scale) as f32,
            (covariance[0][1] / scale) as f32,
        ],
        [
            (covariance[1][0] / scale) as f32,
            (covariance[1][1] / scale) as f32,
        ],
    ]
}

fn propagate_rig_covariance(
    covariance: [[f32; 2]; 2],
    point: [f32; 2],
    map: &dyn Fn([f32; 2]) -> Option<Vec2>,
) -> [[f32; 2]; 2] {
    let h = 0.5f32;
    let Some(left) = map([point[0] - h, point[1]]) else {
        return covariance;
    };
    let Some(right) = map([point[0] + h, point[1]]) else {
        return covariance;
    };
    let Some(above) = map([point[0], point[1] - h]) else {
        return covariance;
    };
    let Some(below) = map([point[0], point[1] + h]) else {
        return covariance;
    };
    let j = [
        [
            (right[0] - left[0]) / f64::from(2.0 * h),
            (below[0] - above[0]) / f64::from(2.0 * h),
        ],
        [
            (right[1] - left[1]) / f64::from(2.0 * h),
            (below[1] - above[1]) / f64::from(2.0 * h),
        ],
    ];
    let c = [
        [f64::from(covariance[0][0]), f64::from(covariance[0][1])],
        [f64::from(covariance[1][0]), f64::from(covariance[1][1])],
    ];
    let jc = [
        [
            j[0][0] * c[0][0] + j[0][1] * c[1][0],
            j[0][0] * c[0][1] + j[0][1] * c[1][1],
        ],
        [
            j[1][0] * c[0][0] + j[1][1] * c[1][0],
            j[1][0] * c[0][1] + j[1][1] * c[1][1],
        ],
    ];
    normalize_covariance([
        [
            jc[0][0] * j[0][0] + jc[0][1] * j[0][1],
            jc[0][0] * j[1][0] + jc[0][1] * j[1][1],
        ],
        [
            jc[1][0] * j[0][0] + jc[1][1] * j[0][1],
            jc[1][0] * j[1][0] + jc[1][1] * j[1][1],
        ],
    ])
}

fn rig_feature_correspondences(
    reference: &Plane,
    rendered: &Plane,
    corners: &[RigCorner],
    initial: &dyn Fn(Vec2) -> Option<Vec2>,
    correction: &Mat3,
    local_scale: f32,
    minimum_alignment_score: f32,
    relaxed: bool,
    report: &mut AlignmentReport,
) -> Vec<AlignmentCorrespondence> {
    // `corners` is a large global reference reservoir. Count only candidates
    // that actually lie in this target camera's rendered overlap so narrow-FOV
    // cameras can receive the same 15k *pair-specific* feature budget as wide
    // cameras instead of inheriting roughly overlap_fraction * 15k points.
    report.rig_feature_candidates = 0;
    report.rig_feature_matches = 0;
    report.rig_feature_rejected_ambiguous = 0;
    report.rig_feature_rejected_forward_backward = 0;
    let half = RIG_FEATURE_PATCH / 2;
    let search_radius = if relaxed {
        RIG_FEATURE_RELAXED_SEARCH_RADIUS
    } else {
        RIG_FEATURE_SEARCH_RADIUS
    };
    let minimum_score = if relaxed {
        (minimum_alignment_score - 0.08).max(RIG_FEATURE_RELAXED_MIN_SCORE)
    } else {
        minimum_alignment_score.max(RIG_FEATURE_MIN_SCORE)
    };
    let minimum_peak_margin = if relaxed {
        RIG_FEATURE_RELAXED_MIN_PEAK_MARGIN
    } else {
        RIG_FEATURE_MIN_PEAK_MARGIN
    };
    let maximum_forward_backward_error = if relaxed {
        RIG_FEATURE_RELAXED_MAX_FORWARD_BACKWARD_ERROR
    } else {
        RIG_FEATURE_MAX_FORWARD_BACKWARD_ERROR
    };
    let maximum_corner_disagreement = if relaxed {
        RIG_FEATURE_RELAXED_MAX_CORNER_DISAGREEMENT
    } else {
        RIG_FEATURE_MAX_CORNER_DISAGREEMENT
    };
    let maximum_matches = if relaxed {
        RIG_FEATURE_RELAXED_MAX_MATCHES
    } else {
        RIG_FEATURE_MAX_MATCHES
    };
    let map_rendered = |point: [f32; 2]| -> Option<Vec2> {
        let raster = [
            2.0 * f64::from(point[0]) + 0.5,
            2.0 * f64::from(point[1]) + 0.5,
        ];
        initial(apply_homography(correction, raster)?)
    };

    let mut matches = Vec::new();
    for &corner in corners {
        if matches.len() >= maximum_matches {
            break;
        }
        let [x, y] = corner.integer;
        let rx = x - half;
        let ry = y - half;
        let search_x = rx - search_radius;
        let search_y = ry - search_radius;
        let search_size = RIG_FEATURE_PATCH + 2 * search_radius;
        if !rendered_covered(rendered, search_x, search_y, search_size) {
            continue;
        }
        if report.rig_feature_candidates >= RIG_FEATURE_PAIR_MAX_CANDIDATES {
            break;
        }
        report.rig_feature_candidates += 1;
        let Some(forward) = match_patch_at(
            reference,
            rendered,
            rx,
            ry,
            rx,
            ry,
            RIG_FEATURE_PATCH,
            search_radius,
        ) else {
            continue;
        };
        let search_limit = search_radius as f32 - 0.25;
        if forward.score < minimum_score
            || forward.peak_margin < minimum_peak_margin
            || forward.shift[0].abs() >= search_limit
            || forward.shift[1].abs() >= search_limit
        {
            // A winner on the search boundary is not a localized point: the
            // true peak may lie outside the residual window. Treat it exactly
            // like an ambiguous/repeated peak rather than feeding a clipped
            // displacement to triangulation.
            report.rig_feature_rejected_ambiguous += 1;
            continue;
        }

        let target_centre = [x as f32 + forward.shift[0], y as f32 + forward.shift[1]];
        let target_integer = [
            target_centre[0].round() as isize,
            target_centre[1].round() as isize,
        ];
        if target_integer[0] < (half + search_radius) as isize
            || target_integer[1] < (half + search_radius) as isize
            || target_integer[0] + (half + search_radius) as isize >= rendered.width as isize
            || target_integer[1] + (half + search_radius) as isize >= rendered.height as isize
        {
            continue;
        }
        let Some(target_corner) = refine_rig_corner(
            rendered,
            target_integer[0] as usize,
            target_integer[1] as usize,
        ) else {
            report.rig_feature_rejected_ambiguous += 1;
            continue;
        };
        let corner_delta = [
            target_corner.subpixel[0] - (corner.subpixel[0] + forward.shift[0]),
            target_corner.subpixel[1] - (corner.subpixel[1] + forward.shift[1]),
        ];
        let corner_disagreement =
            (corner_delta[0] * corner_delta[0] + corner_delta[1] * corner_delta[1]).sqrt();
        if corner_disagreement > maximum_corner_disagreement {
            report.rig_feature_rejected_ambiguous += 1;
            continue;
        }

        let tx = target_corner.integer[0] - half;
        let ty = target_corner.integer[1] - half;
        let Some(backward) = match_patch_at(
            rendered,
            reference,
            tx,
            ty,
            rx,
            ry,
            RIG_FEATURE_PATCH,
            search_radius,
        ) else {
            continue;
        };
        if backward.score < minimum_score
            || backward.peak_margin < minimum_peak_margin
            || backward.shift[0].abs() >= search_limit
            || backward.shift[1].abs() >= search_limit
        {
            report.rig_feature_rejected_ambiguous += 1;
            continue;
        }
        // Close the cycle on independently localized point positions. The
        // backward patch is centred at target_corner.integer; preserve the
        // target corner's fractional offset when predicting its location back
        // in the reference image.
        let target_fraction = [
            target_corner.subpixel[0] - target_corner.integer[0] as f32,
            target_corner.subpixel[1] - target_corner.integer[1] as f32,
        ];
        let backward_reference = [
            x as f32 + backward.shift[0] + target_fraction[0],
            y as f32 + backward.shift[1] + target_fraction[1],
        ];
        let closure = [
            backward_reference[0] - corner.subpixel[0],
            backward_reference[1] - corner.subpixel[1],
        ];
        let forward_backward_error = (closure[0] * closure[0] + closure[1] * closure[1]).sqrt();
        if forward_backward_error > maximum_forward_backward_error {
            report.rig_feature_rejected_forward_backward += 1;
            continue;
        }

        // Both observations are now independently localized corner extrema;
        // NCC establishes their identity rather than defining an arbitrary
        // window-centre point.
        let target_feature = target_corner.subpixel;
        let reference_pixel = [
            2.0 * f64::from(corner.subpixel[0]) + 0.5,
            2.0 * f64::from(corner.subpixel[1]) + 0.5,
        ];
        let Some(target_pixel) = map_rendered(target_feature) else {
            continue;
        };
        let target_covariance =
            propagate_rig_covariance(target_corner.covariance, target_feature, &map_rendered);
        matches.push(AlignmentCorrespondence {
            reference_pixel,
            target_pixel,
            confidence: forward.score.min(backward.score),
            local_scale,
            structure: corner.structure.min(target_corner.structure),
            reference_localization_covariance: corner.covariance,
            target_localization_covariance: target_covariance,
            peak_margin: forward.peak_margin.min(backward.peak_margin),
            forward_backward_error_px: forward_backward_error,
            depth_reliability: None,
        });
    }
    report.rig_feature_matches = matches.len();
    matches
}

/// Exhaustive translation search at a coarse level for the no-calibration
/// fallback. Returns the shift in target raster pixels.
fn coarse_global_shift(
    _reference: &AlignInput<'_>,
    _target: &AlignInput<'_>,
    scale: f64,
    options: &AlignOptions,
    reference_pyramid: &AlignPyramidCache,
    target_pyramid: &AlignPyramidCache,
) -> Result<Vec2> {
    // Downsample the reference to ~64 px wide and the target to the same
    // angular density, then slide the smaller over the larger.
    let reference_pyramid = reference_pyramid.levels_for(32);
    let reference_plane = reference_pyramid.last().unwrap();
    let reference_scale = (1usize << (reference_pyramid.len() - 1)) as f64;
    let target_pyramid = target_pyramid.levels_for(16);
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
                let sample = warp.sample(rx, ry);
                if sample.visibility.blocks_sampling() || sample.confidence <= 0.0 {
                    None
                } else {
                    sample
                        .mapped
                        .and_then(|q| target.sample((q[0] - 0.5) / 2.0, (q[1] - 0.5) / 2.0))
                }
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
    fn rig_corner_metric_rejects_one_dimensional_edges_but_accepts_corners() {
        let mut plane = Plane::new(96, 96);
        for y in 0..96 {
            for x in 0..96 {
                plane.data[y * 96 + x] = if x >= 48 && y >= 48 { 1.0 } else { 0.0 };
            }
        }
        let (_, corner_ratio, _) = rig_corner_metric(&plane, 48, 48).unwrap();
        let edge_ratio = rig_corner_metric(&plane, 48, 70).map_or(0.0, |value| value.1);
        assert!(
            corner_ratio > RIG_FEATURE_MIN_EIGEN_RATIO,
            "corner eigen-ratio {corner_ratio}"
        );
        assert!(
            edge_ratio < RIG_FEATURE_MIN_EIGEN_RATIO,
            "edge eigen-ratio {edge_ratio}"
        );
    }

    #[test]
    fn sparse_rig_features_close_forward_backward_on_known_translation() {
        let (width, height) = (180usize, 140usize);
        let mut reference = Plane::new(width, height);
        // Deterministic non-periodic texture produces many true 2-D corners
        // and unambiguous small patches without relying on an RNG crate.
        let mut state = 0xA5C3_19D7u32;
        for value in &mut reference.data {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *value = ((state >> 8) as f32) / ((1u32 << 24) as f32);
        }
        let shift = [3isize, -2isize];
        let mut rendered = Plane::new(width, height);
        rendered.data.fill(f32::NAN);
        for y in 0..height {
            for x in 0..width {
                let sx = x as isize - shift[0];
                let sy = y as isize - shift[1];
                if sx >= 0 && sy >= 0 && sx < width as isize && sy < height as isize {
                    rendered.data[y * width + x] = reference.at(sx as usize, sy as usize);
                }
            }
        }
        let initial = |point: Vec2| Some(point);
        let mut report = AlignmentReport::default();
        let corners = detect_rig_corners(&reference);
        let matches = rig_feature_correspondences(
            &reference,
            &rendered,
            &corners,
            &initial,
            &crate::math::IDENTITY,
            1.0,
            0.5,
            false,
            &mut report,
        );
        assert!(matches.len() >= 24, "only {} sparse matches", matches.len());
        let mut errors = matches
            .iter()
            .map(|correspondence| {
                let dx = correspondence.target_pixel[0]
                    - correspondence.reference_pixel[0]
                    - 2.0 * shift[0] as f64;
                let dy = correspondence.target_pixel[1]
                    - correspondence.reference_pixel[1]
                    - 2.0 * shift[1] as f64;
                (dx * dx + dy * dy).sqrt()
            })
            .collect::<Vec<_>>();
        errors.sort_by(f64::total_cmp);
        assert!(
            errors[errors.len() / 2] < 0.35,
            "median error {}",
            errors[errors.len() / 2]
        );
        assert!(
            matches
                .iter()
                .all(|correspondence| correspondence.forward_backward_error_px
                    <= RIG_FEATURE_MAX_FORWARD_BACKWARD_ERROR),
        );
    }

    #[test]
    fn checkerboard_does_not_render_occluded_target_samples() {
        let mut reference = Plane::new(8, 8);
        let mut target = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                reference.data[y * 8 + x] = (x + y) as f32;
                target.data[y * 8 + x] = 7.0 - x as f32 * 0.1;
            }
        }
        let mut warp = Warp::from_fn(16, 16, 4, Some);
        warp.visibility.fill(WarpVisibility::Occluded);
        let (samples, width, _) = debug_checkerboard(&reference, &target, &warp, 1);
        // x=1 is a target tile and must be black even though the mapping and
        // target sample are both finite. x=2 is a reference tile and remains.
        assert_eq!(samples[1], 0);
        assert!(samples[2] > 0);
        assert_eq!(width, 8);
    }

    #[test]
    fn visibility_is_conservative_across_occlusion_cells() {
        let mut warp = Warp::from_fn(16, 16, 8, Some);
        assert_eq!(warp.visibility(4.0, 4.0), WarpVisibility::Unknown);

        warp.visibility.fill(WarpVisibility::Visible);
        assert_eq!(warp.visibility(4.0, 4.0), WarpVisibility::Visible);

        // One occluded corner means this interpolation cell straddles a
        // foreground/background transition and must not be blended.
        warp.visibility[0] = WarpVisibility::Occluded;
        assert_eq!(warp.visibility(4.0, 4.0), WarpVisibility::Boundary);

        warp.visibility.fill(WarpVisibility::Occluded);
        assert_eq!(warp.visibility(4.0, 4.0), WarpVisibility::Occluded);
    }

    #[test]
    fn combined_warp_sample_matches_individual_accessors() {
        let mut warp = Warp::from_fn(16, 16, 8, |point| {
            Some([point[0] * 1.25 + 2.0, point[1] * 0.75 - 1.0])
        });
        for (index, confidence) in warp.confidence.iter_mut().enumerate() {
            *confidence = index as f32 / (warp.points.len() - 1) as f32;
        }
        warp.visibility.fill(WarpVisibility::Visible);
        warp.visibility[0] = WarpVisibility::Occluded;

        for point in [
            [0.0, 0.0],
            [3.25, 6.75],
            [8.0, 8.0],
            [15.9, 1.2],
            [-1.0, 4.0],
        ] {
            let sample = warp.sample(point[0], point[1]);
            assert_eq!(sample.mapped, warp.map(point[0], point[1]));
            assert_eq!(sample.confidence, warp.confidence(point[0], point[1]));
            assert_eq!(sample.visibility, warp.visibility(point[0], point[1]));
        }
    }

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
