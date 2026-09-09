//! Dense calibrated multi-view depth refinement.
//!
//! This stage solves dense correspondence and depth together from the calibrated
//! multi-camera rig. A reference-space inverse-depth hypothesis defines a local
//! 3-D surface; every patch sample is projected directly through each target
//! camera. The scene-fitted alignment warp is not an infinity observation: in
//! physical mode it supplies only a perpendicular epipolar search proposal and
//! remains the output fallback when finite depth is unresolved. A warp-seeded
//! compatibility path is retained when calibrated physical geometry itself is
//! unavailable.
//! Independent camera evidence is combined robustly so an occluded majority
//! cannot outvote a surface genuinely observed by a smaller set of views.
//! Eight-direction semi-global matching regularises weakly textured areas while an
//! edge-aware completion pass fills small holes without freely crossing image
//! discontinuities. The resulting metric field drives each camera's exact
//! calibrated parallax and carries local confidence into synthesis.

use std::{path::Path, thread};

use anyhow::Result;
use serde::Serialize;

use crate::{
    align::{AlignInput, ModuleAlignment, Warp, WarpVisibility},
    geometry::{Ray, ResolvedCamera},
    image::Plane,
    math::{Vec2, add, dot, norm, scale, sub},
};

const INFINITY_DEPTH: f64 = 1.0e8;
const MISSING_COST: f32 = 2.25;
const SGM_SMALL_PENALTY: f32 = 0.06;
const SGM_LARGE_PENALTY: f32 = 0.45;
const SGM_DIRECTIONS: f32 = 8.0;
const MINIMUM_REGULARIZED_SCORE: f32 = 0.30;
const MAXIMUM_REGULARIZED_BASELINE_LOSS: f32 = 0.04;
const NEAR_DEPTH_PRIOR: f32 = 0.10;
const WARP_BOUNDARY_DELTA_PX: f32 = 2.0;
const WARP_BOUNDARY_CONTRAST: f32 = 0.30;
// A directly measured surface must occupy more than a chance correlation
// island. At the default 4 px final grid this is still small enough to retain
// a roughly 20x20 px feature, while rejecting the salt-and-pepper clusters
// produced by sensor noise on blank walls and skies.
const MINIMUM_DIRECT_COMPONENT_NODES: usize = 24;

#[inline]
fn configured_worker_count(requested: usize, task_count: usize) -> usize {
    let automatic = thread::available_parallelism().map_or(1, usize::from);
    let requested = if requested == 0 { automatic } else { requested };
    requested.clamp(1, task_count.max(1))
}

// Physical-rig depth aggregation is an adaptive visibility/consensus model,
// not a fixed top-K vote.  A view whose ZNCC is far below the strongest
// mutually compatible evidence is treated as likely occluded/unsupported and
// becomes nearly neutral.  The support factor makes a depth backed by many
// independent cameras more convincing than an accidental match in only a
// couple of views, while still permitting a genuinely visible 2-3-camera
// surface to survive.
const PHYSICAL_CONSENSUS_BAND: f32 = 0.35;
const PHYSICAL_CONSENSUS_SOFTNESS: f32 = 0.07;
// Independent multi-view support changes only hypothesis ranking, never the
// absolute ZNCC threshold.  The confidence saturates quickly enough that a
// broad 8-10-view consensus can beat a chance 2-3-view peak while a genuine
// minority-visible surface is still allowed to exist.
const PHYSICAL_SUPPORT_CONFIDENCE_SATURATION: f32 = 1.5;
const PHYSICAL_MIN_INFORMATION_WEIGHT: f32 = 0.05;
const PHYSICAL_MAX_INFORMATION_WEIGHT: f32 = 2.00;
const PHYSICAL_VISIBLE_COMPATIBILITY: f32 = 0.55;
const PHYSICAL_UNKNOWN_CONFIDENCE_SCALE: f32 = 0.15;
const PHYSICAL_GLOBAL_FALLBACK_CONFIDENCE: f32 = 0.35;
// Direct dense matching runs on half-resolution luminance. Keep neighbouring
// physical hypotheses within roughly three quarters of a matching pixel so a
// narrow ZNCC peak cannot sit entirely between the fixed coarse planes.
const DIRECT_MAX_PROJECTED_STEP_PX: f64 = 1.5;
const DIRECT_MAX_DEPTH_REFINEMENTS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthGeometryMode {
    /// Compatibility path used only when calibrated physical geometry is not
    /// available for enough cameras. Finite depth is measured around the
    /// residual image-space warp produced by the legacy aligner.
    WarpSeeded,
    /// Preferred path: calibrated/refined cameras define depth-dependent
    /// parallax; capture alignment contributes only a perpendicular proposal.
    PhysicalRig,
}

impl DepthGeometryMode {
    #[inline]
    fn is_physical(self) -> bool {
        matches!(self, Self::PhysicalRig)
    }
}

#[derive(Clone, Debug)]
pub struct DepthOptions {
    pub enabled: bool,
    /// Worker threads used by the dense cost volume and direct verification
    /// passes (`0` = all available cores).
    pub threads: usize,
    /// Dense control-grid spacing in reference-raster pixels.
    pub grid_step: usize,
    /// Near and far search bounds in calibration units (believed millimetres).
    pub near_depth: f64,
    pub far_depth: f64,
    /// Number of uniformly spaced coarse finite inverse-depth hypotheses.
    /// Direct verification refines the winning interval until calibrated
    /// projected motion is locally bounded.
    pub planes: usize,
    /// Patch radius in half-resolution luminance pixels.
    pub patch_radius: usize,
    /// At least this many different target cameras must support a depth.
    pub minimum_support: usize,
    /// Legacy warp-seeded compatibility path only: average at most this many
    /// strongest views. Physical-rig depth uses adaptive all-view consensus
    /// instead and deliberately ignores this limit.
    pub best_view_count: usize,
    /// Weakest local ZNCC that may seed the dense reconstruction.
    pub minimum_score: f32,
    /// Minimum regularised separation from a non-adjacent depth label.
    pub minimum_margin: f32,
    /// A locally ambiguous label can still seed when it improves the active
    /// far/baseline hypothesis by at least this amount.
    pub minimum_improvement: f32,
    /// Minimum neighbouring estimates needed to complete a missing grid node.
    pub minimum_neighbour_support: usize,
    /// Optional edge-aware completion iterations for coarse search seeds.
    /// Final depth nodes are always independently remeasured.
    pub completion_iterations: usize,
}

impl Default for DepthOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            threads: 0,
            grid_step: 8,
            near_depth: 500.0,
            // Landscape subjects regularly extend well beyond 100 m. The
            // final inverse-depth bin acts as a calibrated near-infinity
            // hypothesis, while uniform inverse spacing preserves virtually
            // the same resolution for nearby geometry.
            far_depth: 10_000_000.0,
            planes: 96,
            patch_radius: 4,
            minimum_support: 2,
            best_view_count: 3,
            minimum_score: 0.45,
            minimum_margin: 0.01,
            minimum_improvement: 0.01,
            minimum_neighbour_support: 2,
            completion_iterations: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DepthAlignmentReport {
    /// True when dense correspondence used the physical rig depth locus with
    /// only a perpendicular proposal from the measured image alignment.
    pub physical_geometry: bool,
    pub tested_nodes: usize,
    /// Nodes that passed the direct finite-depth photometric decision before
    /// any image-space consistency cleanup.
    pub direct_selected_nodes: usize,
    /// Direct finite-depth nodes retained after requiring a compatible
    /// measurement at adjacent reference positions.
    pub neighbour_consistent_nodes: usize,
    /// Direct finite-depth nodes retained after rejecting small connected
    /// islands. This is the population subsequently reported as measured.
    pub component_consistent_nodes: usize,
    /// Nodes where no finite depth was selected but the measured stage-2
    /// output fallback itself had sufficient direct photometric support.
    pub far_supported_nodes: usize,
    /// Nodes whose depth is supported directly by the multiview cost volume.
    pub measured_nodes: usize,
    /// Nodes inferred by SGM or completed from edge-compatible neighbours.
    pub regularized_nodes: usize,
    /// Tested nodes that did not retain a finite reconstruction and therefore
    /// use the active far/baseline mapping. In physical mode this is a
    /// low-confidence fallback, not proof that the surface is actually at infinity.
    pub fallback_nodes: usize,
    /// Nodes at which this particular module accepted the reconstructed warp.
    pub refined_nodes: usize,
    pub occluded_nodes: usize,
    /// Nodes suppressed around a discontinuous warp boundary so bilinear
    /// interpolation cannot blend foreground and background mappings.
    pub boundary_nodes: usize,
    /// Nodes for which the final per-camera warp has a finite mapping. This is
    /// evidence-gated warp support, not geometric field-of-view overlap.
    pub defined_nodes: usize,
    /// `defined_nodes` divided by the complete reference-space warp grid.
    pub defined_fraction: f32,
    /// Nodes directly supported in this camera as either a refined finite
    /// depth or an independently verified far mapping.
    pub directly_supported_nodes: usize,
    /// `directly_supported_nodes` divided by the complete warp grid.
    pub directly_supported_fraction: f32,
    pub reconstructed_fraction: f32,
    pub refined_fraction: f32,
    pub occluded_fraction: f32,
    pub median_depth: Option<f64>,
    pub median_score_improvement: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthProvenance {
    Unsupported,
    Global,
    Measured,
    Regularized,
}

#[derive(Clone, Copy, Debug)]
pub struct DenseDepthNode {
    pub depth: Option<f64>,
    pub confidence: f32,
    pub provenance: DepthProvenance,
}

/// The exact reference-space control field used to construct depth-aware
/// warps. It intentionally remains at warp-grid resolution: upscaling it
/// would imply per-pixel measurements that the reconstruction does not have.
#[derive(Clone, Debug)]
pub struct DenseDepthMap {
    pub columns: usize,
    pub rows: usize,
    pub step: usize,
    pub near_depth: f64,
    pub far_depth: f64,
    pub nodes: Vec<DenseDepthNode>,
}

impl DenseDepthMap {
    pub fn node(&self, column: usize, row: usize) -> Option<DenseDepthNode> {
        (column < self.columns && row < self.rows).then(|| self.nodes[row * self.columns + column])
    }

    /// Nearest directly represented node for a reference-raster position.
    /// This deliberately does not interpolate across missing nodes: callers
    /// may use finite depth as a prior, but must not manufacture support in a
    /// textureless or contradictory region.
    pub fn sample_nearest(&self, x: f32, y: f32) -> Option<DenseDepthNode> {
        if x < 0.0 || y < 0.0 || self.step == 0 {
            return None;
        }
        let column = (x / self.step as f32).round() as usize;
        let row = (y / self.step as f32).round() as usize;
        self.node(column, row)
    }

    /// Write a quantitative inverse-depth image and a categorical provenance
    /// image. Inverse depth uses 0 for global/unsupported, 1 for the far bound,
    /// and 65535 for the near bound. Provenance is black=unsupported,
    /// blue=far/baseline fallback, green=measured, amber=regularized; finite colours
    /// are brightness-scaled by confidence.
    pub fn write_diagnostics(
        &self,
        inverse_depth_path: &Path,
        provenance_path: &Path,
    ) -> Result<()> {
        let (inverse_depth, provenance) = self.diagnostic_samples();
        chiaro_hotpixel_core::png16::write_gray16_native_atomic(
            inverse_depth_path,
            self.columns,
            self.rows,
            &inverse_depth,
        )?;
        chiaro_hotpixel_core::png16::write_rgb16_native_atomic(
            provenance_path,
            self.columns,
            self.rows,
            &provenance,
        )
    }

    /// Write a viewer-friendly logarithmic depth rendering. Finite depth runs
    /// from blue (far) through cyan/green/yellow to red (near); global fallback
    /// is dark grey and unsupported nodes are black. This is intentionally a
    /// visualization rather than a replacement for the quantitative inverse-
    /// depth image.
    pub fn write_visualization(&self, path: &Path) -> Result<()> {
        chiaro_hotpixel_core::png16::write_rgb16_native_atomic(
            path,
            self.columns,
            self.rows,
            &self.visualization_samples(),
        )
    }

    fn diagnostic_samples(&self) -> (Vec<u16>, Vec<u16>) {
        let near_inverse = 1.0 / self.near_depth;
        let far_inverse = 1.0 / self.far_depth;
        let range = (near_inverse - far_inverse).max(f64::MIN_POSITIVE);
        let mut inverse_depth = Vec::with_capacity(self.nodes.len());
        let mut provenance = Vec::with_capacity(self.nodes.len() * 3);
        for node in &self.nodes {
            let encoded_depth = node.depth.map_or(0, |depth| {
                let normalized = ((1.0 / depth - far_inverse) / range).clamp(0.0, 1.0);
                1 + (normalized * 65_534.0).round() as u16
            });
            inverse_depth.push(encoded_depth);
            let confidence = (0.35 + 0.65 * node.confidence.clamp(0.0, 1.0)) as f64;
            let color = match node.provenance {
                DepthProvenance::Unsupported => [0, 0, 0],
                DepthProvenance::Global => [0, 0, 32_768],
                DepthProvenance::Measured => scale_color([0, 65_535, 0], confidence),
                DepthProvenance::Regularized => scale_color([65_535, 32_768, 0], confidence),
            };
            provenance.extend(color);
        }
        (inverse_depth, provenance)
    }

    fn visualization_samples(&self) -> Vec<u16> {
        let log_range = (self.far_depth / self.near_depth).ln();
        let mut output = Vec::with_capacity(self.nodes.len() * 3);
        for node in &self.nodes {
            let color = match (node.depth, node.provenance) {
                (Some(depth), _) => {
                    let normalized =
                        ((self.far_depth.ln() - depth.ln()) / log_range).clamp(0.0, 1.0);
                    let confidence = 0.45 + 0.55 * f64::from(node.confidence.clamp(0.0, 1.0));
                    scale_color(depth_color(normalized), confidence)
                }
                // A directly supported far/baseline mapping is a censored
                // far-depth observation, not a missing measurement. Keep it
                // visibly distinct from both finite far blue and unsupported
                // black without inventing a metric distance.
                (None, DepthProvenance::Global) => [0, 0, 12_000],
                _ => [0, 0, 0],
            };
            output.extend(color);
        }
        output
    }
}

#[derive(Clone, Copy, Debug)]
struct NodeDepth {
    depth: f64,
    confidence: f32,
    improvement: f32,
    regularized: bool,
}

#[derive(Clone, Copy)]
enum NodeWarp {
    Undefined,
    /// A tested node that deliberately retains the far/baseline mapping.
    Global {
        point: [f32; 2],
        /// Confidence of the image-validated mapping being retained.
        confidence: f32,
    },
    /// Shared finite depth exists, but this camera did not independently
    /// verify that surface. Keep the physically correct finite projection and
    /// low confidence rather than silently jumping to infinity.
    Unknown {
        global: [f32; 2],
        point: [f32; 2],
        confidence: f32,
    },
    Refined {
        global: [f32; 2],
        point: [f32; 2],
        confidence: f32,
        measured: bool,
    },
    Occluded {
        global: [f32; 2],
        point: [f32; 2],
        measured: bool,
    },
    Boundary([f32; 2]),
}

#[derive(Clone, Copy, Debug)]
struct ViewScore {
    source_index: usize,
    /// Raw per-view ZNCC used for reconstruction-quality thresholds.
    score: f32,
    /// The same camera's physically projected far/infinity ZNCC for a finite
    /// hypothesis. Keeping this paired observation lets finite-vs-far
    /// comparisons use exactly the same camera population and weights.
    far_score: Option<f32>,
    /// Per-camera self-normalized evidence used only for visibility mixture
    /// membership. In physical finite-depth mode this is the improvement over
    /// that same camera's measured infinity projection, which makes B/C
    /// optical paths less dependent on having identical absolute ZNCC scales.
    compatibility_score: f32,
    /// Relative inverse-depth information carried by this physical view.
    /// This is normalized robustly during aggregation, so only ratios matter.
    depth_information: f32,
}

#[derive(Clone, Copy, Debug)]
struct AggregateEvidence {
    /// Stable ZNCC-like score used by absolute quality thresholds.
    photometric_score: f32,
    /// Finite and far scores evaluated over the exact same weighted camera
    /// population, when this is a finite hypothesis.  Their difference is the
    /// only valid finite-vs-far improvement metric.
    paired_photometric_score: Option<f32>,
    paired_far_score: Option<f32>,
    /// Ranking score used only to choose between competing depth hypotheses.
    /// It may contain bounded bonuses for independent support and depth
    /// information, but never replaces the photometric score in thresholds.
    ranking_score: f32,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct PhysicalMember {
    source_index: usize,
    score: f32,
    compatibility: f32,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct PhysicalConsensus {
    evidence: AggregateEvidence,
    members: Vec<PhysicalMember>,
}

#[derive(Clone, Copy)]
struct ViewRefinement {
    score: f32,
    point: [f32; 2],
}

#[derive(Clone, Copy)]
struct ViewPair<'a> {
    reference: &'a Plane,
    target: &'a Plane,
    global: &'a Warp,
}

struct CostVolume {
    /// Whether finite hypotheses were generated from calibrated physical
    /// projection. In that mode finite-vs-far decisions must never fall back
    /// to an independently aggregated far camera population.
    physical_geometry: bool,
    /// Label zero is the mode-specific far/baseline hypothesis; remaining
    /// labels are finite depths ordered from far to near in uniform inverse-depth increments.
    labels: Vec<Option<f64>>,
    scores: Vec<f32>,
    /// Finite-vs-far improvement evaluated over identical per-camera weights.
    /// NaN for the far label or when no paired comparison was possible.
    paired_improvements: Vec<f32>,
    costs: Vec<f32>,
    guidance: Vec<f32>,
    tested: Vec<bool>,
}

struct DirectDepthField {
    columns: usize,
    rows: usize,
    step: usize,
    nodes: Vec<Option<NodeDepth>>,
    guidance: Vec<f32>,
    /// At least one hypothesis could be evaluated at this node.
    tested: Vec<bool>,
    /// The far/infinity hypothesis itself had enough direct photometric
    /// support to be a meaningful fallback. This is intentionally distinct
    /// from `tested`: a rejected finite island must not silently become far.
    far_supported: Vec<bool>,
}

#[derive(Clone, Copy)]
struct TargetVisibilitySample {
    pixel: Vec2,
    depth: f64,
}

struct TargetVisibilityBuffer {
    cell_size: f64,
    columns: usize,
    rows: usize,
    samples: Vec<Option<TargetVisibilitySample>>,
}

impl TargetVisibilityBuffer {
    fn nearer_surface_at(&self, pixel: Vec2, point_depth: f64) -> bool {
        if !point_depth.is_finite() || point_depth <= 0.0 {
            return false;
        }
        let column = (pixel[0] / self.cell_size).floor() as isize;
        let row = (pixel[1] / self.cell_size).floor() as isize;
        let mut nearest = None::<f64>;
        let radius = self.cell_size * 1.75;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let x = column + dx;
                let y = row + dy;
                if x < 0 || y < 0 || x >= self.columns as isize || y >= self.rows as isize {
                    continue;
                }
                let Some(sample) = self.samples[y as usize * self.columns + x as usize] else {
                    continue;
                };
                let distance = ((sample.pixel[0] - pixel[0]).powi(2)
                    + (sample.pixel[1] - pixel[1]).powi(2))
                .sqrt();
                if distance > radius {
                    continue;
                }
                nearest = Some(nearest.map_or(sample.depth, |current| current.min(sample.depth)));
            }
        }
        let Some(nearest) = nearest else {
            return false;
        };
        // Conservative geometric ordering tolerance.  This z-buffer is built
        // from a sparse depth grid, so only a clearly nearer surface may hard
        // block the shared point; small differences remain Unknown.
        let tolerance = (point_depth * 0.01).max(50.0);
        nearest + tolerance < point_depth
    }
}

fn build_target_visibility_buffer(
    field: &[Option<NodeDepth>],
    columns: usize,
    rows: usize,
    step: usize,
    reference: &ResolvedCamera,
    target: &ResolvedCamera,
    measured_proposal: &Warp,
    options: &DepthOptions,
) -> TargetVisibilityBuffer {
    let cell_size = step.max(2) as f64;
    let target_columns = target.width.div_ceil(step.max(2)) + 1;
    let target_rows = target.height.div_ceil(step.max(2)) + 1;
    let mut samples: Vec<Option<TargetVisibilitySample>> = vec![None; target_columns * target_rows];
    for row in 0..rows {
        for column in 0..columns {
            let Some(node) = field[row * columns + column] else {
                continue;
            };
            // Hard occlusion is destructive downstream, so build the z-buffer
            // only from directly measured finite surfaces. Regularised depth
            // may guide reconstruction but must not become categorical
            // foreground evidence in another view.
            if node.regularized {
                continue;
            }
            let pixel = [(column * step) as f64, (row * step) as f64];
            let ray = reference.pixel_to_ray(pixel);
            let world = add(ray.origin, scale(ray.direction, node.depth));
            let Some(projected) = local_patch_projection(
                reference,
                target,
                measured_proposal,
                pixel,
                Some(node.depth),
                options,
            )
            .map(|projection| projection.target_centre) else {
                continue;
            };
            if !target.contains(projected) {
                continue;
            }
            let depth = norm(sub(world, target.center()));
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }
            let x = (projected[0] / cell_size)
                .floor()
                .clamp(0.0, (target_columns - 1) as f64) as usize;
            let y = (projected[1] / cell_size)
                .floor()
                .clamp(0.0, (target_rows - 1) as f64) as usize;
            let index = y * target_columns + x;
            if samples[index].is_none_or(|current| depth < current.depth) {
                samples[index] = Some(TargetVisibilitySample {
                    pixel: projected,
                    depth,
                });
            }
        }
    }
    TargetVisibilityBuffer {
        cell_size,
        columns: target_columns,
        rows: target_rows,
        samples,
    }
}

/// Refine calibrated targets against one shared dense reference depth field.
/// In physical-rig mode, dense correspondences use the physical depth locus;
/// the image-fitted warp proposes only its perpendicular displacement and does
/// not compete as an infinity measurement. It remains the output fallback.
/// Warp-seeded mode preserves the previous residual-alignment behavior.
pub fn refine_multiview_depth(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &mut [ModuleAlignment],
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
) -> Option<DenseDepthMap> {
    if !valid_options(options, geometry_mode)
        || inputs.len() != alignments.len()
        || reference_index >= inputs.len()
        || inputs[reference_index].camera.is_none()
    {
        return None;
    }
    if geometry_mode.is_physical() {
        for (index, alignment) in alignments.iter_mut().enumerate() {
            alignment.report.geometry_accepted = Some(index == reference_index);
        }
    }
    let reference = &inputs[reference_index];
    let coarse_step = options.grid_step.max(4);
    let coarse_columns = reference.width.div_ceil(coarse_step) + 1;
    let coarse_rows = reference.height.div_ceil(coarse_step) + 1;
    let volume = build_cost_volume(
        inputs,
        reference_index,
        alignments,
        coarse_columns,
        coarse_rows,
        coarse_step,
        options,
        geometry_mode,
    );
    let regularised = semi_global_costs(
        &volume.costs,
        &volume.guidance,
        coarse_columns,
        coarse_rows,
        volume.labels.len(),
    );
    let (mut coarse_field, fillable) = select_depths(&volume, &regularised, options);
    let inverse_step =
        (1.0 / options.near_depth - 1.0 / options.far_depth) / (options.planes - 1) as f64;
    if options.completion_iterations > 0 {
        complete_depth_field(
            &mut coarse_field,
            &volume.guidance,
            &fillable,
            coarse_columns,
            coarse_rows,
            options,
        );
    }
    drop(regularised);
    drop(volume);

    // SGM supplies only a search seed. Final nodes live on a finer grid and
    // must independently reproduce multiview evidence; no coarse value is
    // copied into the output and no final hole is spatially completed.
    let direct = measure_direct_depths(
        inputs,
        reference_index,
        alignments,
        &coarse_field,
        coarse_columns,
        coarse_rows,
        coarse_step,
        inverse_step,
        options,
        geometry_mode,
    );
    let DirectDepthField {
        columns,
        rows,
        step,
        nodes: mut field,
        guidance,
        tested,
        far_supported,
    } = direct;
    let direct_selected_nodes = field.iter().flatten().count();
    let far_supported_nodes = far_supported.iter().filter(|&&supported| supported).count();
    reject_isolated_direct_depths(&mut field, &guidance, columns, rows, inverse_step * 2.5);
    let neighbour_consistent_nodes = field.iter().flatten().count();
    reject_small_direct_components(
        &mut field,
        &guidance,
        columns,
        rows,
        inverse_step * 2.5,
        MINIMUM_DIRECT_COMPONENT_NODES,
    );
    let component_consistent_nodes = field.iter().flatten().count();
    fit_local_depth_planes(&mut field, &guidance, columns, rows, inverse_step * 4.0);

    let tested_nodes = tested.iter().filter(|&&tested| tested).count();
    let measured_nodes = field
        .iter()
        .flatten()
        .filter(|node| !node.regularized)
        .count();
    let regularized_nodes = field
        .iter()
        .flatten()
        .filter(|node| node.regularized)
        .count();
    let mut selected_depths = field
        .iter()
        .flatten()
        .map(|node| node.depth)
        .collect::<Vec<_>>();
    let mut improvements = field
        .iter()
        .flatten()
        .filter(|node| !node.regularized)
        .map(|node| node.improvement)
        .collect::<Vec<_>>();
    selected_depths.sort_by(f64::total_cmp);
    improvements.sort_by(f32::total_cmp);

    // Freeze the pre-dense warps before any target is rewritten. Physical
    // visibility is one shared scene-surface consensus, so evaluate it once per
    // node against this immutable snapshot rather than once per target against
    // a progressively mutated mixture of old and new warps. Warp-seeded mode
    // also reads from the snapshot so the construction loop has no mutable/read
    // aliasing and every target sees the same input alignment generation.
    let alignment_snapshot = alignments.to_vec();
    let physical_scoring = geometry_mode.is_physical().then(|| {
        physical_visibility_memberships(
            inputs,
            reference_index,
            &alignment_snapshot,
            &field,
            &far_supported,
            columns,
            rows,
            step,
            options,
        )
    });

    for target_index in 0..alignments.len() {
        if target_index == reference_index
            || inputs[target_index].camera.is_none()
            || (!geometry_mode.is_physical() && !alignments[target_index].report.accepted)
        {
            continue;
        }
        let reference_camera = reference.camera.expect("validated reference camera");
        let target_camera = inputs[target_index]
            .camera
            .expect("filtered calibrated target camera");
        let base_alignment = &alignment_snapshot[target_index];
        let target_visibility = geometry_mode.is_physical().then(|| {
            build_target_visibility_buffer(
                &field,
                columns,
                rows,
                step,
                reference_camera,
                target_camera,
                &base_alignment.warp,
                options,
            )
        });
        let mut decisions = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                let p = [(column * step) as f64, (row * step) as f64];
                match geometry_mode {
                    DepthGeometryMode::PhysicalRig => {
                        let far_q =
                            base_alignment
                                .warp
                                .map(p[0] as f32, p[1] as f32)
                                .filter(|point| {
                                    target_camera
                                        .contains([f64::from(point[0]), f64::from(point[1])])
                                });
                        let Some(node) = field[index] else {
                            if far_supported[index] {
                                let Some(far_q) = far_q else {
                                    decisions.push(NodeWarp::Undefined);
                                    continue;
                                };
                                let far_reference_ray = reference_camera.pixel_to_ray(p);
                                let far_world = add(
                                    far_reference_ray.origin,
                                    scale(far_reference_ray.direction, INFINITY_DEPTH),
                                );
                                let far_target_depth = norm(sub(far_world, target_camera.center()));
                                let far_occluded =
                                    target_visibility.as_ref().is_some_and(|buffer| {
                                        buffer.nearer_surface_at(
                                            [f64::from(far_q[0]), f64::from(far_q[1])],
                                            far_target_depth,
                                        )
                                    });
                                if !inputs[target_index].depth_evidence_enabled {
                                    // Held-out validation may use geometry
                                    // inferred from other cameras, including a
                                    // physical z-buffer occlusion decision, but
                                    // its image values are never consulted.
                                    if far_occluded {
                                        decisions.push(NodeWarp::Occluded {
                                            global: far_q,
                                            point: far_q,
                                            measured: false,
                                        });
                                    } else {
                                        decisions.push(NodeWarp::Unknown {
                                            global: far_q,
                                            point: far_q,
                                            confidence: PHYSICAL_UNKNOWN_CONFIDENCE_SCALE,
                                        });
                                    }
                                    continue;
                                }
                                let visible = physical_scoring.as_ref().is_some_and(
                                    |(memberships, words_per_node)| {
                                        physical_membership_contains(
                                            memberships,
                                            *words_per_node,
                                            index,
                                            target_index,
                                        )
                                    },
                                );
                                if visible {
                                    decisions.push(NodeWarp::Global {
                                        point: far_q,
                                        confidence: base_alignment
                                            .warp
                                            .confidence(p[0] as f32, p[1] as f32),
                                    });
                                } else if far_occluded {
                                    decisions.push(NodeWarp::Occluded {
                                        global: far_q,
                                        point: far_q,
                                        measured: false,
                                    });
                                } else {
                                    decisions.push(NodeWarp::Unknown {
                                        global: far_q,
                                        point: far_q,
                                        confidence: PHYSICAL_UNKNOWN_CONFIDENCE_SCALE,
                                    });
                                }
                            } else {
                                // Either nothing was measurable or a finite
                                // hypothesis was rejected by spatial
                                // consistency. Neither case is evidence for
                                // infinity.
                                decisions.push(NodeWarp::Undefined);
                            }
                            continue;
                        };

                        let Some(mapped) = local_patch_projection(
                            reference_camera,
                            target_camera,
                            &base_alignment.warp,
                            p,
                            Some(node.depth),
                            options,
                        )
                        .map(|projection| projection.target_centre) else {
                            decisions.push(NodeWarp::Undefined);
                            continue;
                        };
                        if !target_camera.contains(mapped) {
                            decisions.push(NodeWarp::Undefined);
                            continue;
                        }
                        let finite_q = [mapped[0] as f32, mapped[1] as f32];
                        let global_q = far_q.unwrap_or(finite_q);
                        let reference_ray = reference_camera.pixel_to_ray(p);
                        let world_point = add(
                            reference_ray.origin,
                            scale(reference_ray.direction, node.depth),
                        );
                        let target_depth = norm(sub(world_point, target_camera.center()));
                        let geometrically_occluded = target_visibility
                            .as_ref()
                            .is_some_and(|buffer| buffer.nearer_surface_at(mapped, target_depth));

                        if !inputs[target_index].depth_evidence_enabled {
                            // Geometry for a held-out camera is projected from
                            // the depth inferred by the other cameras.  Keep it
                            // usable for evaluation without consulting held-out
                            // radiance for visibility/refinement.
                            if geometrically_occluded {
                                decisions.push(NodeWarp::Occluded {
                                    global: global_q,
                                    point: finite_q,
                                    measured: !node.regularized,
                                });
                            } else {
                                decisions.push(NodeWarp::Unknown {
                                    global: global_q,
                                    point: finite_q,
                                    confidence: node.confidence,
                                });
                            }
                            continue;
                        }

                        // Membership in the shared physical consensus was
                        // evaluated once for this node from the immutable
                        // pre-dense warp snapshot. Visibility is decided before
                        // any per-camera local residual refinement.
                        let visible = physical_scoring.as_ref().is_some_and(
                            |(memberships, words_per_node)| {
                                physical_membership_contains(
                                    memberships,
                                    *words_per_node,
                                    index,
                                    target_index,
                                )
                            },
                        );

                        if visible {
                            if let Some(selected) = refine_one_view_physical(
                                reference,
                                &inputs[target_index],
                                &base_alignment.warp,
                                p,
                                node.depth,
                                options,
                            ) {
                                decisions.push(NodeWarp::Refined {
                                    global: global_q,
                                    point: selected.point,
                                    confidence: if node.regularized {
                                        node.confidence * 0.6
                                    } else {
                                        0.5 + 0.5 * node.confidence
                                    },
                                    measured: !node.regularized,
                                });
                            } else {
                                decisions.push(NodeWarp::Unknown {
                                    global: global_q,
                                    point: finite_q,
                                    confidence: node.confidence * PHYSICAL_UNKNOWN_CONFIDENCE_SCALE,
                                });
                            }
                            continue;
                        }

                        // Hard occlusion requires geometric depth ordering
                        // in the target view.  Photometric disagreement alone
                        // remains Unknown so a different PSF/spectrum cannot
                        // accidentally block a valid source.
                        if geometrically_occluded {
                            decisions.push(NodeWarp::Occluded {
                                global: global_q,
                                point: finite_q,
                                measured: !node.regularized,
                            });
                        } else {
                            decisions.push(NodeWarp::Unknown {
                                global: global_q,
                                point: finite_q,
                                confidence: node.confidence * PHYSICAL_UNKNOWN_CONFIDENCE_SCALE,
                            });
                        }
                    }
                    DepthGeometryMode::WarpSeeded => {
                        let Some(fallback_q) = base_alignment
                            .warp
                            .map(p[0] as f32, p[1] as f32)
                            .filter(|point| {
                                target_camera.contains([f64::from(point[0]), f64::from(point[1])])
                            })
                        else {
                            decisions.push(NodeWarp::Undefined);
                            continue;
                        };
                        let Some(node) = field[index] else {
                            decisions.push(NodeWarp::Global {
                                point: fallback_q,
                                confidence: base_alignment
                                    .warp
                                    .confidence(p[0] as f32, p[1] as f32),
                            });
                            continue;
                        };
                        let selected = refine_one_view_warp_seeded(
                            reference,
                            &inputs[target_index],
                            &base_alignment.warp,
                            p,
                            node.depth,
                            inverse_step,
                            options,
                        );
                        let baseline = score_one_view_warp_seeded(
                            reference,
                            &inputs[target_index],
                            &base_alignment.warp,
                            p,
                            None,
                            options.patch_radius,
                        );
                        let supported = selected.is_some_and(|selected| {
                            selected.score >= options.minimum_score
                                && baseline
                                    .is_none_or(|baseline| selected.score + 0.01 >= baseline.score)
                        });
                        let contradicted = match (selected, baseline) {
                            (Some(selected), Some(baseline)) => {
                                selected.score + 0.12 < baseline.score
                            }
                            _ => false,
                        };
                        if contradicted {
                            decisions.push(NodeWarp::Occluded {
                                global: fallback_q,
                                point: fallback_q,
                                measured: !node.regularized,
                            });
                        } else if supported {
                            let selected = selected.expect("supported refinement");
                            decisions.push(NodeWarp::Refined {
                                global: fallback_q,
                                point: selected.point,
                                confidence: if node.regularized {
                                    node.confidence * 0.6
                                } else {
                                    0.5 + 0.5 * node.confidence
                                },
                                measured: !node.regularized,
                            });
                        } else {
                            decisions.push(NodeWarp::Global {
                                point: fallback_q,
                                confidence: base_alignment
                                    .warp
                                    .confidence(p[0] as f32, p[1] as f32),
                            });
                        }
                    }
                }
            }
        }
        enforce_warp_consensus(&mut decisions, columns, rows, geometry_mode);
        suppress_warp_boundaries(&mut decisions, &guidance, columns, rows);
        let mut points = Vec::with_capacity(columns * rows);
        let mut confidence = Vec::with_capacity(columns * rows);
        let mut visibility = Vec::with_capacity(columns * rows);
        let mut refined_nodes = 0usize;
        let mut far_nodes = 0usize;
        let mut occluded_nodes = 0usize;
        let mut boundary_nodes = 0usize;
        let mut defined_nodes = 0usize;
        for decision in decisions {
            match decision {
                NodeWarp::Undefined => {
                    points.push([f32::NAN; 2]);
                    confidence.push(0.0);
                    visibility.push(WarpVisibility::Unknown);
                }
                NodeWarp::Global {
                    point,
                    confidence: c,
                } => {
                    points.push(point);
                    confidence.push(c);
                    // In physical mode `Global` is emitted only after this
                    // particular camera independently supports the shared far
                    // hypothesis.  Unverified far mappings are `Unknown`.
                    visibility.push(if geometry_mode.is_physical() {
                        WarpVisibility::Visible
                    } else {
                        WarpVisibility::Unknown
                    });
                    far_nodes += 1;
                    defined_nodes += 1;
                }
                NodeWarp::Unknown {
                    point,
                    confidence: c,
                    ..
                } => {
                    points.push(point);
                    confidence.push(c);
                    visibility.push(WarpVisibility::Unknown);
                    defined_nodes += 1;
                }
                NodeWarp::Refined {
                    point,
                    confidence: c,
                    ..
                } => {
                    points.push(point);
                    confidence.push(c);
                    visibility.push(WarpVisibility::Visible);
                    refined_nodes += 1;
                    defined_nodes += 1;
                }
                NodeWarp::Occluded { point, .. } => {
                    points.push(point);
                    confidence.push(0.0);
                    visibility.push(WarpVisibility::Occluded);
                    occluded_nodes += 1;
                    defined_nodes += 1;
                }
                NodeWarp::Boundary(point) => {
                    points.push(point);
                    confidence.push(0.0);
                    visibility.push(WarpVisibility::Boundary);
                    boundary_nodes += 1;
                    defined_nodes += 1;
                }
            }
        }
        alignments[target_index].warp = Warp {
            step,
            columns,
            rows,
            points,
            confidence,
            visibility,
        };
        let directly_supported_nodes = refined_nodes + far_nodes;
        let warp_node_count = columns * rows;
        alignments[target_index].report.depth = Some(DepthAlignmentReport {
            physical_geometry: geometry_mode.is_physical(),
            tested_nodes,
            direct_selected_nodes,
            neighbour_consistent_nodes,
            component_consistent_nodes,
            far_supported_nodes,
            measured_nodes,
            regularized_nodes,
            fallback_nodes: far_nodes,
            refined_nodes,
            occluded_nodes,
            boundary_nodes,
            defined_nodes,
            defined_fraction: fraction(defined_nodes, warp_node_count),
            directly_supported_nodes,
            directly_supported_fraction: fraction(directly_supported_nodes, warp_node_count),
            reconstructed_fraction: fraction(measured_nodes + regularized_nodes, tested_nodes),
            refined_fraction: fraction(refined_nodes, tested_nodes),
            occluded_fraction: fraction(occluded_nodes, tested_nodes),
            median_depth: median(&selected_depths),
            median_score_improvement: median(&improvements),
        });
        if geometry_mode.is_physical() {
            // Keep `AlignmentReport::coverage` as the geometric overlap
            // measured before dense depth. Evidence-gated warp support is
            // reported separately above. A distant scene may be usable
            // through a directly supported far mapping even when it contains
            // no finite-depth nodes, so admission must not require
            // `refined_nodes` alone.
            let admission_nodes = if inputs[target_index].depth_evidence_enabled {
                directly_supported_nodes
            } else {
                // Held-out validation geometry is inferred entirely from other
                // views, so target-radiance support is intentionally absent.
                defined_nodes
            };
            alignments[target_index].report.geometry_accepted =
                Some(admission_nodes >= MINIMUM_DIRECT_COMPONENT_NODES);
        }
    }
    Some(DenseDepthMap {
        columns,
        rows,
        step,
        near_depth: options.near_depth,
        far_depth: options.far_depth,
        nodes: field
            .into_iter()
            .zip(far_supported)
            .map(|(node, far_supported)| match node {
                Some(node) => DenseDepthNode {
                    depth: Some(node.depth),
                    confidence: node.confidence,
                    provenance: if node.regularized {
                        DepthProvenance::Regularized
                    } else {
                        DepthProvenance::Measured
                    },
                },
                None if far_supported => DenseDepthNode {
                    depth: None,
                    confidence: if geometry_mode.is_physical() {
                        PHYSICAL_GLOBAL_FALLBACK_CONFIDENCE
                    } else {
                        1.0
                    },
                    provenance: DepthProvenance::Global,
                },
                None => DenseDepthNode {
                    depth: None,
                    confidence: 0.0,
                    provenance: DepthProvenance::Unsupported,
                },
            })
            .collect(),
    })
}

fn enforce_warp_consensus(
    decisions: &mut [NodeWarp],
    columns: usize,
    rows: usize,
    geometry_mode: DepthGeometryMode,
) {
    let source = decisions.to_vec();
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let (kind, global, point, measured, minimum_neighbours) = match source[index] {
                NodeWarp::Refined {
                    global,
                    point,
                    measured,
                    ..
                } => (0, global, point, measured, if measured { 2 } else { 4 }),
                NodeWarp::Occluded {
                    global,
                    point,
                    measured,
                } => (1, global, point, measured, if measured { 3 } else { 5 }),
                _ => continue,
            };
            let mut neighbours = 0usize;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (x, y) = (column as i32 + dx, row as i32 + dy);
                    if x < 0 || y < 0 || x >= columns as i32 || y >= rows as i32 {
                        continue;
                    }
                    neighbours += usize::from(warp_decisions_agree(
                        kind,
                        source[index],
                        source[y as usize * columns + x as usize],
                    ));
                }
            }
            if neighbours < minimum_neighbours {
                decisions[index] = if geometry_mode.is_physical() {
                    NodeWarp::Unknown {
                        global,
                        point,
                        confidence: PHYSICAL_UNKNOWN_CONFIDENCE_SCALE
                            * if measured { 0.75 } else { 0.40 },
                    }
                } else {
                    NodeWarp::Global {
                        point: global,
                        confidence: 1.0,
                    }
                };
            }
        }
    }
}

fn suppress_warp_boundaries(
    decisions: &mut [NodeWarp],
    guidance: &[f32],
    columns: usize,
    rows: usize,
) {
    let source = decisions.to_vec();
    let mut boundary = vec![false; decisions.len()];
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let Some((global, point)) = warp_mapping(source[index]) else {
                continue;
            };
            let delta = [point[0] - global[0], point[1] - global[1]];
            // Four-connected pairs are sufficient for bilinear grid cells and
            // avoid turning an isolated diagonal contrast sample into a broad
            // two-node exclusion band.
            for (x, y) in [(column + 1, row), (column, row + 1)] {
                if x >= columns || y >= rows {
                    continue;
                }
                let neighbour_index = y * columns + x;
                let Some((neighbour_global, neighbour_point)) =
                    warp_mapping(source[neighbour_index])
                else {
                    continue;
                };
                let neighbour_delta = [
                    neighbour_point[0] - neighbour_global[0],
                    neighbour_point[1] - neighbour_global[1],
                ];
                let difference = [delta[0] - neighbour_delta[0], delta[1] - neighbour_delta[1]];
                let mapping_edge = difference[0] * difference[0] + difference[1] * difference[1]
                    > WARP_BOUNDARY_DELTA_PX * WARP_BOUNDARY_DELTA_PX;
                let image_edge =
                    (guidance[index] - guidance[neighbour_index]).abs() > WARP_BOUNDARY_CONTRAST;
                if mapping_edge && image_edge {
                    boundary[index] = true;
                    boundary[neighbour_index] = true;
                }
            }
        }
    }
    for (index, is_boundary) in boundary.into_iter().enumerate() {
        if is_boundary {
            let Some((_, point)) = warp_mapping(source[index]) else {
                continue;
            };
            decisions[index] = NodeWarp::Boundary(point);
        }
    }
}

fn warp_mapping(decision: NodeWarp) -> Option<([f32; 2], [f32; 2])> {
    match decision {
        NodeWarp::Global { point, .. } => Some((point, point)),
        NodeWarp::Unknown { global, point, .. }
        | NodeWarp::Refined { global, point, .. }
        | NodeWarp::Occluded { global, point, .. } => Some((global, point)),
        _ => None,
    }
}

fn warp_decisions_agree(kind: u8, centre: NodeWarp, neighbour: NodeWarp) -> bool {
    match (kind, centre, neighbour) {
        (
            0,
            NodeWarp::Refined {
                global: centre_global,
                point: centre_point,
                ..
            },
            NodeWarp::Refined {
                global: neighbour_global,
                point: neighbour_point,
                ..
            },
        ) => {
            let centre_delta = [
                centre_point[0] - centre_global[0],
                centre_point[1] - centre_global[1],
            ];
            let neighbour_delta = [
                neighbour_point[0] - neighbour_global[0],
                neighbour_point[1] - neighbour_global[1],
            ];
            let dx = centre_delta[0] - neighbour_delta[0];
            let dy = centre_delta[1] - neighbour_delta[1];
            dx * dx + dy * dy <= 9.0
        }
        (1, NodeWarp::Occluded { .. }, NodeWarp::Occluded { .. }) => true,
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_direct_depths(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    coarse: &[Option<NodeDepth>],
    coarse_columns: usize,
    coarse_rows: usize,
    coarse_step: usize,
    inverse_step: f64,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
) -> DirectDepthField {
    let step = (coarse_step / 2).max(4);
    let columns = inputs[reference_index].width.div_ceil(step) + 1;
    let rows = inputs[reference_index].height.div_ceil(step) + 1;
    let worker_count = configured_worker_count(options.threads, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let direct_options = DepthOptions {
        patch_radius: options.patch_radius.max(3),
        ..options.clone()
    };
    let wide_options = DepthOptions {
        patch_radius: (options.patch_radius * 2).max(8),
        ..options.clone()
    };
    let active_view_indices = (0..inputs.len())
        .filter(|&index| {
            index != reference_index
                && inputs[index].camera.is_some()
                && inputs[index].depth_evidence_enabled
                && (geometry_mode.is_physical() || alignments[index].report.accepted)
        })
        .collect::<Vec<_>>();
    let chunks = thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(rows_per_worker)
            .map(|first_row| {
                let last_row = (first_row + rows_per_worker).min(rows);
                let direct_options = &direct_options;
                let wide_options = &wide_options;
                let active_view_indices = &active_view_indices;
                scope.spawn(move || {
                    let capacity = (last_row - first_row) * columns;
                    let mut nodes = Vec::with_capacity(capacity);
                    let mut guidance = Vec::with_capacity(capacity);
                    let mut tested = Vec::with_capacity(capacity);
                    let mut far_supported = Vec::with_capacity(capacity);

                    // Reuse all hot-path storage for every node handled by this
                    // worker. At the final dense grid this avoids millions of
                    // small allocations and keeps the prepared reference patch
                    // and per-view scoring buffers resident.
                    let mut candidates = Vec::with_capacity(40);
                    let mut scores = Vec::with_capacity(40);
                    let mut refinement_candidates = Vec::with_capacity(2);
                    let mut scored_additions = Vec::with_capacity(2);
                    let mut merge_scratch = Vec::with_capacity(48);
                    let mut wide_scores = Vec::with_capacity(8);
                    let mut view_scores = Vec::with_capacity(active_view_indices.len());
                    let mut aggregate_scratch = AggregateScratch {
                        ordered: Vec::with_capacity(active_view_indices.len()),
                        positive_information: Vec::with_capacity(active_view_indices.len()),
                    };
                    let mut reference_patch =
                        PreparedReferencePatch::with_radius(direct_options.patch_radius);
                    let mut wide_reference_patch =
                        PreparedReferencePatch::with_radius(wide_options.patch_radius);
                    let mut far_scores = vec![None; inputs.len()];
                    let mut wide_far_scores = vec![None; inputs.len()];
                    let mut depth_information = vec![0.0f32; inputs.len()];

                    for row in first_row..last_row {
                        for column in 0..columns {
                            let p = [(column * step) as f64, (row * step) as f64];
                            guidance.push(reference_guidance(&inputs[reference_index], p));
                            let seed = nearest_coarse_depth(
                                coarse,
                                coarse_columns,
                                coarse_rows,
                                coarse_step,
                                p,
                            );
                            direct_depth_candidates_into(
                                seed.map(|node| node.depth),
                                inverse_step,
                                direct_options,
                                &mut candidates,
                            );
                            scores.clear();
                            scores.reserve(candidates.len());

                            let (prepared_reference, reference_rays) = prepare_physical_score_cache(
                                inputs,
                                reference_index,
                                alignments,
                                active_view_indices,
                                p,
                                direct_options,
                                geometry_mode,
                                &mut reference_patch,
                                &mut far_scores,
                                &mut depth_information,
                            );
                            let baseline = score_prepared_depth(
                                inputs,
                                reference_index,
                                alignments,
                                active_view_indices,
                                p,
                                None,
                                direct_options,
                                geometry_mode,
                                prepared_reference.then_some(&reference_patch),
                                reference_rays.as_ref(),
                                &far_scores,
                                &depth_information,
                                &mut view_scores,
                                &mut aggregate_scratch,
                            );
                            for &depth in &candidates {
                                scores.push(score_prepared_depth(
                                    inputs,
                                    reference_index,
                                    alignments,
                                    active_view_indices,
                                    p,
                                    Some(depth),
                                    direct_options,
                                    geometry_mode,
                                    prepared_reference.then_some(&reference_patch),
                                    reference_rays.as_ref(),
                                    &far_scores,
                                    &depth_information,
                                    &mut view_scores,
                                    &mut aggregate_scratch,
                                ));
                            }
                            if geometry_mode.is_physical() {
                                for _ in 0..DIRECT_MAX_DEPTH_REFINEMENTS {
                                    let Some(best) =
                                        best_depth_candidate(&candidates, &scores, direct_options)
                                    else {
                                        break;
                                    };
                                    direct_depth_refinement_candidates_into(
                                        inputs,
                                        reference_index,
                                        p,
                                        &candidates,
                                        best,
                                        &mut refinement_candidates,
                                    );
                                    if refinement_candidates.is_empty() {
                                        break;
                                    }
                                    scored_additions.clear();
                                    for &depth in &refinement_candidates {
                                        let score = score_prepared_depth(
                                            inputs,
                                            reference_index,
                                            alignments,
                                            active_view_indices,
                                            p,
                                            Some(depth),
                                            direct_options,
                                            geometry_mode,
                                            prepared_reference.then_some(&reference_patch),
                                            reference_rays.as_ref(),
                                            &far_scores,
                                            &depth_information,
                                            &mut view_scores,
                                            &mut aggregate_scratch,
                                        );
                                        scored_additions.push((depth, score));
                                    }
                                    merge_depth_scores_with_scratch(
                                        &mut candidates,
                                        &mut scores,
                                        &scored_additions,
                                        &mut merge_scratch,
                                    );
                                }
                            }
                            let node_tested =
                                baseline.is_some() || scores.iter().any(Option::is_some);
                            let mut far_evidence = baseline;
                            let mut selected = select_direct_depth(
                                &candidates,
                                &scores,
                                baseline,
                                seed.is_some(),
                                direct_options,
                                geometry_mode,
                            );
                            if selected.is_none()
                                && let Some(best) =
                                    best_depth_candidate(&candidates, &scores, direct_options)
                                && scores[best].is_some_and(|score| {
                                    score.photometric_score >= MINIMUM_REGULARIZED_SCORE
                                })
                            {
                                let first = best.saturating_sub(3);
                                let last = (best + 3).min(candidates.len() - 1);
                                let wide_candidates = &candidates[first..=last];
                                let (wide_prepared_reference, wide_reference_rays) =
                                    prepare_physical_score_cache(
                                        inputs,
                                        reference_index,
                                        alignments,
                                        active_view_indices,
                                        p,
                                        wide_options,
                                        geometry_mode,
                                        &mut wide_reference_patch,
                                        &mut wide_far_scores,
                                        &mut depth_information,
                                    );
                                let wide_baseline = score_prepared_depth(
                                    inputs,
                                    reference_index,
                                    alignments,
                                    active_view_indices,
                                    p,
                                    None,
                                    wide_options,
                                    geometry_mode,
                                    wide_prepared_reference.then_some(&wide_reference_patch),
                                    wide_reference_rays.as_ref(),
                                    &wide_far_scores,
                                    &depth_information,
                                    &mut view_scores,
                                    &mut aggregate_scratch,
                                );
                                wide_scores.clear();
                                for &depth in wide_candidates {
                                    wide_scores.push(score_prepared_depth(
                                        inputs,
                                        reference_index,
                                        alignments,
                                        active_view_indices,
                                        p,
                                        Some(depth),
                                        wide_options,
                                        geometry_mode,
                                        wide_prepared_reference.then_some(&wide_reference_patch),
                                        wide_reference_rays.as_ref(),
                                        &wide_far_scores,
                                        &depth_information,
                                        &mut view_scores,
                                        &mut aggregate_scratch,
                                    ));
                                }
                                if wide_baseline.is_some() {
                                    far_evidence = wide_baseline;
                                }
                                selected = select_direct_depth(
                                    wide_candidates,
                                    &wide_scores,
                                    wide_baseline,
                                    seed.is_some(),
                                    wide_options,
                                    geometry_mode,
                                );
                            }
                            let far_score_floor = (direct_options.minimum_score - 0.03)
                                .max(MINIMUM_REGULARIZED_SCORE);
                            let node_far_supported = selected.is_none()
                                && far_evidence.is_some_and(|evidence| {
                                    evidence.photometric_score >= far_score_floor
                                });
                            nodes.push(selected);
                            tested.push(node_tested);
                            far_supported.push(node_far_supported);
                        }
                    }
                    (nodes, guidance, tested, far_supported)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("direct depth worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut nodes = Vec::with_capacity(columns * rows);
    let mut guidance = Vec::with_capacity(columns * rows);
    let mut tested = Vec::with_capacity(columns * rows);
    let mut far_supported = Vec::with_capacity(columns * rows);
    for (chunk_nodes, chunk_guidance, chunk_tested, chunk_far_supported) in chunks {
        nodes.extend(chunk_nodes);
        guidance.extend(chunk_guidance);
        tested.extend(chunk_tested);
        far_supported.extend(chunk_far_supported);
    }
    DirectDepthField {
        columns,
        rows,
        step,
        nodes,
        guidance,
        tested,
        far_supported,
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_physical_score_cache(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    active_view_indices: &[usize],
    centre: Vec2,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
    reference_patch: &mut PreparedReferencePatch,
    far_scores: &mut [Option<f32>],
    depth_information: &mut [f32],
) -> (bool, Option<LocalPatchReferenceRays>) {
    if !geometry_mode.is_physical() {
        return (false, None);
    }
    far_scores.fill(None);
    depth_information.fill(0.0);
    let prepared_reference =
        prepare_reference_patch(&inputs[reference_index], centre, options, reference_patch);
    let reference_rays = inputs[reference_index]
        .camera
        .map(|camera| local_patch_reference_rays(camera, centre));
    if prepared_reference
        && let Some(reference_camera) = inputs[reference_index].camera
        && let Some(reference_rays) = reference_rays.as_ref()
    {
        for &index in active_view_indices {
            let target = &inputs[index];
            let Some(target_camera) = target.camera else {
                continue;
            };
            depth_information[index] =
                physical_depth_information(reference_camera, target_camera, centre);
            let Some(projection) = local_patch_projection_from_rays(
                reference_camera,
                target_camera,
                &alignments[index].warp,
                centre,
                None,
                options,
                reference_rays,
            ) else {
                continue;
            };
            far_scores[index] =
                projected_patch_zncc_prepared(target, &projection, [0.0, 0.0], reference_patch);
        }
    }
    (prepared_reference, reference_rays)
}

#[allow(clippy::too_many_arguments)]
fn score_prepared_depth(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    active_view_indices: &[usize],
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
    reference_patch: Option<&PreparedReferencePatch>,
    reference_rays: Option<&LocalPatchReferenceRays>,
    far_scores: &[Option<f32>],
    depth_information: &[f32],
    view_scores: &mut Vec<ViewScore>,
    aggregate_scratch: &mut AggregateScratch,
) -> Option<AggregateEvidence> {
    score_views_prepared_into(
        inputs,
        reference_index,
        alignments,
        active_view_indices,
        centre,
        depth,
        options,
        geometry_mode,
        reference_patch,
        reference_rays,
        far_scores,
        depth_information,
        view_scores,
    );
    aggregate_with_scratch(view_scores, options, geometry_mode, aggregate_scratch)
}

#[allow(clippy::too_many_arguments)]
fn physical_visibility_memberships(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    field: &[Option<NodeDepth>],
    far_supported: &[bool],
    columns: usize,
    rows: usize,
    step: usize,
    options: &DepthOptions,
) -> (Vec<u64>, usize) {
    let words_per_node = inputs.len().div_ceil(u64::BITS as usize).max(1);
    let active_view_indices = (0..inputs.len())
        .filter(|&index| {
            index != reference_index
                && inputs[index].camera.is_some()
                && inputs[index].depth_evidence_enabled
        })
        .collect::<Vec<_>>();
    let worker_count = configured_worker_count(options.threads, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let chunks = thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(rows_per_worker)
            .map(|first_row| {
                let last_row = (first_row + rows_per_worker).min(rows);
                let active_view_indices = &active_view_indices;
                scope.spawn(move || {
                    let chunk_nodes = (last_row - first_row) * columns;
                    let mut memberships = vec![0u64; chunk_nodes * words_per_node];
                    let mut view_scores = Vec::with_capacity(active_view_indices.len());
                    let mut aggregate_scratch = AggregateScratch {
                        ordered: Vec::with_capacity(active_view_indices.len()),
                        positive_information: Vec::with_capacity(active_view_indices.len()),
                    };
                    let mut reference_patch =
                        PreparedReferencePatch::with_radius(options.patch_radius);
                    let mut far_scores = vec![None; inputs.len()];
                    let mut depth_information = vec![0.0f32; inputs.len()];
                    for row in first_row..last_row {
                        for column in 0..columns {
                            let global_index = row * columns + column;
                            let depth = if let Some(node) = field[global_index] {
                                Some(node.depth)
                            } else if far_supported[global_index] {
                                None
                            } else {
                                continue;
                            };
                            let p = [(column * step) as f64, (row * step) as f64];
                            let (prepared_reference, reference_rays) = prepare_physical_score_cache(
                                inputs,
                                reference_index,
                                alignments,
                                active_view_indices,
                                p,
                                options,
                                DepthGeometryMode::PhysicalRig,
                                &mut reference_patch,
                                &mut far_scores,
                                &mut depth_information,
                            );
                            score_views_prepared_into(
                                inputs,
                                reference_index,
                                alignments,
                                active_view_indices,
                                p,
                                depth,
                                options,
                                DepthGeometryMode::PhysicalRig,
                                prepared_reference.then_some(&reference_patch),
                                reference_rays.as_ref(),
                                &far_scores,
                                &depth_information,
                                &mut view_scores,
                            );
                            let local_index = (row - first_row) * columns + column;
                            let bits = &mut memberships
                                [local_index * words_per_node..(local_index + 1) * words_per_node];
                            physical_consensus_visibility_bits(
                                &view_scores,
                                options,
                                bits,
                                &mut aggregate_scratch,
                            );
                        }
                    }
                    memberships
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("physical visibility worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut memberships = Vec::with_capacity(columns * rows * words_per_node);
    for chunk in chunks {
        memberships.extend(chunk);
    }
    (memberships, words_per_node)
}

fn physical_consensus_visibility_bits(
    scores: &[ViewScore],
    options: &DepthOptions,
    bits: &mut [u64],
    scratch: &mut AggregateScratch,
) {
    bits.fill(0);
    if scores.len() < options.minimum_support {
        return;
    }
    scratch.ordered.clear();
    scratch
        .ordered
        .extend(scores.iter().map(|view| view.compatibility_score));
    scratch.ordered.sort_by(|left, right| right.total_cmp(left));
    let anchor_count = options.minimum_support.max(2).min(scratch.ordered.len());
    let anchor = scratch.ordered[..anchor_count].iter().sum::<f32>() / anchor_count as f32;
    let threshold = anchor - PHYSICAL_CONSENSUS_BAND;

    scratch.positive_information.clear();
    scratch.positive_information.extend(
        scores
            .iter()
            .map(|view| view.depth_information)
            .filter(|value| value.is_finite() && *value > 1.0e-6),
    );
    scratch.positive_information.sort_by(f32::total_cmp);
    let median_information = scratch
        .positive_information
        .get(scratch.positive_information.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .max(1.0e-6);

    let per_view_score_floor = (options.minimum_score - 0.10).max(MINIMUM_REGULARIZED_SCORE);
    let mut effective_support = 0.0f32;
    let mut total_weight = 0.0f32;
    for view in scores {
        let x = ((view.compatibility_score - threshold) / PHYSICAL_CONSENSUS_SOFTNESS)
            .clamp(-20.0, 20.0);
        let compatibility = 1.0 / (1.0 + (-x).exp());
        effective_support += compatibility;
        let relative_information = if view.depth_information > 1.0e-6 {
            view.depth_information / median_information
        } else {
            0.0
        };
        let information_weight = relative_information.clamp(
            PHYSICAL_MIN_INFORMATION_WEIGHT,
            PHYSICAL_MAX_INFORMATION_WEIGHT,
        );
        total_weight += compatibility * (0.75 + 0.25 * information_weight);
        if compatibility >= PHYSICAL_VISIBLE_COMPATIBILITY && view.score >= per_view_score_floor {
            let word = view.source_index / u64::BITS as usize;
            let bit = view.source_index % u64::BITS as usize;
            if let Some(value) = bits.get_mut(word) {
                *value |= 1u64 << bit;
            }
        }
    }
    if effective_support < options.minimum_support as f32 * 0.75 || total_weight <= 1.0e-6 {
        bits.fill(0);
    }
}

#[inline]
fn physical_membership_contains(
    memberships: &[u64],
    words_per_node: usize,
    node_index: usize,
    source_index: usize,
) -> bool {
    let word = source_index / u64::BITS as usize;
    let bit = source_index % u64::BITS as usize;
    memberships
        .get(node_index * words_per_node + word)
        .is_some_and(|value| value & (1u64 << bit) != 0)
}

fn reject_isolated_direct_depths(
    field: &mut [Option<NodeDepth>],
    guidance: &[f32],
    columns: usize,
    rows: usize,
    inverse_tolerance: f64,
) {
    let source = field.to_vec();
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let Some(node) = source[index] else {
                continue;
            };
            let inverse = 1.0 / node.depth;
            let mut agreeing = 0usize;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (x, y) = (column as i32 + dx, row as i32 + dy);
                    if x < 0 || y < 0 || x >= columns as i32 || y >= rows as i32 {
                        continue;
                    }
                    let neighbour_index = y as usize * columns + x as usize;
                    let Some(neighbour) = source[neighbour_index] else {
                        continue;
                    };
                    let image_edge = (guidance[index] - guidance[neighbour_index]).abs();
                    if image_edge <= 0.6
                        && (1.0 / neighbour.depth - inverse).abs() <= inverse_tolerance
                    {
                        agreeing += 1;
                    }
                }
            }
            let minimum = if node.confidence >= 0.75 { 1 } else { 2 };
            if agreeing < minimum {
                field[index] = None;
            }
        }
    }
}

/// Reject small connected islands of otherwise locally plausible depth.
///
/// Independent noise can accidentally produce two or three mutually agreeing
/// ZNCC matches on a textureless surface. Those islands pass a local neighbour
/// test but do not constitute a reproducible surface. Connectivity requires
/// both compatible inverse depth and no strong reference-image discontinuity.
/// This operation only removes measurements; it never completes a hole.
fn reject_small_direct_components(
    field: &mut [Option<NodeDepth>],
    guidance: &[f32],
    columns: usize,
    rows: usize,
    inverse_tolerance: f64,
    minimum_nodes: usize,
) {
    if minimum_nodes <= 1 {
        return;
    }
    let source = field.to_vec();
    let mut visited = vec![false; source.len()];
    for seed in 0..source.len() {
        if visited[seed] || source[seed].is_none() {
            continue;
        }
        visited[seed] = true;
        let mut pending = vec![seed];
        let mut component = Vec::new();
        while let Some(index) = pending.pop() {
            component.push(index);
            let column = index % columns;
            let row = index / columns;
            let inverse = 1.0 / source[index].expect("visited depth node").depth;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (x, y) = (column as i32 + dx, row as i32 + dy);
                    if x < 0 || y < 0 || x >= columns as i32 || y >= rows as i32 {
                        continue;
                    }
                    let neighbour = y as usize * columns + x as usize;
                    if visited[neighbour] {
                        continue;
                    }
                    let Some(neighbour_node) = source[neighbour] else {
                        continue;
                    };
                    let image_edge = (guidance[index] - guidance[neighbour]).abs();
                    let inverse_difference = (1.0 / neighbour_node.depth - inverse).abs();
                    if image_edge <= 0.6 && inverse_difference <= inverse_tolerance {
                        visited[neighbour] = true;
                        pending.push(neighbour);
                    }
                }
            }
        }
        if component.len() < minimum_nodes {
            for index in component {
                field[index] = None;
            }
        }
    }
}

/// Fit a local inverse-depth plane to coherent, directly measured neighbours.
/// The fit updates only an existing measurement and is clamped near that
/// measurement, so it removes quantisation/speckle without growing surfaces
/// into holes or across a competing depth layer.
fn fit_local_depth_planes(
    field: &mut [Option<NodeDepth>],
    guidance: &[f32],
    columns: usize,
    rows: usize,
    inverse_tolerance: f64,
) {
    let source = field.to_vec();
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let Some(mut node) = source[index] else {
                continue;
            };
            let centre_inverse = 1.0 / node.depth;
            let mut normal = [[0.0f64; 3]; 3];
            let mut rhs = [0.0f64; 3];
            let mut support = 0usize;
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (x, y) = (column as i32 + dx, row as i32 + dy);
                    if x < 0 || y < 0 || x >= columns as i32 || y >= rows as i32 {
                        continue;
                    }
                    let neighbour_index = y as usize * columns + x as usize;
                    let Some(neighbour) = source[neighbour_index] else {
                        continue;
                    };
                    let inverse = 1.0 / neighbour.depth;
                    let image_edge = (guidance[index] - guidance[neighbour_index]).abs();
                    if image_edge > 0.6 || (inverse - centre_inverse).abs() > inverse_tolerance {
                        continue;
                    }
                    let spatial = (-(dx * dx + dy * dy) as f64 / 6.0).exp();
                    let weight = f64::from(neighbour.confidence)
                        * spatial
                        * (-2.5 * f64::from(image_edge)).exp();
                    let basis = [1.0, dx as f64, dy as f64];
                    for r in 0..3 {
                        rhs[r] += weight * basis[r] * inverse;
                        for c in 0..3 {
                            normal[r][c] += weight * basis[r] * basis[c];
                        }
                    }
                    support += 1;
                }
            }
            if support < 6 {
                continue;
            }
            let Some(inverse) = crate::math::inverse(&normal)
                .map(|matrix| crate::math::mul_vec(&matrix, rhs)[0])
                .filter(|inverse| inverse.is_finite() && *inverse > 0.0)
            else {
                continue;
            };
            let maximum_adjustment = inverse_tolerance * 0.25;
            let inverse = inverse.clamp(
                centre_inverse - maximum_adjustment,
                centre_inverse + maximum_adjustment,
            );
            node.depth = 1.0 / inverse;
            field[index] = Some(node);
        }
    }
}

fn nearest_coarse_depth(
    coarse: &[Option<NodeDepth>],
    columns: usize,
    rows: usize,
    step: usize,
    pixel: Vec2,
) -> Option<NodeDepth> {
    let column = (pixel[0] / step as f64)
        .round()
        .clamp(0.0, (columns - 1) as f64) as usize;
    let row = (pixel[1] / step as f64)
        .round()
        .clamp(0.0, (rows - 1) as f64) as usize;
    coarse[row * columns + column]
}

fn direct_depth_candidates_into(
    seed: Option<f64>,
    inverse_step: f64,
    options: &DepthOptions,
    depths: &mut Vec<f64>,
) {
    depths.clear();
    let far_inverse = 1.0 / options.far_depth;
    let near_inverse = 1.0 / options.near_depth;
    match seed {
        Some(depth) => {
            for offset in -6..=6 {
                let inverse = (1.0 / depth + offset as f64 * inverse_step * 0.5)
                    .clamp(far_inverse, near_inverse);
                let candidate = 1.0 / inverse;
                if depths.last().is_none_or(|last: &f64| {
                    (1.0 / *last - 1.0 / candidate).abs() > inverse_step * 0.1
                }) {
                    depths.push(candidate);
                }
            }
        }
        None => {
            depths.extend(inverse_depth_samples(
                options.near_depth,
                options.far_depth,
                32,
            ));
            depths.reverse();
        }
    }
}

fn direct_depth_refinement_candidates_into(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    centre: Vec2,
    depths: &[f64],
    best: usize,
    additions: &mut Vec<f64>,
) {
    additions.clear();
    for neighbour in [
        best.checked_sub(1),
        (best + 1 < depths.len()).then_some(best + 1),
    ]
    .into_iter()
    .flatten()
    {
        if maximum_projected_depth_motion(
            inputs,
            reference_index,
            centre,
            depths[best],
            depths[neighbour],
        ) <= DIRECT_MAX_PROJECTED_STEP_PX
        {
            continue;
        }
        let inverse = (1.0 / depths[best] + 1.0 / depths[neighbour]) * 0.5;
        let depth = 1.0 / inverse;
        if depth.is_finite() && depth > 0.0 {
            additions.push(depth);
        }
    }
}

fn maximum_projected_depth_motion(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    centre: Vec2,
    first_depth: f64,
    second_depth: f64,
) -> f64 {
    let Some(reference) = inputs.get(reference_index).and_then(|input| input.camera) else {
        return 0.0;
    };
    inputs
        .iter()
        .enumerate()
        .filter(|(index, input)| {
            *index != reference_index && input.depth_evidence_enabled && input.camera.is_some()
        })
        .filter_map(|(_, input)| {
            let target = input.camera?;
            let first = target.map_from(reference, centre, first_depth)?;
            let second = target.map_from(reference, centre, second_depth)?;
            if !target.contains(first) || !target.contains(second) {
                return None;
            }
            let delta = [second[0] - first[0], second[1] - first[1]];
            Some((delta[0] * delta[0] + delta[1] * delta[1]).sqrt())
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
fn merge_depth_scores(
    depths: &mut Vec<f64>,
    scores: &mut Vec<Option<AggregateEvidence>>,
    additions: Vec<(f64, Option<AggregateEvidence>)>,
) {
    let mut scratch = Vec::with_capacity(depths.len() + additions.len());
    merge_depth_scores_with_scratch(depths, scores, &additions, &mut scratch);
}

fn merge_depth_scores_with_scratch(
    depths: &mut Vec<f64>,
    scores: &mut Vec<Option<AggregateEvidence>>,
    additions: &[(f64, Option<AggregateEvidence>)],
    scratch: &mut Vec<(f64, Option<AggregateEvidence>)>,
) {
    scratch.clear();
    scratch.extend(depths.iter().copied().zip(scores.iter().copied()));
    scratch.extend_from_slice(additions);
    scratch.sort_by(|left, right| (1.0 / left.0).total_cmp(&(1.0 / right.0)));
    scratch.dedup_by(|left, right| (1.0 / left.0 - 1.0 / right.0).abs() <= f64::EPSILON);
    depths.clear();
    scores.clear();
    depths.extend(scratch.iter().map(|(depth, _)| *depth));
    scores.extend(scratch.iter().map(|(_, score)| *score));
}

fn select_direct_depth(
    depths: &[f64],
    scores: &[Option<AggregateEvidence>],
    baseline: Option<AggregateEvidence>,
    seeded: bool,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
) -> Option<NodeDepth> {
    let far_inverse = 1.0 / options.far_depth;
    let inverse_range = 1.0 / options.near_depth - far_inverse;
    let objective_at = |index: usize| {
        let depth = depths[index];
        let evidence = scores[index]?;
        let near_fraction = ((1.0 / depth - far_inverse) / inverse_range).clamp(0.0, 1.0);
        Some(evidence.ranking_score - NEAR_DEPTH_PRIOR * near_fraction as f32)
    };

    // The previous implementation materialised and sorted every valid finite
    // hypothesis. Only the best objective and the strongest non-neighbouring
    // competitor are needed, so two linear scans preserve the same selection
    // while eliminating a per-node allocation and sort. Ties keep the earliest
    // candidate, matching stable sort behaviour.
    let mut best = None::<(usize, f32)>;
    for index in 0..depths.len() {
        let Some(objective) = objective_at(index) else {
            continue;
        };
        if best.is_none_or(|(_, current)| objective.total_cmp(&current).is_gt()) {
            best = Some((index, objective));
        }
    }
    let (best_index, best_objective) = best?;
    let depth = depths[best_index];
    let score = scores[best_index]?.photometric_score;
    let mut competing = None::<f32>;
    for index in 0..depths.len() {
        if index.abs_diff(best_index) <= 1 {
            continue;
        }
        let Some(objective) = objective_at(index) else {
            continue;
        };
        if competing.is_none_or(|current| objective.total_cmp(&current).is_gt()) {
            competing = Some(objective);
        }
    }
    let competing = competing.unwrap_or(best_objective);
    let margin = (best_objective - competing).max(0.0);
    let paired_improvement = scores[best_index]
        .and_then(|evidence| Some(evidence.paired_photometric_score? - evidence.paired_far_score?));
    let comparable_improvement = if geometry_mode.is_physical() {
        // Never compare a finite physical consensus against a separately
        // aggregated far consensus: that would reintroduce camera-population
        // bias whenever the two hypotheses are visible in different views.
        paired_improvement
    } else {
        baseline.map(|baseline| score - baseline.photometric_score)
    };
    let improvement = comparable_improvement.unwrap_or(0.0);
    let score_floor = (options.minimum_score - 0.03).max(MINIMUM_REGULARIZED_SCORE);
    let required_margin = if seeded {
        options.minimum_margin * 0.6
    } else {
        options.minimum_margin
    };
    // A finite label being unique among other finite labels is insufficient:
    // distant textured surfaces can have an extremely shallow cost curve and
    // acquire a coherent but fictitious finite depth. Require the candidate
    // to improve measurably on the active far/baseline hypothesis. When that
    // patch is unavailable, uniqueness remains the only usable evidence.
    let improves_global = comparable_improvement
        .is_none_or(|improvement| improvement >= options.minimum_improvement * 0.5);
    let supported = score >= score_floor
        && improves_global
        && (margin >= required_margin
            || comparable_improvement
                .is_some_and(|improvement| improvement >= options.minimum_improvement));
    if !supported {
        return None;
    }
    let score_confidence =
        ((score - score_floor) / (1.0 - score_floor).max(1.0e-3)).clamp(0.0, 1.0);
    let margin_confidence = (margin / required_margin.max(1.0e-3)).clamp(0.0, 1.0);
    Some(NodeDepth {
        depth,
        confidence: (0.30 + 0.45 * score_confidence + 0.25 * margin_confidence).clamp(0.0, 1.0),
        improvement,
        regularized: false,
    })
}

fn best_depth_candidate(
    depths: &[f64],
    scores: &[Option<AggregateEvidence>],
    options: &DepthOptions,
) -> Option<usize> {
    let far_inverse = 1.0 / options.far_depth;
    let inverse_range = 1.0 / options.near_depth - far_inverse;
    depths
        .iter()
        .zip(scores)
        .enumerate()
        .filter_map(|(index, (&depth, &evidence))| {
            let evidence = evidence?;
            let near_fraction = ((1.0 / depth - far_inverse) / inverse_range).clamp(0.0, 1.0);
            Some((
                index,
                evidence.ranking_score - NEAR_DEPTH_PRIOR * near_fraction as f32,
            ))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

fn build_cost_volume(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    columns: usize,
    rows: usize,
    step: usize,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
) -> CostVolume {
    let mut finite_depths =
        inverse_depth_samples(options.near_depth, options.far_depth, options.planes);
    finite_depths.reverse();
    let labels = std::iter::once(None)
        .chain(finite_depths.into_iter().map(Some))
        .collect::<Vec<_>>();
    let label_count = labels.len();
    let node_count = columns * rows;
    let active_view_indices = (0..inputs.len())
        .filter(|&index| {
            index != reference_index
                && inputs[index].camera.is_some()
                && inputs[index].depth_evidence_enabled
                && (geometry_mode.is_physical() || alignments[index].report.accepted)
        })
        .collect::<Vec<_>>();
    let worker_count = configured_worker_count(options.threads, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let chunks = thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(rows_per_worker)
            .map(|first_row| {
                let last_row = (first_row + rows_per_worker).min(rows);
                let labels = &labels;
                let active_view_indices = &active_view_indices;
                scope.spawn(move || {
                    let chunk_nodes = (last_row - first_row) * columns;
                    let mut scores = Vec::with_capacity(chunk_nodes * label_count);
                    let mut paired_improvements = Vec::with_capacity(chunk_nodes * label_count);
                    let mut costs = Vec::with_capacity(chunk_nodes * label_count);
                    let mut guidance = Vec::with_capacity(chunk_nodes);
                    let mut tested = Vec::with_capacity(chunk_nodes);
                    let mut view_scores = Vec::with_capacity(active_view_indices.len());
                    let mut aggregate_scratch = AggregateScratch {
                        ordered: Vec::with_capacity(active_view_indices.len()),
                        positive_information: Vec::with_capacity(active_view_indices.len()),
                    };
                    let mut reference_patch =
                        PreparedReferencePatch::with_radius(options.patch_radius);
                    let mut far_scores = vec![None; inputs.len()];
                    let mut depth_information = vec![0.0f32; inputs.len()];
                    for row in first_row..last_row {
                        for column in 0..columns {
                            let p = [(column * step) as f64, (row * step) as f64];
                            guidance.push(reference_guidance(&inputs[reference_index], p));

                            let prepared_reference = if geometry_mode.is_physical() {
                                prepare_reference_patch(
                                    &inputs[reference_index],
                                    p,
                                    options,
                                    &mut reference_patch,
                                )
                            } else {
                                false
                            };
                            let reference_rays = if geometry_mode.is_physical() {
                                inputs[reference_index]
                                    .camera
                                    .map(|camera| local_patch_reference_rays(camera, p))
                            } else {
                                None
                            };
                            if geometry_mode.is_physical() {
                                far_scores.fill(None);
                                depth_information.fill(0.0);
                                if prepared_reference {
                                    if let Some(reference_camera) = inputs[reference_index].camera {
                                        for &index in active_view_indices {
                                            let target = &inputs[index];
                                            let Some(target_camera) = target.camera else {
                                                continue;
                                            };
                                            depth_information[index] = physical_depth_information(
                                                reference_camera,
                                                target_camera,
                                                p,
                                            );
                                            let Some(reference_rays) = reference_rays.as_ref()
                                            else {
                                                continue;
                                            };
                                            let Some(projection) = local_patch_projection_from_rays(
                                                reference_camera,
                                                target_camera,
                                                &alignments[index].warp,
                                                p,
                                                None,
                                                options,
                                                reference_rays,
                                            ) else {
                                                continue;
                                            };
                                            far_scores[index] = projected_patch_zncc_prepared(
                                                target,
                                                &projection,
                                                [0.0, 0.0],
                                                &reference_patch,
                                            );
                                        }
                                    }
                                }
                            }

                            let mut node_tested = false;
                            for (label, &depth) in labels.iter().enumerate() {
                                score_views_prepared_into(
                                    inputs,
                                    reference_index,
                                    alignments,
                                    active_view_indices,
                                    p,
                                    depth,
                                    options,
                                    geometry_mode,
                                    prepared_reference.then_some(&reference_patch),
                                    reference_rays.as_ref(),
                                    &far_scores,
                                    &depth_information,
                                    &mut view_scores,
                                );
                                let evidence = aggregate_with_scratch(
                                    &view_scores,
                                    options,
                                    geometry_mode,
                                    &mut aggregate_scratch,
                                );
                                let photometric_score = evidence
                                    .map(|evidence| evidence.photometric_score)
                                    .unwrap_or(f32::NAN);
                                let ranking_score = evidence
                                    .map(|evidence| evidence.ranking_score)
                                    .unwrap_or(f32::NAN);
                                let paired_improvement = evidence
                                    .and_then(|evidence| {
                                        Some(
                                            evidence.paired_photometric_score?
                                                - evidence.paired_far_score?,
                                        )
                                    })
                                    .unwrap_or(f32::NAN);
                                node_tested |= photometric_score.is_finite();
                                scores.push(photometric_score);
                                paired_improvements.push(paired_improvement);
                                costs.push(if ranking_score.is_finite() {
                                    let near_prior = if label == 0 {
                                        0.0
                                    } else {
                                        NEAR_DEPTH_PRIOR * (label - 1) as f32
                                            / (label_count - 2) as f32
                                    };
                                    (1.0 - ranking_score).clamp(0.0, 2.0) + near_prior
                                } else {
                                    MISSING_COST
                                });
                            }
                            tested.push(node_tested);
                        }
                    }
                    (scores, paired_improvements, costs, guidance, tested)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("depth cost worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut scores = Vec::with_capacity(node_count * label_count);
    let mut paired_improvements = Vec::with_capacity(node_count * label_count);
    let mut costs = Vec::with_capacity(node_count * label_count);
    let mut guidance = Vec::with_capacity(node_count);
    let mut tested = Vec::with_capacity(node_count);
    for (chunk_scores, chunk_paired_improvements, chunk_costs, chunk_guidance, chunk_tested) in
        chunks
    {
        scores.extend(chunk_scores);
        paired_improvements.extend(chunk_paired_improvements);
        costs.extend(chunk_costs);
        guidance.extend(chunk_guidance);
        tested.extend(chunk_tested);
    }
    CostVolume {
        physical_geometry: geometry_mode.is_physical(),
        labels,
        scores,
        paired_improvements,
        costs,
        guidance,
        tested,
    }
}

fn semi_global_costs(
    data: &[f32],
    guidance: &[f32],
    columns: usize,
    rows: usize,
    labels: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; data.len()];
    for row in 0..rows {
        let forward = (0..columns)
            .map(|column| row * columns + column)
            .collect::<Vec<_>>();
        let reverse = forward.iter().rev().copied().collect::<Vec<_>>();
        accumulate_path(data, guidance, &mut output, labels, &forward);
        accumulate_path(data, guidance, &mut output, labels, &reverse);
    }
    for column in 0..columns {
        let forward = (0..rows)
            .map(|row| row * columns + column)
            .collect::<Vec<_>>();
        let reverse = forward.iter().rev().copied().collect::<Vec<_>>();
        accumulate_path(data, guidance, &mut output, labels, &forward);
        accumulate_path(data, guidance, &mut output, labels, &reverse);
    }
    let down_right = diagonal_paths(columns, rows, false);
    let down_left = diagonal_paths(columns, rows, true);
    for path in down_right.iter().chain(&down_left) {
        accumulate_path(data, guidance, &mut output, labels, path);
        let reverse = path.iter().rev().copied().collect::<Vec<_>>();
        accumulate_path(data, guidance, &mut output, labels, &reverse);
    }
    output
}

fn diagonal_paths(columns: usize, rows: usize, mirrored: bool) -> Vec<Vec<usize>> {
    let mut paths = Vec::with_capacity(columns + rows - 1);
    for start_column in 0..columns {
        let mut path = Vec::new();
        let mut column = start_column as i32;
        let mut row = 0usize;
        while column >= 0 && column < columns as i32 && row < rows {
            path.push(row * columns + column as usize);
            column += if mirrored { -1 } else { 1 };
            row += 1;
        }
        paths.push(path);
    }
    for start_row in 1..rows {
        let mut path = Vec::new();
        let mut column = if mirrored { columns as i32 - 1 } else { 0 };
        let mut row = start_row;
        while column >= 0 && column < columns as i32 && row < rows {
            path.push(row * columns + column as usize);
            column += if mirrored { -1 } else { 1 };
            row += 1;
        }
        paths.push(path);
    }
    paths
}

fn accumulate_path(
    data: &[f32],
    guidance: &[f32],
    output: &mut [f32],
    labels: usize,
    path: &[usize],
) {
    let mut previous = vec![0.0; labels];
    let mut current = vec![0.0; labels];
    for (path_index, &node) in path.iter().enumerate() {
        let offset = node * labels;
        if path_index == 0 {
            previous.copy_from_slice(&data[offset..offset + labels]);
            for label in 0..labels {
                output[offset + label] += previous[label];
            }
            continue;
        }
        let previous_node = path[path_index - 1];
        let edge = (guidance[node] - guidance[previous_node]).abs();
        let edge_scale = 1.0 / (1.0 + 4.0 * edge);
        let p1 = SGM_SMALL_PENALTY * edge_scale.max(0.35);
        let p2 = SGM_LARGE_PENALTY * edge_scale.max(0.12);
        let minimum_previous = previous.iter().copied().fold(f32::INFINITY, f32::min);
        for label in 0..labels {
            let same = previous[label];
            let lower = if label > 0 {
                previous[label - 1] + p1
            } else {
                f32::INFINITY
            };
            let higher = if label + 1 < labels {
                previous[label + 1] + p1
            } else {
                f32::INFINITY
            };
            let jump = minimum_previous + p2;
            current[label] =
                data[offset + label] + same.min(lower).min(higher).min(jump) - minimum_previous;
            output[offset + label] += current[label];
        }
        std::mem::swap(&mut previous, &mut current);
    }
}

fn select_depths(
    volume: &CostVolume,
    regularised: &[f32],
    options: &DepthOptions,
) -> (Vec<Option<NodeDepth>>, Vec<bool>) {
    let labels = volume.labels.len();
    let mut field = vec![None; volume.tested.len()];
    let mut fillable = vec![false; volume.tested.len()];
    for (node, &tested) in volume.tested.iter().enumerate() {
        if !tested {
            continue;
        }
        let offset = node * labels;
        // The active far/baseline projection is a useful photometric baseline
        // but not a finite depth hypothesis. Select among finite calibrated planes
        // and use the baseline score to reject only a clearly worse
        // reconstruction. This lets SGM infer depth throughout
        // weakly textured but still observable surfaces instead of retaining
        // only isolated high-contrast edges.
        let Some((best_label, best_cost)) = regularised[offset + 1..offset + labels]
            .iter()
            .copied()
            .enumerate()
            .map(|(label, cost)| (label + 1, cost))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        else {
            continue;
        };
        let depth = sublabel_depth(
            &volume.labels,
            &regularised[offset..offset + labels],
            best_label,
        );
        let score = volume.scores[offset + best_label];
        if !score.is_finite() || score < MINIMUM_REGULARIZED_SCORE {
            fillable[node] = true;
            continue;
        }
        let competing = regularised[offset + 1..offset + labels]
            .iter()
            .copied()
            .enumerate()
            .map(|(label, cost)| (label + 1, cost))
            .filter(|(label, _)| label.abs_diff(best_label) > 2)
            .map(|(_, cost)| cost)
            .min_by(f32::total_cmp)
            .unwrap_or(best_cost);
        let margin = ((competing - best_cost) / SGM_DIRECTIONS).max(0.0);
        let baseline = volume.scores[offset];
        let paired_improvement = volume.paired_improvements[offset + best_label];
        let comparable_improvement = if paired_improvement.is_finite() {
            Some(paired_improvement)
        } else if !volume.physical_geometry && baseline.is_finite() {
            Some(score - baseline)
        } else {
            None
        };
        let improvement = comparable_improvement.unwrap_or(0.0);
        if comparable_improvement
            .is_some_and(|improvement| improvement < -MAXIMUM_REGULARIZED_BASELINE_LOSS)
        {
            continue;
        }
        let measured = score >= options.minimum_score
            && comparable_improvement
                .is_none_or(|improvement| improvement >= options.minimum_improvement * 0.5)
            && (margin >= options.minimum_margin
                || comparable_improvement
                    .is_some_and(|improvement| improvement >= options.minimum_improvement));
        let score_confidence = ((score - options.minimum_score)
            / (1.0 - options.minimum_score).max(1.0e-3))
        .clamp(0.0, 1.0);
        let margin_confidence = (margin / options.minimum_margin.max(1.0e-3)).clamp(0.0, 1.0);
        let improvement_confidence =
            (improvement / options.minimum_improvement.max(1.0e-3)).clamp(0.0, 1.0);
        field[node] = Some(NodeDepth {
            depth,
            confidence: if measured {
                (0.30 + 0.45 * score_confidence + 0.25 * margin_confidence).clamp(0.0, 1.0)
            } else {
                (0.10
                    + 0.25 * score_confidence
                    + 0.15 * margin_confidence
                    + 0.10 * improvement_confidence)
                    .clamp(0.10, 0.55)
            },
            improvement,
            regularized: !measured,
        });
    }
    (field, fillable)
}

fn sublabel_depth(labels: &[Option<f64>], costs: &[f32], best_label: usize) -> f64 {
    let centre_depth = labels[best_label].expect("finite depth label");
    if best_label <= 1 || best_label + 1 >= labels.len() {
        return centre_depth;
    }
    let (left, centre, right) = (
        costs[best_label - 1],
        costs[best_label],
        costs[best_label + 1],
    );
    let curvature = left - 2.0 * centre + right;
    if !curvature.is_finite() || curvature <= 1.0e-6 {
        return centre_depth;
    }
    let offset = (0.5 * (left - right) / curvature).clamp(-0.75, 0.75) as f64;
    let lower_inverse = 1.0 / labels[best_label - 1].expect("finite lower label");
    let upper_inverse = 1.0 / labels[best_label + 1].expect("finite upper label");
    let inverse_step = (upper_inverse - lower_inverse) * 0.5;
    1.0 / (1.0 / centre_depth + offset * inverse_step)
}

fn complete_depth_field(
    field: &mut [Option<NodeDepth>],
    guidance: &[f32],
    fillable: &[bool],
    columns: usize,
    rows: usize,
    options: &DepthOptions,
) {
    for _ in 0..options.completion_iterations {
        let previous = field.to_vec();
        let mut changed = false;
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                if previous[index].is_some() || !fillable[index] {
                    continue;
                }
                let mut candidates = Vec::new();
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (x, y) = (column as i32 + dx, row as i32 + dy);
                        if x < 0 || y < 0 || x >= columns as i32 || y >= rows as i32 {
                            continue;
                        }
                        let neighbour_index = y as usize * columns + x as usize;
                        let Some(neighbour) = previous[neighbour_index] else {
                            continue;
                        };
                        let edge = (guidance[index] - guidance[neighbour_index]).abs();
                        if edge > 0.35 {
                            continue;
                        }
                        let spatial = if dx != 0 && dy != 0 { 0.707 } else { 1.0 };
                        let weight = neighbour.confidence * spatial * (-3.0 * edge).exp();
                        if weight >= 0.05 {
                            candidates.push((1.0 / neighbour.depth, weight));
                        }
                    }
                }
                if candidates.len() < options.minimum_neighbour_support {
                    continue;
                }
                candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
                let total_weight = candidates.iter().map(|candidate| candidate.1).sum::<f32>();
                let mut accumulated = 0.0;
                let inverse_depth = candidates
                    .iter()
                    .find_map(|&(inverse_depth, weight)| {
                        accumulated += weight;
                        (accumulated >= total_weight * 0.5).then_some(inverse_depth)
                    })
                    .unwrap_or(candidates[candidates.len() / 2].0);
                let confidence = (candidates
                    .iter()
                    .map(|candidate| candidate.1)
                    .fold(0.0, f32::max)
                    * 0.92)
                    .clamp(0.15, 0.75);
                field[index] = Some(NodeDepth {
                    depth: 1.0 / inverse_depth,
                    confidence,
                    improvement: 0.0,
                    regularized: true,
                });
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn score_views_prepared_into(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    active_view_indices: &[usize],
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
    reference_patch: Option<&PreparedReferencePatch>,
    reference_rays: Option<&LocalPatchReferenceRays>,
    far_scores: &[Option<f32>],
    depth_information: &[f32],
    output: &mut Vec<ViewScore>,
) {
    output.clear();
    for &index in active_view_indices {
        let view = match geometry_mode {
            DepthGeometryMode::PhysicalRig => {
                let Some(reference_patch) = reference_patch else {
                    continue;
                };
                let Some(reference_rays) = reference_rays else {
                    continue;
                };
                let Some(reference_camera) = inputs[reference_index].camera else {
                    continue;
                };
                let Some(target_camera) = inputs[index].camera else {
                    continue;
                };
                let score = match depth {
                    None => far_scores[index],
                    Some(depth) => projected_patch_zncc_from_prepared_rays(
                        &inputs[reference_index],
                        &inputs[index],
                        &alignments[index].warp,
                        centre,
                        Some(depth),
                        [0.0, 0.0],
                        options,
                        reference_patch,
                        reference_rays,
                    ),
                };
                let Some(score) = score else {
                    continue;
                };
                let far_score = depth.and_then(|_| far_scores[index]);
                let compatibility_score =
                    far_score.map_or(score, |baseline| (score - baseline).clamp(-1.0, 1.0));
                let information = if depth_information[index] > 0.0 {
                    depth_information[index]
                } else {
                    physical_depth_information(reference_camera, target_camera, centre)
                };
                ViewScore {
                    source_index: index,
                    score,
                    far_score,
                    compatibility_score,
                    depth_information: information,
                }
            }
            DepthGeometryMode::WarpSeeded => {
                let Some(mut view) = score_one_view_warp_seeded(
                    &inputs[reference_index],
                    &inputs[index],
                    &alignments[index].warp,
                    centre,
                    depth,
                    options.patch_radius,
                ) else {
                    continue;
                };
                view.source_index = index;
                view
            }
        };
        output.push(view);
    }
}

/// Refine the shared multiview depth for one target camera, then permit a
/// tiny image-space residual around the proposal-guided physical projection.
/// This final residual is deliberately local and only absorbs
/// sub-pixel calibration/PSF mismatch after shared depth has been solved.
fn refine_one_view_physical(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    measured_proposal: &Warp,
    centre: Vec2,
    shared_depth: f64,
    options: &DepthOptions,
) -> Option<ViewRefinement> {
    let reference_camera = reference.camera?;
    let target_camera = target.camera?;
    let projection = local_patch_projection(
        reference_camera,
        target_camera,
        measured_proposal,
        centre,
        Some(shared_depth),
        options,
    )?;
    let depth_point = [
        projection.target_centre[0] as f32,
        projection.target_centre[1] as f32,
    ];
    let mut reference_patch = PreparedReferencePatch::with_radius(options.patch_radius);
    if !prepare_reference_patch(reference, centre, options, &mut reference_patch) {
        return None;
    }
    let depth_score =
        projected_patch_zncc_prepared(target, &projection, [0.0, 0.0], &reference_patch)?;

    // Visibility is decided at the exact shared multiview depth before this
    // function is called. Only then may a visible view absorb a very small
    // residual image-space error. Per-camera depth search is deliberately not
    // allowed here: otherwise an occluded camera could jump to a different
    // scene layer and appear to validate the shared surface. The physical
    // projection and reference patch are invariant across these residuals.
    let mut best = (depth_score, depth_score, depth_point);
    for dy in [-1.5f32, 0.0, 1.5] {
        for dx in [-1.5f32, 0.0, 1.5] {
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let Some(score) = projected_patch_zncc_prepared(
                target,
                &projection,
                [f64::from(dx), f64::from(dy)],
                &reference_patch,
            ) else {
                continue;
            };
            let residual_sq = dx * dx + dy * dy;
            let objective = score - residual_sq * 0.002;
            if objective > best.0 {
                best = (objective, score, [depth_point[0] + dx, depth_point[1] + dy]);
            }
        }
    }
    Some(ViewRefinement {
        score: best.1,
        point: best.2,
    })
}

fn score_one_view_warp_seeded(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    global: &Warp,
    centre: Vec2,
    depth: Option<f64>,
    patch_radius: usize,
) -> Option<ViewScore> {
    let reference_camera = reference.camera?;
    let target_camera = target.camera?;
    let global_centre = global.map(centre[0] as f32, centre[1] as f32)?;
    let mapped_centre = match depth {
        Some(depth) => {
            map_at_depth_warp_seeded(reference_camera, target_camera, global, centre, depth)?
        }
        None => [f64::from(global_centre[0]), f64::from(global_centre[1])],
    };
    let delta = [
        mapped_centre[0] - f64::from(global_centre[0]),
        mapped_centre[1] - f64::from(global_centre[1]),
    ];
    if delta[0].abs() > 128.0 || delta[1].abs() > 128.0 {
        return None;
    }
    let score = warped_patch_zncc(
        ViewPair {
            reference: reference.luminance,
            target: target.luminance,
            global,
        },
        centre,
        delta,
        patch_radius,
    )?;
    Some(ViewScore {
        source_index: usize::MAX,
        score,
        far_score: None,
        compatibility_score: score,
        depth_information: 1.0,
    })
}

fn refine_one_view_warp_seeded(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    global: &Warp,
    centre: Vec2,
    shared_depth: f64,
    inverse_step: f64,
    options: &DepthOptions,
) -> Option<ViewRefinement> {
    let reference_camera = reference.camera?;
    let target_camera = target.camera?;
    let global_point = global.map(centre[0] as f32, centre[1] as f32)?;
    let min_inverse = 1.0 / options.far_depth;
    let max_inverse = 1.0 / options.near_depth;
    let shared_inverse = 1.0 / shared_depth;
    let mut best: Option<(f32, f32, [f32; 2])> = None;

    for label_offset in [-1.0f64, -0.5, 0.0, 0.5, 1.0] {
        let inverse =
            (shared_inverse + label_offset * inverse_step).clamp(min_inverse, max_inverse);
        let Some(point) = map_at_depth_warp_seeded(
            reference_camera,
            target_camera,
            global,
            centre,
            1.0 / inverse,
        ) else {
            continue;
        };
        let point = [point[0] as f32, point[1] as f32];
        let Some(score) = score_one_view_at_point_warp_seeded(
            reference,
            target,
            global,
            centre,
            global_point,
            point,
            options.patch_radius,
        ) else {
            continue;
        };
        let objective = score - label_offset.abs() as f32 * 0.002;
        if best.is_none_or(|(best_objective, _, _)| objective > best_objective) {
            best = Some((objective, score, point));
        }
    }

    let (_, depth_score, depth_point) = best?;
    let mut best = (depth_score, depth_score, depth_point);
    for dy in [-1.5f32, 0.0, 1.5] {
        for dx in [-1.5f32, 0.0, 1.5] {
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let point = [depth_point[0] + dx, depth_point[1] + dy];
            let Some(score) = score_one_view_at_point_warp_seeded(
                reference,
                target,
                global,
                centre,
                global_point,
                point,
                options.patch_radius,
            ) else {
                continue;
            };
            let residual_sq = dx * dx + dy * dy;
            let objective = score - residual_sq * 0.002;
            if objective > best.0 {
                best = (objective, score, point);
            }
        }
    }
    Some(ViewRefinement {
        score: best.1,
        point: best.2,
    })
}

fn score_one_view_at_point_warp_seeded(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    global: &Warp,
    centre: Vec2,
    global_point: [f32; 2],
    point: [f32; 2],
    patch_radius: usize,
) -> Option<f32> {
    let delta = [
        f64::from(point[0] - global_point[0]),
        f64::from(point[1] - global_point[1]),
    ];
    if delta[0].abs() > 128.0 || delta[1].abs() > 128.0 {
        return None;
    }
    warped_patch_zncc(
        ViewPair {
            reference: reference.luminance,
            target: target.luminance,
            global,
        },
        centre,
        delta,
        patch_radius,
    )
}

fn map_at_depth_warp_seeded(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    global: &Warp,
    pixel: Vec2,
    depth: f64,
) -> Option<Vec2> {
    let global_q = global.map(pixel[0] as f32, pixel[1] as f32)?;
    let corrected_reference = reference_camera.map_from(
        target_camera,
        [f64::from(global_q[0]), f64::from(global_q[1])],
        INFINITY_DEPTH,
    )?;
    let q = target_camera.map_from(reference_camera, corrected_reference, depth)?;
    (q[0].is_finite() && q[1].is_finite()).then_some(q)
}

fn warped_patch_zncc(pair: ViewPair<'_>, centre: Vec2, delta: Vec2, radius: usize) -> Option<f32> {
    let centre_reference = pair.reference.sample(
        ((centre[0] - 0.5) * 0.5) as f32,
        ((centre[1] - 0.5) * 0.5) as f32,
    )?;
    let sigma = (radius as f32 * 0.75).max(1.0);
    let mut count = 0.0f32;
    let mut sum_reference = 0.0f32;
    let mut sum_target = 0.0f32;
    let mut sum_reference_sq = 0.0f32;
    let mut sum_target_sq = 0.0f32;
    let mut sum_product = 0.0f32;
    for dy in -(radius as isize)..=radius as isize {
        for dx in -(radius as isize)..=radius as isize {
            let p = [centre[0] + dx as f64 * 2.0, centre[1] + dy as f64 * 2.0];
            let reference_value = pair
                .reference
                .sample(((p[0] - 0.5) * 0.5) as f32, ((p[1] - 0.5) * 0.5) as f32)?;
            let global_q = pair.global.map(p[0] as f32, p[1] as f32)?;
            let q = [
                f64::from(global_q[0]) + delta[0],
                f64::from(global_q[1]) + delta[1],
            ];
            let target_value = pair
                .target
                .sample(((q[0] - 0.5) * 0.5) as f32, ((q[1] - 0.5) * 0.5) as f32)?;
            let distance_sq = (dx * dx + dy * dy) as f32;
            let spatial = (-distance_sq / (2.0 * sigma * sigma)).exp();
            let range = (-1.2 * (reference_value - centre_reference).abs()).exp();
            let weight = spatial * range;
            count += weight;
            sum_reference += weight * reference_value;
            sum_target += weight * target_value;
            sum_reference_sq += weight * reference_value * reference_value;
            sum_target_sq += weight * target_value * target_value;
            sum_product += weight * reference_value * target_value;
        }
    }
    let covariance = sum_product - sum_reference * sum_target / count;
    let reference_energy = sum_reference_sq - sum_reference * sum_reference / count;
    let target_energy = sum_target_sq - sum_target * sum_target / count;
    let denominator = (reference_energy.max(0.0) * target_energy.max(0.0)).sqrt();
    (denominator > 1.0e-6).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

#[derive(Clone, Copy, Debug)]
struct LocalPatchProjection {
    centre: Vec2,
    target_centre: Vec2,
    target_dx: Vec2,
    target_dy: Vec2,
}

#[derive(Clone, Copy)]
struct LocalPatchReferenceRays {
    centre: Ray,
    left: Ray,
    right: Ray,
    above: Ray,
    below: Ray,
}

#[derive(Clone, Copy)]
struct PreparedReferenceSample {
    point: Vec2,
    value: f32,
    weight: f32,
}

struct PreparedReferencePatch {
    samples: Vec<PreparedReferenceSample>,
    count: f32,
    sum_reference: f32,
    sum_reference_sq: f32,
}

impl PreparedReferencePatch {
    fn with_radius(radius: usize) -> Self {
        let diameter = radius.saturating_mul(2).saturating_add(1);
        Self {
            samples: Vec::with_capacity(diameter.saturating_mul(diameter)),
            count: 0.0,
            sum_reference: 0.0,
            sum_reference_sq: 0.0,
        }
    }
}

#[derive(Default)]
struct AggregateScratch {
    ordered: Vec<f32>,
    positive_information: Vec<f32>,
}

impl LocalPatchProjection {
    #[inline]
    fn map(self, point: Vec2) -> Vec2 {
        let dx = point[0] - self.centre[0];
        let dy = point[1] - self.centre[1];
        [
            self.target_centre[0] + self.target_dx[0] * dx + self.target_dy[0] * dy,
            self.target_centre[1] + self.target_dx[1] * dx + self.target_dy[1] * dy,
        ]
    }
}

fn physical_epipolar_tangent(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
) -> Option<Vec2> {
    let minimum_inverse = 1.0 / options.far_depth;
    let maximum_inverse = 1.0 / options.near_depth;
    let inverse_step =
        (maximum_inverse - minimum_inverse) / options.planes.saturating_sub(1).max(1) as f64;
    let inverse = depth.map_or(minimum_inverse, |depth| 1.0 / depth);
    let first_inverse = (inverse - inverse_step).clamp(minimum_inverse, maximum_inverse);
    let second_inverse = (inverse + inverse_step).clamp(minimum_inverse, maximum_inverse);
    if (second_inverse - first_inverse).abs() <= f64::EPSILON {
        return None;
    }
    let first = target_camera.map_from(reference_camera, centre, 1.0 / first_inverse)?;
    let second = target_camera.map_from(reference_camera, centre, 1.0 / second_inverse)?;
    let delta = [second[0] - first[0], second[1] - first[1]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    (length > 1.0e-6).then_some([delta[0] / length, delta[1] / length])
}

fn measured_perpendicular_proposal(
    measured_warp: &Warp,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    physical: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
) -> Vec2 {
    let Some(measured) = measured_warp.map(centre[0] as f32, centre[1] as f32) else {
        return [0.0, 0.0];
    };
    let delta = [
        f64::from(measured[0]) - physical[0],
        f64::from(measured[1]) - physical[1],
    ];
    if !delta[0].is_finite() || !delta[1].is_finite() {
        return [0.0, 0.0];
    }
    let tangent =
        physical_epipolar_tangent(reference_camera, target_camera, centre, depth, options);
    perpendicular_proposal(delta, tangent)
}

fn perpendicular_proposal(delta: Vec2, tangent: Option<Vec2>) -> Vec2 {
    let Some(tangent) = tangent else {
        // With no observable physical depth motion, importing the complete
        // scene-fitted displacement would recreate the false far anchor.
        return [0.0, 0.0];
    };
    let normal = [-tangent[1], tangent[0]];
    let distance = delta[0] * normal[0] + delta[1] * normal[1];
    [normal[0] * distance, normal[1] * distance]
}

/// Build the local reference-to-target projection induced by one physical
/// depth hypothesis. The scene-fitted stage-2 warp supplies only a constant
/// displacement perpendicular to the physical epipolar trajectory. Its
/// along-trajectory component is scene parallax and must not become an
/// infinity anchor or be added again to a finite-depth hypothesis.
fn local_patch_reference_rays(
    reference_camera: &ResolvedCamera,
    centre: Vec2,
) -> LocalPatchReferenceRays {
    const DERIVATIVE_STEP: f64 = 2.0;
    LocalPatchReferenceRays {
        centre: reference_camera.pixel_to_ray(centre),
        left: reference_camera.pixel_to_ray([centre[0] - DERIVATIVE_STEP, centre[1]]),
        right: reference_camera.pixel_to_ray([centre[0] + DERIVATIVE_STEP, centre[1]]),
        above: reference_camera.pixel_to_ray([centre[0], centre[1] - DERIVATIVE_STEP]),
        below: reference_camera.pixel_to_ray([centre[0], centre[1] + DERIVATIVE_STEP]),
    }
}

fn local_patch_projection_from_rays(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    measured_proposal: &Warp,
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
    rays: &LocalPatchReferenceRays,
) -> Option<LocalPatchProjection> {
    const DERIVATIVE_STEP: f64 = 2.0;

    let surface_depth = depth.unwrap_or(INFINITY_DEPTH);
    let centre_ray = rays.centre;
    let surface_point = add(
        centre_ray.origin,
        scale(centre_ray.direction, surface_depth),
    );
    let surface_normal = centre_ray.direction;
    let project_ray = |ray: Ray| -> Option<Vec2> {
        let denominator = dot(ray.direction, surface_normal);
        if denominator.abs() <= 1.0e-9 {
            return None;
        }
        let distance = dot(sub(surface_point, ray.origin), surface_normal) / denominator;
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        target_camera.project(add(ray.origin, scale(ray.direction, distance)))
    };

    let physical_centre = project_ray(rays.centre)?;
    let proposal = measured_perpendicular_proposal(
        measured_proposal,
        reference_camera,
        target_camera,
        centre,
        physical_centre,
        depth,
        options,
    );
    let target_centre = [
        physical_centre[0] + proposal[0],
        physical_centre[1] + proposal[1],
    ];
    let left = project_ray(rays.left)?;
    let right = project_ray(rays.right)?;
    let above = project_ray(rays.above)?;
    let below = project_ray(rays.below)?;
    let derivative_scale = 1.0 / (2.0 * DERIVATIVE_STEP);
    Some(LocalPatchProjection {
        centre,
        target_centre,
        target_dx: [
            (right[0] - left[0]) * derivative_scale,
            (right[1] - left[1]) * derivative_scale,
        ],
        target_dy: [
            (below[0] - above[0]) * derivative_scale,
            (below[1] - above[1]) * derivative_scale,
        ],
    })
}

fn local_patch_projection(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    measured_proposal: &Warp,
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
) -> Option<LocalPatchProjection> {
    let rays = local_patch_reference_rays(reference_camera, centre);
    local_patch_projection_from_rays(
        reference_camera,
        target_camera,
        measured_proposal,
        centre,
        depth,
        options,
        &rays,
    )
}

fn prepare_reference_patch(
    reference: &AlignInput<'_>,
    centre: Vec2,
    options: &DepthOptions,
    patch: &mut PreparedReferencePatch,
) -> bool {
    patch.samples.clear();
    patch.count = 0.0;
    patch.sum_reference = 0.0;
    patch.sum_reference_sq = 0.0;

    let Some(centre_reference) = reference.luminance.sample(
        ((centre[0] - 0.5) * 0.5) as f32,
        ((centre[1] - 0.5) * 0.5) as f32,
    ) else {
        return false;
    };
    let radius = options.patch_radius;
    let sigma = (radius as f32 * 0.75).max(1.0);
    for dy in -(radius as isize)..=radius as isize {
        for dx in -(radius as isize)..=radius as isize {
            let point = [centre[0] + dx as f64 * 2.0, centre[1] + dy as f64 * 2.0];
            let Some(reference_value) = reference.luminance.sample(
                ((point[0] - 0.5) * 0.5) as f32,
                ((point[1] - 0.5) * 0.5) as f32,
            ) else {
                patch.samples.clear();
                return false;
            };
            let distance_sq = (dx * dx + dy * dy) as f32;
            let spatial = (-distance_sq / (2.0 * sigma * sigma)).exp();
            let range = (-1.2 * (reference_value - centre_reference).abs()).exp();
            let weight = spatial * range;
            patch.count += weight;
            patch.sum_reference += weight * reference_value;
            patch.sum_reference_sq += weight * reference_value * reference_value;
            patch.samples.push(PreparedReferenceSample {
                point,
                value: reference_value,
                weight,
            });
        }
    }
    true
}

fn projected_patch_zncc_prepared(
    target: &AlignInput<'_>,
    projection: &LocalPatchProjection,
    residual: Vec2,
    reference_patch: &PreparedReferencePatch,
) -> Option<f32> {
    let mut sum_target = 0.0f32;
    let mut sum_target_sq = 0.0f32;
    let mut sum_product = 0.0f32;
    for sample in &reference_patch.samples {
        let mapped = projection.map(sample.point);
        let mapped = [mapped[0] + residual[0], mapped[1] + residual[1]];
        let target_value = target.luminance.sample(
            ((mapped[0] - 0.5) * 0.5) as f32,
            ((mapped[1] - 0.5) * 0.5) as f32,
        )?;
        sum_target += sample.weight * target_value;
        sum_target_sq += sample.weight * target_value * target_value;
        sum_product += sample.weight * sample.value * target_value;
    }
    let count = reference_patch.count;
    let covariance = sum_product - reference_patch.sum_reference * sum_target / count;
    let reference_energy = reference_patch.sum_reference_sq
        - reference_patch.sum_reference * reference_patch.sum_reference / count;
    let target_energy = sum_target_sq - sum_target * sum_target / count;
    let denominator = (reference_energy.max(0.0) * target_energy.max(0.0)).sqrt();
    (denominator > 1.0e-6).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn projected_patch_zncc_from_prepared_rays(
    reference: &AlignInput<'_>,
    target: &AlignInput<'_>,
    measured_proposal: &Warp,
    centre: Vec2,
    depth: Option<f64>,
    residual: Vec2,
    options: &DepthOptions,
    reference_patch: &PreparedReferencePatch,
    reference_rays: &LocalPatchReferenceRays,
) -> Option<f32> {
    let projection = local_patch_projection_from_rays(
        reference.camera?,
        target.camera?,
        measured_proposal,
        centre,
        depth,
        options,
        reference_rays,
    )?;
    projected_patch_zncc_prepared(target, &projection, residual, reference_patch)
}

fn aggregate_with_scratch(
    scores: &[ViewScore],
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
    scratch: &mut AggregateScratch,
) -> Option<AggregateEvidence> {
    match geometry_mode {
        DepthGeometryMode::PhysicalRig => {
            aggregate_physical_evidence_with_scratch(scores, options, scratch)
        }
        DepthGeometryMode::WarpSeeded => {
            if scores.len() < options.minimum_support {
                return None;
            }
            scratch.ordered.clear();
            scratch
                .ordered
                .extend(scores.iter().map(|score| score.score));
            scratch.ordered.sort_by(|left, right| right.total_cmp(left));
            scratch
                .ordered
                .truncate(options.best_view_count.min(scratch.ordered.len()));
            (scratch.ordered.len() >= options.minimum_support).then(|| {
                let score = scratch.ordered.iter().sum::<f32>() / scratch.ordered.len() as f32;
                AggregateEvidence {
                    photometric_score: score,
                    paired_photometric_score: None,
                    paired_far_score: None,
                    ranking_score: score,
                }
            })
        }
    }
}

fn aggregate_physical_evidence_with_scratch(
    scores: &[ViewScore],
    options: &DepthOptions,
    scratch: &mut AggregateScratch,
) -> Option<AggregateEvidence> {
    if scores.len() < options.minimum_support {
        return None;
    }

    scratch.ordered.clear();
    scratch
        .ordered
        .extend(scores.iter().map(|view| view.compatibility_score));
    scratch.ordered.sort_by(|left, right| right.total_cmp(left));
    let anchor_count = options.minimum_support.max(2).min(scratch.ordered.len());
    let anchor = scratch.ordered[..anchor_count].iter().sum::<f32>() / anchor_count as f32;
    let threshold = anchor - PHYSICAL_CONSENSUS_BAND;

    scratch.positive_information.clear();
    scratch.positive_information.extend(
        scores
            .iter()
            .map(|view| view.depth_information)
            .filter(|value| value.is_finite() && *value > 1.0e-6),
    );
    scratch.positive_information.sort_by(f32::total_cmp);
    let median_information = scratch
        .positive_information
        .get(scratch.positive_information.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .max(1.0e-6);

    let mut weighted_score = 0.0f32;
    let mut total_weight = 0.0f32;
    let mut paired_finite = 0.0f32;
    let mut paired_far = 0.0f32;
    let mut paired_weight = 0.0f32;
    let mut paired_support = 0.0f32;
    let mut effective_support = 0.0f32;
    let mut effective_independent_support = 0.0f32;
    for view in scores {
        let x = ((view.compatibility_score - threshold) / PHYSICAL_CONSENSUS_SOFTNESS)
            .clamp(-20.0, 20.0);
        let compatibility = 1.0 / (1.0 + (-x).exp());
        effective_support += compatibility;

        let relative_information = if view.depth_information > 1.0e-6 {
            view.depth_information / median_information
        } else {
            0.0
        };
        let information_weight = relative_information.clamp(
            PHYSICAL_MIN_INFORMATION_WEIGHT,
            PHYSICAL_MAX_INFORMATION_WEIGHT,
        );
        let photometric_weight = compatibility * (0.75 + 0.25 * information_weight);
        weighted_score += photometric_weight * view.score;
        total_weight += photometric_weight;
        if let Some(far_score) = view.far_score {
            paired_finite += photometric_weight * view.score;
            paired_far += photometric_weight * far_score;
            paired_weight += photometric_weight;
            paired_support += compatibility;
        }
        let independent_fraction = if relative_information > 0.0 {
            relative_information / (1.0 + relative_information)
        } else {
            0.0
        };
        effective_independent_support += compatibility * independent_fraction;
    }

    if effective_support < options.minimum_support as f32 * 0.75 || total_weight <= 1.0e-6 {
        return None;
    }
    let photometric_score = (weighted_score / total_weight).clamp(-1.0, 1.0);
    let (paired_photometric_score, paired_far_score) =
        if paired_weight > 1.0e-6 && paired_support >= options.minimum_support as f32 * 0.75 {
            (
                Some((paired_finite / paired_weight).clamp(-1.0, 1.0)),
                Some((paired_far / paired_weight).clamp(-1.0, 1.0)),
            )
        } else {
            (None, None)
        };
    let minimum_independent = options.minimum_support as f32 * 0.5;
    let excess_support = (effective_independent_support - minimum_independent).max(0.0);
    let support_confidence = 1.0 - (-excess_support / PHYSICAL_SUPPORT_CONFIDENCE_SATURATION).exp();
    let ranking_score = if photometric_score > 0.0 {
        photometric_score + (1.0 - photometric_score) * support_confidence
    } else {
        photometric_score
    };
    Some(AggregateEvidence {
        photometric_score,
        paired_photometric_score,
        paired_far_score,
        ranking_score: ranking_score.clamp(-1.0, 1.0),
    })
}

#[cfg(test)]
fn aggregate(
    scores: &[ViewScore],
    options: &DepthOptions,
    geometry_mode: DepthGeometryMode,
) -> Option<AggregateEvidence> {
    match geometry_mode {
        DepthGeometryMode::PhysicalRig => {
            aggregate_physical_consensus(scores, options).map(|consensus| consensus.evidence)
        }
        DepthGeometryMode::WarpSeeded => aggregate_warp_seeded(scores, options),
    }
}

/// Legacy compatibility aggregation used only when the dense search is still
/// seeded by the old image-space warp.
#[cfg(test)]
fn aggregate_warp_seeded(
    scores: &[ViewScore],
    options: &DepthOptions,
) -> Option<AggregateEvidence> {
    if scores.len() < options.minimum_support {
        return None;
    }
    let mut values = scores.iter().map(|score| score.score).collect::<Vec<_>>();
    values.sort_by(|left, right| right.total_cmp(left));
    values.truncate(options.best_view_count.min(values.len()));
    (values.len() >= options.minimum_support).then(|| {
        let score = values.iter().sum::<f32>() / values.len() as f32;
        AggregateEvidence {
            photometric_score: score,
            paired_photometric_score: None,
            paired_far_score: None,
            ranking_score: score,
        }
    })
}

/// Robust all-view consensus for calibrated physical geometry.
///
/// Absolute quality thresholds remain on the stable ZNCC-like
/// `photometric_score`. Independent baseline-diverse support changes only
/// `ranking_score`; it never changes the absolute `minimum_score` threshold or
/// makes a negative correlation look better merely because more views exist.
#[cfg(test)]
fn aggregate_physical_consensus(
    scores: &[ViewScore],
    options: &DepthOptions,
) -> Option<PhysicalConsensus> {
    if scores.len() < options.minimum_support {
        return None;
    }

    // The strongest minimum-support set locates the visible component, but no
    // view is discarded solely by rank. This is only a soft membership model.
    let mut ordered = scores
        .iter()
        .map(|view| view.compatibility_score)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| right.total_cmp(left));
    let anchor_count = options.minimum_support.max(2).min(ordered.len());
    let anchor = ordered[..anchor_count].iter().sum::<f32>() / anchor_count as f32;
    let threshold = anchor - PHYSICAL_CONSENSUS_BAND;

    // Normalize depth information by the median positive value.  A
    // near-coincident view may confirm appearance but must not count as an
    // independent depth baseline.
    let mut positive_information = scores
        .iter()
        .map(|view| view.depth_information)
        .filter(|value| value.is_finite() && *value > 1.0e-6)
        .collect::<Vec<_>>();
    positive_information.sort_by(f32::total_cmp);
    let median_information = positive_information
        .get(positive_information.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .max(1.0e-6);

    let mut weighted_score = 0.0f32;
    let mut total_weight = 0.0f32;
    let mut paired_finite = 0.0f32;
    let mut paired_far = 0.0f32;
    let mut paired_weight = 0.0f32;
    let mut paired_support = 0.0f32;
    let mut effective_support = 0.0f32;
    let mut effective_independent_support = 0.0f32;
    let mut members = Vec::with_capacity(scores.len());
    for view in scores {
        let x = ((view.compatibility_score - threshold) / PHYSICAL_CONSENSUS_SOFTNESS)
            .clamp(-20.0, 20.0);
        let compatibility = 1.0 / (1.0 + (-x).exp());
        effective_support += compatibility;

        let relative_information = if view.depth_information > 1.0e-6 {
            view.depth_information / median_information
        } else {
            0.0
        };
        let information_weight = relative_information.clamp(
            PHYSICAL_MIN_INFORMATION_WEIGHT,
            PHYSICAL_MAX_INFORMATION_WEIGHT,
        );
        let photometric_weight = compatibility * (0.75 + 0.25 * information_weight);
        weighted_score += photometric_weight * view.score;
        total_weight += photometric_weight;

        // Paired finite/far comparison must use exactly the same view and
        // exactly the same weight.  This prevents a finite hypothesis from
        // appearing better/worse merely because its consensus membership is
        // different from the independently aggregated far hypothesis.
        if let Some(far_score) = view.far_score {
            paired_finite += photometric_weight * view.score;
            paired_far += photometric_weight * far_score;
            paired_weight += photometric_weight;
            paired_support += compatibility;
        }

        let independent_fraction = if relative_information > 0.0 {
            relative_information / (1.0 + relative_information)
        } else {
            0.0
        };
        effective_independent_support += compatibility * independent_fraction;
        members.push(PhysicalMember {
            source_index: view.source_index,
            score: view.score,
            compatibility,
        });
    }

    if effective_support < options.minimum_support as f32 * 0.75 || total_weight <= 1.0e-6 {
        return None;
    }

    let photometric_score = (weighted_score / total_weight).clamp(-1.0, 1.0);
    let (paired_photometric_score, paired_far_score) =
        if paired_weight > 1.0e-6 && paired_support >= options.minimum_support as f32 * 0.75 {
            (
                Some((paired_finite / paired_weight).clamp(-1.0, 1.0)),
                Some((paired_far / paired_weight).clamp(-1.0, 1.0)),
            )
        } else {
            (None, None)
        };

    // Treat independent support as confidence in the photometric hypothesis,
    // not as another signed correlation term.  This keeps the ranking inside
    // [-1, 1], leaves negative/weak hypotheses weak, and lets broad coherent
    // support accumulate enough evidence to beat a chance few-view peak.
    let minimum_independent = options.minimum_support as f32 * 0.5;
    let excess_support = (effective_independent_support - minimum_independent).max(0.0);
    let support_confidence = 1.0 - (-excess_support / PHYSICAL_SUPPORT_CONFIDENCE_SATURATION).exp();
    let ranking_score = if photometric_score > 0.0 {
        photometric_score + (1.0 - photometric_score) * support_confidence
    } else {
        photometric_score
    };

    Some(PhysicalConsensus {
        evidence: AggregateEvidence {
            photometric_score,
            paired_photometric_score,
            paired_far_score,
            ranking_score: ranking_score.clamp(-1.0, 1.0),
        },
        members,
    })
}

/// Geometry-only proxy for how much inverse-depth information a target view
/// contributes at this reference location. For a pinhole-like camera the
/// sensitivity is proportional to focal length times the component of the
/// inter-camera baseline perpendicular to the reference ray. It is used only
/// as a bounded relative weight; photometric compatibility still decides
/// whether the view belongs to the visible surface.
fn physical_depth_information(
    reference: &ResolvedCamera,
    target: &ResolvedCamera,
    centre: Vec2,
) -> f32 {
    let ray = reference.pixel_to_ray(centre);
    let baseline = sub(target.center(), ray.origin);
    let axial = scale(ray.direction, dot(baseline, ray.direction));
    let lateral = sub(baseline, axial);
    let perpendicular_baseline = dot(lateral, lateral).max(0.0).sqrt();
    let information = perpendicular_baseline * target.focal_px.max(1.0);
    if information.is_finite() && information > 0.0 {
        information.min(f32::MAX as f64) as f32
    } else {
        0.0
    }
}

fn reference_guidance(reference: &AlignInput<'_>, pixel: Vec2) -> f32 {
    reference
        .luminance
        .sample(
            ((pixel[0] - 0.5) * 0.5) as f32,
            ((pixel[1] - 0.5) * 0.5) as f32,
        )
        .unwrap_or(0.0)
}

fn inverse_depth_samples(near: f64, far: f64, count: usize) -> Vec<f64> {
    let near_inverse = 1.0 / near;
    let far_inverse = 1.0 / far;
    (0..count)
        .map(|index| {
            let weight = index as f64 / (count - 1) as f64;
            1.0 / (near_inverse * (1.0 - weight) + far_inverse * weight)
        })
        .collect()
}

fn median<T: Copy>(sorted: &[T]) -> Option<T> {
    sorted.get(sorted.len() / 2).copied()
}

fn fraction(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

fn scale_color(color: [u16; 3], scale: f64) -> [u16; 3] {
    color.map(|channel| (f64::from(channel) * scale).round() as u16)
}

fn depth_color(normalized: f64) -> [u16; 3] {
    let stops = [
        [0.0, 0.1, 0.7],
        [0.0, 0.9, 1.0],
        [0.0, 1.0, 0.2],
        [1.0, 0.9, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let position = normalized.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let lower = (position.floor() as usize).min(stops.len() - 1);
    let upper = (lower + 1).min(stops.len() - 1);
    let fraction = position - lower as f64;
    std::array::from_fn(|channel| {
        ((stops[lower][channel] * (1.0 - fraction) + stops[upper][channel] * fraction) * 65_535.0)
            .round() as u16
    })
}

fn valid_options(options: &DepthOptions, geometry_mode: DepthGeometryMode) -> bool {
    options.enabled
        && options.grid_step > 0
        && options.near_depth.is_finite()
        && options.far_depth.is_finite()
        && options.near_depth > 0.0
        && options.far_depth > options.near_depth
        && options.planes >= 3
        && options.patch_radius > 0
        && options.minimum_support > 0
        && (geometry_mode.is_physical() || options.best_view_count >= options.minimum_support)
        && options.minimum_neighbour_support > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_mode_distinguishes_physical_correspondence() {
        assert!(!DepthGeometryMode::WarpSeeded.is_physical());
        assert!(DepthGeometryMode::PhysicalRig.is_physical());
    }

    #[test]
    fn local_patch_projection_applies_its_depth_induced_jacobian() {
        let projection = LocalPatchProjection {
            centre: [100.0, 50.0],
            target_centre: [210.0, 80.0],
            target_dx: [1.5, 0.25],
            target_dy: [-0.5, 2.0],
        };
        let mapped = projection.map([102.0, 47.0]);
        assert!((mapped[0] - 214.5).abs() < 1.0e-12);
        assert!((mapped[1] - 74.5).abs() < 1.0e-12);
    }

    #[test]
    fn prepared_patch_zncc_matches_inline_accumulation() {
        let mut reference_plane = Plane::new(24, 24);
        let mut target_plane = Plane::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                let index = y * 24 + x;
                reference_plane.data[index] = ((x * 13 + y * 7 + x * y * 3) % 41) as f32 / 40.0;
                target_plane.data[index] = ((x * 11 + y * 5 + x * y * 2 + 3) % 37) as f32 / 36.0;
            }
        }
        let reference = AlignInput {
            name: "reference",
            luminance: &reference_plane,
            width: 48,
            height: 48,
            camera: None,
            depth_evidence_enabled: true,
            nominal_focal_px: 1_000.0,
        };
        let target = AlignInput {
            name: "target",
            luminance: &target_plane,
            width: 48,
            height: 48,
            camera: None,
            depth_evidence_enabled: true,
            nominal_focal_px: 1_000.0,
        };
        let options = DepthOptions {
            patch_radius: 3,
            ..DepthOptions::default()
        };
        let centre = [24.5, 22.5];
        let projection = LocalPatchProjection {
            centre,
            target_centre: [25.1, 22.2],
            target_dx: [1.01, 0.02],
            target_dy: [-0.01, 0.99],
        };
        let residual = [0.15, -0.08];

        let mut prepared = PreparedReferencePatch::with_radius(options.patch_radius);
        assert!(prepare_reference_patch(
            &reference,
            centre,
            &options,
            &mut prepared
        ));
        let actual = projected_patch_zncc_prepared(&target, &projection, residual, &prepared)
            .expect("prepared ZNCC");

        let centre_reference = reference
            .luminance
            .sample(
                ((centre[0] - 0.5) * 0.5) as f32,
                ((centre[1] - 0.5) * 0.5) as f32,
            )
            .unwrap();
        let radius = options.patch_radius;
        let sigma = (radius as f32 * 0.75).max(1.0);
        let mut count = 0.0f32;
        let mut sum_reference = 0.0f32;
        let mut sum_target = 0.0f32;
        let mut sum_reference_sq = 0.0f32;
        let mut sum_target_sq = 0.0f32;
        let mut sum_product = 0.0f32;
        for dy in -(radius as isize)..=radius as isize {
            for dx in -(radius as isize)..=radius as isize {
                let point = [centre[0] + dx as f64 * 2.0, centre[1] + dy as f64 * 2.0];
                let reference_value = reference
                    .luminance
                    .sample(
                        ((point[0] - 0.5) * 0.5) as f32,
                        ((point[1] - 0.5) * 0.5) as f32,
                    )
                    .unwrap();
                let mapped = projection.map(point);
                let target_value = target
                    .luminance
                    .sample(
                        ((mapped[0] + residual[0] - 0.5) * 0.5) as f32,
                        ((mapped[1] + residual[1] - 0.5) * 0.5) as f32,
                    )
                    .unwrap();
                let distance_sq = (dx * dx + dy * dy) as f32;
                let spatial = (-distance_sq / (2.0 * sigma * sigma)).exp();
                let range = (-1.2 * (reference_value - centre_reference).abs()).exp();
                let weight = spatial * range;
                count += weight;
                sum_reference += weight * reference_value;
                sum_target += weight * target_value;
                sum_reference_sq += weight * reference_value * reference_value;
                sum_target_sq += weight * target_value * target_value;
                sum_product += weight * reference_value * target_value;
            }
        }
        let covariance = sum_product - sum_reference * sum_target / count;
        let reference_energy = sum_reference_sq - sum_reference * sum_reference / count;
        let target_energy = sum_target_sq - sum_target * sum_target / count;
        let expected = (covariance / (reference_energy.max(0.0) * target_energy.max(0.0)).sqrt())
            .clamp(-1.0, 1.0);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn physical_proposal_discards_scene_parallax_along_the_depth_locus() {
        assert_eq!(
            perpendicular_proposal([33.0, 14.0], Some([1.0, 0.0])),
            [0.0, 14.0]
        );
        assert_eq!(perpendicular_proposal([33.0, 14.0], None), [0.0, 0.0]);
    }

    #[test]
    fn target_visibility_buffer_only_blocks_clearly_nearer_surfaces() {
        let buffer = TargetVisibilityBuffer {
            cell_size: 4.0,
            columns: 2,
            rows: 2,
            samples: vec![
                Some(TargetVisibilitySample {
                    pixel: [2.0, 2.0],
                    depth: 1_000.0,
                }),
                None,
                None,
                None,
            ],
        };
        assert!(buffer.nearer_surface_at([2.0, 2.0], 2_000.0));
        assert!(!buffer.nearer_surface_at([2.0, 2.0], 1_020.0));
        assert!(!buffer.nearer_surface_at([20.0, 20.0], 2_000.0));
    }

    #[test]
    fn inverse_depth_planes_include_bounds_and_favour_near_resolution() {
        let depths = inverse_depth_samples(500.0, 100_000.0, 5);
        assert!((depths[0] - 500.0).abs() < 1e-9);
        assert!((depths[4] - 100_000.0).abs() < 1e-6);
        assert!(depths.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(depths[1] - depths[0] < depths[4] - depths[3]);
    }

    #[test]
    fn direct_depth_refinement_merges_in_inverse_depth_order() {
        let evidence = |score| {
            Some(AggregateEvidence {
                photometric_score: score,
                paired_photometric_score: None,
                paired_far_score: None,
                ranking_score: score,
            })
        };
        let mut depths = vec![10_000.0, 2_000.0, 1_000.0];
        let mut scores = vec![evidence(0.1), evidence(0.2), evidence(0.3)];
        merge_depth_scores(&mut depths, &mut scores, vec![(4_000.0, evidence(0.9))]);

        assert_eq!(depths, vec![10_000.0, 4_000.0, 2_000.0, 1_000.0]);
        assert_eq!(scores[1].expect("inserted score").ranking_score, 0.9);
    }

    #[test]
    fn sublabel_fit_refines_inverse_depth_between_planes() {
        let labels = [None, Some(10_000.0), Some(5_000.0), Some(10_000.0 / 3.0)];
        let depth = sublabel_depth(&labels, &[9.0, 2.0, 0.0, 1.0], 2);
        assert!(depth < 5_000.0);
        assert!(depth > 10_000.0 / 3.0);
    }

    #[test]
    fn physical_consensus_uses_all_compatible_views_without_majority_outvoting() {
        let options = DepthOptions::default();
        let view = |source_index, score| ViewScore {
            source_index,
            score,
            far_score: None,
            compatibility_score: score,
            depth_information: 1.0,
        };
        let one = [view(0, 0.9)];
        assert!(aggregate(&one, &options, DepthGeometryMode::PhysicalRig).is_none());

        // Three cameras can establish a real surface even when five other
        // viewpoints are occluded or see another layer. Those five become
        // nearly neutral instead of outvoting the visible set.
        let three_visible_five_conflicting = [
            view(0, 0.94),
            view(1, 0.91),
            view(2, 0.88),
            view(3, 0.20),
            view(4, 0.15),
            view(5, 0.10),
            view(6, 0.05),
            view(7, -0.10),
        ];
        let evidence = aggregate(
            &three_visible_five_conflicting,
            &options,
            DepthGeometryMode::PhysicalRig,
        )
        .unwrap();
        assert!(
            evidence.photometric_score > 0.80,
            "visible minority should remain the photometric consensus: {:?}",
            evidence
        );

        // Broad support is additional ranking evidence, but it is deliberately
        // bounded: a genuine surface visible in only three cameras must not be
        // outvoted merely because another layer is visible to more cameras.
        // Absolute thresholds remain on the separate photometric score.
        let accidental_three = [
            view(0, 0.97),
            view(1, 0.95),
            view(2, 0.94),
            view(3, 0.18),
            view(4, 0.16),
            view(5, 0.15),
            view(6, 0.12),
            view(7, 0.10),
            view(8, 0.08),
            view(9, 0.05),
            view(10, 0.02),
        ];
        let broad_nine = [
            view(0, 0.86),
            view(1, 0.85),
            view(2, 0.84),
            view(3, 0.83),
            view(4, 0.82),
            view(5, 0.81),
            view(6, 0.80),
            view(7, 0.79),
            view(8, 0.78),
            view(9, 0.10),
            view(10, 0.05),
        ];
        let accidental =
            aggregate(&accidental_three, &options, DepthGeometryMode::PhysicalRig).unwrap();
        let broad = aggregate(&broad_nine, &options, DepthGeometryMode::PhysicalRig).unwrap();
        assert!(broad.ranking_score > broad.photometric_score);
        assert!(accidental.ranking_score >= accidental.photometric_score);
        assert!(broad.photometric_score > 0.75);
        assert!(
            broad.ranking_score > accidental.ranking_score,
            "broad independent support should beat a chance few-view peak: {broad:?} vs {accidental:?}"
        );
    }

    #[test]
    fn physical_consensus_pairs_finite_and_far_on_identical_view_weights() {
        let options = DepthOptions::default();
        let finite = [0.90, 0.90, 0.50, 0.50];
        let far = [0.90, 0.90, 0.40, 0.40];
        let views = finite
            .into_iter()
            .zip(far)
            .enumerate()
            .map(|(source_index, (score, far_score))| ViewScore {
                source_index,
                score,
                far_score: Some(far_score),
                compatibility_score: score - far_score,
                depth_information: 1.0,
            })
            .collect::<Vec<_>>();
        let evidence = aggregate_physical_consensus(&views, &options)
            .unwrap()
            .evidence;
        let paired_finite = evidence.paired_photometric_score.unwrap();
        let paired_far = evidence.paired_far_score.unwrap();
        assert!(paired_finite > paired_far);
        assert!((paired_finite - paired_far) > 0.0);
    }

    #[test]
    fn physical_consensus_keeps_absolute_score_scale_stable_across_view_count() {
        let options = DepthOptions::default();
        let views2 = (0..2)
            .map(|source_index| ViewScore {
                source_index,
                score: 0.90,
                far_score: None,
                compatibility_score: 0.90,
                depth_information: 1.0,
            })
            .collect::<Vec<_>>();
        let views10 = (0..10)
            .map(|source_index| ViewScore {
                source_index,
                score: 0.90,
                far_score: None,
                compatibility_score: 0.90,
                depth_information: 1.0,
            })
            .collect::<Vec<_>>();
        let two = aggregate(&views2, &options, DepthGeometryMode::PhysicalRig).unwrap();
        let ten = aggregate(&views10, &options, DepthGeometryMode::PhysicalRig).unwrap();
        assert!((two.photometric_score - 0.90).abs() < 1.0e-5);
        assert!((ten.photometric_score - 0.90).abs() < 1.0e-5);
        assert!(ten.ranking_score > two.ranking_score);
    }

    #[test]
    fn physical_visibility_bits_match_consensus_membership() {
        let options = DepthOptions {
            minimum_support: 2,
            ..DepthOptions::default()
        };
        let views = [
            ViewScore {
                source_index: 1,
                score: 0.91,
                far_score: Some(0.72),
                compatibility_score: 0.19,
                depth_information: 0.4,
            },
            ViewScore {
                source_index: 3,
                score: 0.84,
                far_score: Some(0.71),
                compatibility_score: 0.13,
                depth_information: 1.0,
            },
            ViewScore {
                source_index: 5,
                score: 0.68,
                far_score: Some(0.75),
                compatibility_score: -0.07,
                depth_information: 2.5,
            },
        ];
        let consensus = aggregate_physical_consensus(&views, &options).expect("consensus");
        let per_view_score_floor = (options.minimum_score - 0.10).max(MINIMUM_REGULARIZED_SCORE);
        let mut bits = [0u64; 1];
        let mut scratch = AggregateScratch::default();
        physical_consensus_visibility_bits(&views, &options, &mut bits, &mut scratch);
        for view in views {
            let expected = consensus.members.iter().any(|member| {
                member.source_index == view.source_index
                    && member.compatibility >= PHYSICAL_VISIBLE_COMPATIBILITY
                    && member.score >= per_view_score_floor
            });
            let actual = bits[0] & (1u64 << view.source_index) != 0;
            assert_eq!(actual, expected, "source {}", view.source_index);
        }
    }

    #[test]
    fn scratch_aggregation_matches_the_allocating_paths() {
        let options = DepthOptions {
            best_view_count: 3,
            minimum_support: 2,
            ..DepthOptions::default()
        };
        let views = [
            ViewScore {
                source_index: 1,
                score: 0.91,
                far_score: Some(0.72),
                compatibility_score: 0.19,
                depth_information: 0.4,
            },
            ViewScore {
                source_index: 2,
                score: 0.84,
                far_score: Some(0.71),
                compatibility_score: 0.13,
                depth_information: 1.0,
            },
            ViewScore {
                source_index: 3,
                score: 0.68,
                far_score: Some(0.75),
                compatibility_score: -0.07,
                depth_information: 2.5,
            },
            ViewScore {
                source_index: 4,
                score: 0.37,
                far_score: None,
                compatibility_score: -0.33,
                depth_information: 0.0,
            },
        ];

        for mode in [
            DepthGeometryMode::PhysicalRig,
            DepthGeometryMode::WarpSeeded,
        ] {
            let expected = aggregate(&views, &options, mode).expect("allocating aggregate");
            let mut scratch = AggregateScratch::default();
            let actual = aggregate_with_scratch(&views, &options, mode, &mut scratch)
                .expect("scratch aggregate");
            assert_eq!(
                actual.photometric_score.to_bits(),
                expected.photometric_score.to_bits()
            );
            assert_eq!(
                actual.paired_photometric_score.map(f32::to_bits),
                expected.paired_photometric_score.map(f32::to_bits)
            );
            assert_eq!(
                actual.paired_far_score.map(f32::to_bits),
                expected.paired_far_score.map(f32::to_bits)
            );
            assert_eq!(
                actual.ranking_score.to_bits(),
                expected.ranking_score.to_bits()
            );
        }
    }

    #[test]
    fn physical_consensus_does_not_truncate_at_legacy_best_view_count() {
        let options = DepthOptions {
            best_view_count: 3,
            ..DepthOptions::default()
        };
        let views = (0..11)
            .map(|source_index| ViewScore {
                source_index,
                score: 0.80 + source_index as f32 * 0.005,
                far_score: None,
                compatibility_score: 0.80 + source_index as f32 * 0.005,
                depth_information: 1.0,
            })
            .collect::<Vec<_>>();
        let physical = aggregate(&views, &options, DepthGeometryMode::PhysicalRig).unwrap();
        let legacy = aggregate(&views, &options, DepthGeometryMode::WarpSeeded).unwrap();
        assert!(physical.photometric_score > 0.75);
        assert!(legacy.photometric_score > physical.photometric_score);

        // Changing the legacy top-K knob cannot alter physical-rig consensus.
        let mut wider = options.clone();
        wider.best_view_count = 11;
        let physical_wider = aggregate(&views, &wider, DepthGeometryMode::PhysicalRig).unwrap();
        assert!((physical.photometric_score - physical_wider.photometric_score).abs() < 1.0e-6);
        assert!((physical.ranking_score - physical_wider.ranking_score).abs() < 1.0e-6);
        assert!(valid_options(&wider, DepthGeometryMode::PhysicalRig));
    }

    #[test]
    fn physical_consensus_bounds_geometric_information_weight() {
        let options = DepthOptions::default();
        let views = [
            ViewScore {
                source_index: 0,
                score: 0.80,
                far_score: None,
                compatibility_score: 0.80,
                depth_information: 1.0,
            },
            ViewScore {
                source_index: 1,
                score: 0.80,
                far_score: None,
                compatibility_score: 0.80,
                depth_information: 1.0,
            },
            ViewScore {
                source_index: 2,
                score: 0.30,
                far_score: None,
                compatibility_score: 0.30,
                depth_information: 1.0e6,
            },
        ];
        let evidence = aggregate(&views, &options, DepthGeometryMode::PhysicalRig).unwrap();
        assert!(
            evidence.photometric_score > 0.65,
            "one extreme baseline must not dominate consensus: {evidence:?}"
        );
    }

    #[test]
    fn semi_global_matching_completes_an_ambiguous_middle_node() {
        let labels = 4;
        let columns = 5;
        let rows = 1;
        let mut costs = vec![1.0; columns * labels];
        for column in [0, 1, 3, 4] {
            costs[column * labels + 2] = 0.0;
        }
        let guidance = vec![0.5; columns];
        let regularised = semi_global_costs(&costs, &guidance, columns, rows, labels);
        let middle = 2 * labels;
        let best = regularised[middle..middle + labels]
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .map(|(label, _)| label);
        assert_eq!(best, Some(2));
    }

    #[test]
    fn diagonal_paths_visit_every_node_in_both_families() {
        for mirrored in [false, true] {
            let paths = diagonal_paths(5, 4, mirrored);
            let mut visits = vec![0usize; 20];
            for index in paths.into_iter().flatten() {
                visits[index] += 1;
            }
            assert!(visits.into_iter().all(|count| count == 1));
        }
    }

    #[test]
    fn warp_consensus_requires_similar_parallax() {
        let refined = |dx: f32| NodeWarp::Refined {
            global: [10.0, 20.0],
            point: [10.0 + dx, 20.0],
            confidence: 1.0,
            measured: true,
        };
        assert!(warp_decisions_agree(0, refined(2.0), refined(4.5)));
        assert!(!warp_decisions_agree(0, refined(2.0), refined(6.0)));
        assert!(!warp_decisions_agree(
            0,
            refined(2.0),
            NodeWarp::Global {
                point: [10.0, 20.0],
                confidence: 1.0,
            }
        ));

        let mut isolated = vec![
            NodeWarp::Global {
                point: [10.0, 20.0],
                confidence: 1.0,
            };
            9
        ];
        isolated[4] = refined(2.0);
        enforce_warp_consensus(&mut isolated, 3, 3, DepthGeometryMode::PhysicalRig);
        assert!(matches!(isolated[4], NodeWarp::Unknown { .. }));

        let mut legacy = vec![
            NodeWarp::Global {
                point: [10.0, 20.0],
                confidence: 1.0,
            };
            9
        ];
        legacy[4] = refined(2.0);
        enforce_warp_consensus(&mut legacy, 3, 3, DepthGeometryMode::WarpSeeded);
        assert!(matches!(legacy[4], NodeWarp::Global { .. }));
    }

    #[test]
    fn warp_boundary_suppression_requires_an_image_edge() {
        let refined = |dx: f32| NodeWarp::Refined {
            global: [10.0, 20.0],
            point: [10.0 + dx, 20.0],
            confidence: 1.0,
            measured: true,
        };
        let mut smooth = vec![refined(0.0), refined(4.0)];
        suppress_warp_boundaries(&mut smooth, &[1.0, 1.1], 2, 1);
        assert!(
            smooth
                .iter()
                .all(|decision| matches!(decision, NodeWarp::Refined { .. }))
        );

        let mut edge = vec![refined(0.0), refined(4.0)];
        suppress_warp_boundaries(&mut edge, &[1.0, 1.5], 2, 1);
        assert!(
            edge.iter()
                .all(|decision| matches!(decision, NodeWarp::Boundary(_)))
        );
    }

    #[test]
    fn confident_global_label_is_not_treated_as_a_completion_hole() {
        let volume = CostVolume {
            physical_geometry: false,
            labels: vec![None, Some(10_000.0), Some(2_000.0)],
            scores: vec![0.9, 0.7, 0.6],
            paired_improvements: vec![f32::NAN; 3],
            costs: vec![0.1, 0.3, 0.4],
            guidance: vec![0.5],
            tested: vec![true],
        };
        let regularised = vec![0.4, 1.2, 1.6];
        let (field, fillable) = select_depths(&volume, &regularised, &DepthOptions::default());
        assert!(field[0].is_none());
        assert!(!fillable[0]);
    }

    #[test]
    fn sgm_supported_ambiguous_depth_is_kept_as_regularized() {
        let volume = CostVolume {
            physical_geometry: false,
            labels: vec![None, Some(100_000.0), Some(10_000.0), Some(2_000.0)],
            scores: vec![0.72, 0.70, 0.68, 0.62],
            paired_improvements: vec![f32::NAN; 4],
            costs: vec![0.28, 0.30, 0.32, 0.38],
            guidance: vec![0.5],
            tested: vec![true],
        };
        let regularised = vec![0.8, 0.7, 0.75, 0.9];
        let (field, fillable) = select_depths(&volume, &regularised, &DepthOptions::default());
        assert!(field[0].is_some_and(|node| node.regularized));
        assert!(!fillable[0]);
    }

    #[test]
    fn diagnostics_distinguish_depth_and_provenance() {
        let map = DenseDepthMap {
            columns: 4,
            rows: 1,
            step: 32,
            near_depth: 500.0,
            far_depth: 100_000.0,
            nodes: vec![
                DenseDepthNode {
                    depth: None,
                    confidence: 0.0,
                    provenance: DepthProvenance::Unsupported,
                },
                DenseDepthNode {
                    depth: None,
                    confidence: 1.0,
                    provenance: DepthProvenance::Global,
                },
                DenseDepthNode {
                    depth: Some(500.0),
                    confidence: 1.0,
                    provenance: DepthProvenance::Measured,
                },
                DenseDepthNode {
                    depth: Some(100_000.0),
                    confidence: 0.0,
                    provenance: DepthProvenance::Regularized,
                },
            ],
        };
        let (depth, provenance) = map.diagnostic_samples();
        assert_eq!(depth, [0, 0, 65_535, 1]);
        assert_eq!(&provenance[0..3], &[0, 0, 0]);
        assert_eq!(&provenance[3..6], &[0, 0, 32_768]);
        assert_eq!(&provenance[6..9], &[0, 65_535, 0]);
        assert!(provenance[9] > provenance[10]);
        assert_eq!(provenance[11], 0);

        let visualization = map.visualization_samples();
        assert_eq!(&visualization[0..3], &[0, 0, 0]);
        assert_eq!(&visualization[3..6], &[0, 0, 12_000]);
        assert!(visualization[6] > visualization[8]);
        assert!(visualization[11] > visualization[9]);
    }

    #[test]
    fn completion_does_not_cross_an_unsupported_region() {
        let options = DepthOptions {
            completion_iterations: 8,
            minimum_neighbour_support: 1,
            ..DepthOptions::default()
        };
        let mut field = vec![None; 5];
        field[0] = Some(NodeDepth {
            depth: 2_000.0,
            confidence: 1.0,
            improvement: 0.2,
            regularized: false,
        });
        let guidance = vec![0.0; 5];
        let fillable = [true, true, false, true, true];
        complete_depth_field(&mut field, &guidance, &fillable, 5, 1, &options);
        assert!(field[1].is_some());
        assert!(field[2].is_none());
        assert!(field[3].is_none());
        assert!(field[4].is_none());
    }

    #[test]
    fn direct_consistency_rejects_an_island_without_filling_holes() {
        let node = |depth| {
            Some(NodeDepth {
                depth,
                confidence: 0.6,
                improvement: 0.1,
                regularized: false,
            })
        };
        let mut field = vec![node(2_000.0), None, node(500.0)];
        reject_isolated_direct_depths(&mut field, &[0.5; 3], 3, 1, 1.0e-4);
        assert!(field.iter().all(Option::is_none));
    }

    #[test]
    fn direct_component_filter_removes_speckles_without_growing_surfaces() {
        let node = || {
            Some(NodeDepth {
                depth: 2_000.0,
                confidence: 0.9,
                improvement: 0.1,
                regularized: false,
            })
        };
        let mut field = vec![None; 36];
        // A coherent 3x3 surface survives a six-node minimum.
        for row in 2..=4 {
            for column in 1..=3 {
                field[row * 6 + column] = node();
            }
        }
        // A separate two-node chance island is removed.
        field[3 * 6 + 5] = node();
        field[4 * 6 + 5] = node();
        reject_small_direct_components(&mut field, &[0.5; 36], 6, 6, 1.0e-4, 6);
        assert_eq!(field.iter().flatten().count(), 9);
        assert!(field[3 * 6 + 2].is_some());
        assert!(field[3 * 6 + 5].is_none());
        assert!(field[4 * 6 + 5].is_none());
    }

    #[test]
    fn direct_depth_must_improve_on_far_baseline() {
        let depths = [10_000.0, 9_000.0, 8_000.0, 7_000.0, 6_000.0];
        let evidence = |score: f32| {
            Some(AggregateEvidence {
                photometric_score: score,
                paired_photometric_score: None,
                paired_far_score: None,
                ranking_score: score,
            })
        };
        let scores = [
            evidence(0.60),
            evidence(0.65),
            evidence(0.90),
            evidence(0.65),
            evidence(0.60),
        ];
        let options = DepthOptions::default();

        // The finite winner is extremely clear relative to the other finite
        // labels, but it is indistinguishable from the legacy far/baseline mapping.
        assert!(
            select_direct_depth(
                &depths,
                &scores,
                evidence(0.899),
                false,
                &options,
                DepthGeometryMode::WarpSeeded,
            )
            .is_none()
        );

        // The same unique label is accepted once it also improves on the baseline.
        assert!(
            select_direct_depth(
                &depths,
                &scores,
                evidence(0.89),
                false,
                &options,
                DepthGeometryMode::WarpSeeded,
            )
            .is_some()
        );
    }

    #[test]
    fn local_plane_fit_updates_only_existing_measurements() {
        let mut field = (0..25)
            .map(|index| {
                (index != 12).then_some(NodeDepth {
                    depth: 1.0 / (0.000_2 + (index % 5) as f64 * 0.000_001),
                    confidence: 1.0,
                    improvement: 0.1,
                    regularized: false,
                })
            })
            .collect::<Vec<_>>();
        fit_local_depth_planes(&mut field, &[0.5; 25], 5, 5, 0.000_1);
        assert!(field[12].is_none());
        assert!(field.iter().flatten().all(|node| !node.regularized));
    }
}
