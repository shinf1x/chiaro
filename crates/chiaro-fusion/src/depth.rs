//! Dense calibrated multi-view depth refinement.
//!
//! Global homographies absorb bearing and calibration bias. This stage builds
//! a reference-space inverse-depth cost volume, projects every patch sample
//! through each candidate plane, and aggregates independent camera evidence.
//! Eight-direction semi-global matching regularises weakly textured areas while an
//! edge-aware completion pass fills small holes without freely crossing image
//! discontinuities. The resulting metric field drives each camera's exact
//! calibrated parallax and carries local confidence into synthesis.

use std::{path::Path, thread};

use anyhow::Result;
use serde::Serialize;

use crate::{
    align::{AlignInput, ModuleAlignment, Warp},
    geometry::ResolvedCamera,
    image::Plane,
    math::Vec2,
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

#[derive(Clone, Debug)]
pub struct DepthOptions {
    pub enabled: bool,
    /// Dense control-grid spacing in reference-raster pixels.
    pub grid_step: usize,
    /// Near and far search bounds in calibration units (believed millimetres).
    pub near_depth: f64,
    pub far_depth: f64,
    /// Number of uniformly spaced finite inverse-depth hypotheses.
    pub planes: usize,
    /// Patch radius in half-resolution luminance pixels.
    pub patch_radius: usize,
    /// At least this many different target cameras must support a depth.
    pub minimum_support: usize,
    /// Average only the strongest views, limiting one weak camera's effect.
    pub best_view_count: usize,
    /// Weakest local ZNCC that may seed the dense reconstruction.
    pub minimum_score: f32,
    /// Minimum regularised separation from a non-adjacent depth label.
    pub minimum_margin: f32,
    /// A locally ambiguous label can still seed when it improves the global
    /// warp by at least this amount.
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
    pub tested_nodes: usize,
    /// Nodes whose depth is supported directly by the multiview cost volume.
    pub measured_nodes: usize,
    /// Nodes inferred by SGM or completed from edge-compatible neighbours.
    pub regularized_nodes: usize,
    /// Supported nodes deliberately retaining the global/infinite warp.
    pub fallback_nodes: usize,
    /// Nodes at which this particular module accepted the reconstructed warp.
    pub refined_nodes: usize,
    pub occluded_nodes: usize,
    /// Nodes suppressed around a discontinuous warp boundary so bilinear
    /// interpolation cannot blend foreground and background mappings.
    pub boundary_nodes: usize,
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

    /// Write a quantitative inverse-depth image and a categorical provenance
    /// image. Inverse depth uses 0 for global/unsupported, 1 for the far bound,
    /// and 65535 for the near bound. Provenance is black=unsupported,
    /// blue=global/infinite, green=measured, amber=regularized; finite colours
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
                // A directly supported global/infinite warp is a censored
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
    Global([f32; 2]),
    Refined {
        global: [f32; 2],
        point: [f32; 2],
        confidence: f32,
        measured: bool,
    },
    Occluded {
        global: [f32; 2],
        measured: bool,
    },
    Boundary([f32; 2]),
}

#[derive(Clone, Copy)]
struct ViewScore {
    score: f32,
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
    /// Label zero is the existing global warp; remaining labels are finite
    /// depths ordered from far to near in uniform inverse-depth increments.
    labels: Vec<Option<f64>>,
    scores: Vec<f32>,
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
    tested: Vec<bool>,
}

/// Refine every accepted calibrated target against one shared dense reference
/// depth field. The global warp remains unchanged outside calibrated overlap.
pub fn refine_multiview_depth(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &mut [ModuleAlignment],
    options: &DepthOptions,
) -> Option<DenseDepthMap> {
    if !valid_options(options)
        || inputs.len() != alignments.len()
        || reference_index >= inputs.len()
        || inputs[reference_index].camera.is_none()
    {
        return None;
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
    );
    let DirectDepthField {
        columns,
        rows,
        step,
        nodes: mut field,
        guidance,
        tested,
    } = direct;
    reject_isolated_direct_depths(&mut field, &guidance, columns, rows, inverse_step * 2.5);
    reject_small_direct_components(
        &mut field,
        &guidance,
        columns,
        rows,
        inverse_step * 2.5,
        MINIMUM_DIRECT_COMPONENT_NODES,
    );
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
    let fallback_nodes = tested_nodes.saturating_sub(measured_nodes + regularized_nodes);
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

    for target_index in 0..alignments.len() {
        if target_index == reference_index
            || !alignments[target_index].report.accepted
            || inputs[target_index].camera.is_none()
        {
            continue;
        }
        let global = alignments[target_index].warp.clone();
        let mut decisions = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                let p = [(column * step) as f64, (row * step) as f64];
                let Some(global_q) = global.map(p[0] as f32, p[1] as f32) else {
                    decisions.push(NodeWarp::Undefined);
                    continue;
                };
                let Some(node) = field[index] else {
                    decisions.push(NodeWarp::Global(global_q));
                    continue;
                };
                let selected = refine_one_view(
                    reference,
                    &inputs[target_index],
                    &global,
                    p,
                    node.depth,
                    inverse_step,
                    options,
                );
                let baseline = score_one_view(
                    reference,
                    &inputs[target_index],
                    &global,
                    p,
                    None,
                    options.patch_radius,
                );
                let supported = selected.is_some_and(|selected| {
                    selected.score >= options.minimum_score
                        && baseline.is_none_or(|baseline| selected.score + 0.01 >= baseline.score)
                });
                let contradicted = match (selected, baseline) {
                    (Some(selected), Some(baseline)) => selected.score + 0.12 < baseline.score,
                    _ => false,
                };
                if contradicted {
                    decisions.push(NodeWarp::Occluded {
                        global: global_q,
                        measured: !node.regularized,
                    });
                } else if supported {
                    let selected = selected.unwrap();
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
                    decisions.push(NodeWarp::Global(global_q));
                }
            }
        }
        enforce_warp_consensus(&mut decisions, columns, rows);
        suppress_warp_boundaries(&mut decisions, &guidance, columns, rows);
        let mut points = Vec::with_capacity(columns * rows);
        let mut confidence = Vec::with_capacity(columns * rows);
        let mut refined_nodes = 0usize;
        let mut occluded_nodes = 0usize;
        let mut boundary_nodes = 0usize;
        for decision in decisions {
            match decision {
                NodeWarp::Undefined => {
                    points.push([f32::NAN; 2]);
                    confidence.push(0.0);
                }
                NodeWarp::Global(point) => {
                    points.push(point);
                    confidence.push(1.0);
                }
                NodeWarp::Refined {
                    point,
                    confidence: c,
                    ..
                } => {
                    points.push(point);
                    confidence.push(c);
                    refined_nodes += 1;
                }
                NodeWarp::Occluded { global, .. } => {
                    points.push(global);
                    confidence.push(0.0);
                    occluded_nodes += 1;
                }
                NodeWarp::Boundary(global) => {
                    points.push(global);
                    confidence.push(0.0);
                    boundary_nodes += 1;
                }
            }
        }
        alignments[target_index].warp = Warp {
            step,
            columns,
            rows,
            points,
            confidence,
        };
        alignments[target_index].report.depth = Some(DepthAlignmentReport {
            tested_nodes,
            measured_nodes,
            regularized_nodes,
            fallback_nodes,
            refined_nodes,
            occluded_nodes,
            boundary_nodes,
            reconstructed_fraction: fraction(measured_nodes + regularized_nodes, tested_nodes),
            refined_fraction: fraction(refined_nodes, tested_nodes),
            occluded_fraction: fraction(occluded_nodes, tested_nodes),
            median_depth: median(&selected_depths),
            median_score_improvement: median(&improvements),
        });
    }
    Some(DenseDepthMap {
        columns,
        rows,
        step,
        near_depth: options.near_depth,
        far_depth: options.far_depth,
        nodes: field
            .into_iter()
            .zip(tested)
            .map(|(node, tested)| match node {
                Some(node) => DenseDepthNode {
                    depth: Some(node.depth),
                    confidence: node.confidence,
                    provenance: if node.regularized {
                        DepthProvenance::Regularized
                    } else {
                        DepthProvenance::Measured
                    },
                },
                None if tested => DenseDepthNode {
                    depth: None,
                    confidence: 1.0,
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

fn enforce_warp_consensus(decisions: &mut [NodeWarp], columns: usize, rows: usize) {
    let source = decisions.to_vec();
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let (kind, global, minimum_neighbours) = match source[index] {
                NodeWarp::Refined {
                    global, measured, ..
                } => (0, global, if measured { 2 } else { 4 }),
                NodeWarp::Occluded { global, measured } => {
                    (1, global, if measured { 3 } else { 5 })
                }
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
                decisions[index] = NodeWarp::Global(global);
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
            let Some((global, _)) = warp_mapping(source[index]) else {
                continue;
            };
            decisions[index] = NodeWarp::Boundary(global);
        }
    }
}

fn warp_mapping(decision: NodeWarp) -> Option<([f32; 2], [f32; 2])> {
    match decision {
        NodeWarp::Global(point) => Some((point, point)),
        NodeWarp::Refined { global, point, .. } => Some((global, point)),
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
) -> DirectDepthField {
    let step = (coarse_step / 2).max(4);
    let columns = inputs[reference_index].width.div_ceil(step) + 1;
    let rows = inputs[reference_index].height.div_ceil(step) + 1;
    let automatic_workers = thread::available_parallelism().map_or(1, usize::from);
    let worker_count = automatic_workers.clamp(1, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let direct_options = DepthOptions {
        patch_radius: options.patch_radius.max(3),
        ..options.clone()
    };
    let wide_options = DepthOptions {
        patch_radius: (options.patch_radius * 2).max(8),
        ..options.clone()
    };
    let chunks = thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(rows_per_worker)
            .map(|first_row| {
                let last_row = (first_row + rows_per_worker).min(rows);
                let direct_options = &direct_options;
                let wide_options = &wide_options;
                scope.spawn(move || {
                    let capacity = (last_row - first_row) * columns;
                    let mut nodes = Vec::with_capacity(capacity);
                    let mut guidance = Vec::with_capacity(capacity);
                    let mut tested = Vec::with_capacity(capacity);
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
                            let candidates = direct_depth_candidates(
                                seed.map(|node| node.depth),
                                inverse_step,
                                direct_options,
                            );
                            let baseline = aggregate(
                                &score_views(
                                    inputs,
                                    reference_index,
                                    alignments,
                                    p,
                                    None,
                                    direct_options,
                                ),
                                direct_options,
                            );
                            let mut scores = Vec::with_capacity(candidates.len());
                            for &depth in &candidates {
                                let score = aggregate(
                                    &score_views(
                                        inputs,
                                        reference_index,
                                        alignments,
                                        p,
                                        Some(depth),
                                        direct_options,
                                    ),
                                    direct_options,
                                );
                                scores.push(score);
                            }
                            let node_tested =
                                baseline.is_some() || scores.iter().any(Option::is_some);
                            let mut selected = select_direct_depth(
                                &candidates,
                                &scores,
                                baseline,
                                seed.is_some(),
                                direct_options,
                            );
                            if selected.is_none()
                                && let Some(best) =
                                    best_depth_candidate(&candidates, &scores, direct_options)
                                && scores[best]
                                    .is_some_and(|score| score >= MINIMUM_REGULARIZED_SCORE)
                            {
                                let first = best.saturating_sub(3);
                                let last = (best + 3).min(candidates.len() - 1);
                                let wide_candidates = &candidates[first..=last];
                                let wide_baseline = aggregate(
                                    &score_views(
                                        inputs,
                                        reference_index,
                                        alignments,
                                        p,
                                        None,
                                        wide_options,
                                    ),
                                    wide_options,
                                );
                                let wide_scores = wide_candidates
                                    .iter()
                                    .map(|&depth| {
                                        aggregate(
                                            &score_views(
                                                inputs,
                                                reference_index,
                                                alignments,
                                                p,
                                                Some(depth),
                                                wide_options,
                                            ),
                                            wide_options,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                selected = select_direct_depth(
                                    wide_candidates,
                                    &wide_scores,
                                    wide_baseline,
                                    seed.is_some(),
                                    wide_options,
                                );
                            }
                            nodes.push(selected);
                            tested.push(node_tested);
                        }
                    }
                    (nodes, guidance, tested)
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
    for (chunk_nodes, chunk_guidance, chunk_tested) in chunks {
        nodes.extend(chunk_nodes);
        guidance.extend(chunk_guidance);
        tested.extend(chunk_tested);
    }
    DirectDepthField {
        columns,
        rows,
        step,
        nodes,
        guidance,
        tested,
    }
}

/// Remove isolated measurements that cannot be reproduced at adjacent image
/// positions. This is a consistency test, not completion: rejected nodes
/// become explicit global fallback and no missing node receives a depth.
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

fn direct_depth_candidates(
    seed: Option<f64>,
    inverse_step: f64,
    options: &DepthOptions,
) -> Vec<f64> {
    let far_inverse = 1.0 / options.far_depth;
    let near_inverse = 1.0 / options.near_depth;
    match seed {
        Some(depth) => (-6..=6)
            .map(|offset| {
                let inverse = (1.0 / depth + offset as f64 * inverse_step * 0.5)
                    .clamp(far_inverse, near_inverse);
                1.0 / inverse
            })
            .fold(Vec::new(), |mut depths, depth| {
                if depths
                    .last()
                    .is_none_or(|last: &f64| (1.0 / *last - 1.0 / depth).abs() > inverse_step * 0.1)
                {
                    depths.push(depth);
                }
                depths
            }),
        None => {
            let mut depths = inverse_depth_samples(options.near_depth, options.far_depth, 32);
            depths.reverse();
            depths
        }
    }
}

fn select_direct_depth(
    depths: &[f64],
    scores: &[Option<f32>],
    baseline: Option<f32>,
    seeded: bool,
    options: &DepthOptions,
) -> Option<NodeDepth> {
    let far_inverse = 1.0 / options.far_depth;
    let inverse_range = 1.0 / options.near_depth - far_inverse;
    let mut ranked = depths
        .iter()
        .copied()
        .zip(scores.iter().copied())
        .enumerate()
        .filter_map(|(index, (depth, score))| {
            let score = score?;
            let near_fraction = ((1.0 / depth - far_inverse) / inverse_range).clamp(0.0, 1.0);
            let objective = score - NEAR_DEPTH_PRIOR * near_fraction as f32;
            Some((index, depth, score, objective))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.3.total_cmp(&left.3));
    let &(best_index, depth, score, best_objective) = ranked.first()?;
    let competing = ranked
        .iter()
        .find(|(index, _, _, _)| index.abs_diff(best_index) > 1)
        .map_or(best_objective, |candidate| candidate.3);
    let margin = (best_objective - competing).max(0.0);
    let improvement = baseline.map_or(0.0, |baseline| score - baseline);
    let score_floor = (options.minimum_score - 0.03).max(MINIMUM_REGULARIZED_SCORE);
    let required_margin = if seeded {
        options.minimum_margin * 0.6
    } else {
        options.minimum_margin
    };
    // A finite label being unique among other finite labels is insufficient:
    // distant textured surfaces can have an extremely shallow cost curve and
    // acquire a coherent but fictitious finite depth. Require the candidate
    // to improve measurably on the already fitted global warp. When the global
    // patch is unavailable, uniqueness remains the only usable evidence.
    let improves_global = baseline.is_none_or(|_| improvement >= options.minimum_improvement * 0.5);
    let supported = score >= score_floor
        && improves_global
        && (margin >= required_margin || improvement >= options.minimum_improvement);
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
    scores: &[Option<f32>],
    options: &DepthOptions,
) -> Option<usize> {
    let far_inverse = 1.0 / options.far_depth;
    let inverse_range = 1.0 / options.near_depth - far_inverse;
    depths
        .iter()
        .zip(scores)
        .enumerate()
        .filter_map(|(index, (&depth, &score))| {
            let score = score?;
            let near_fraction = ((1.0 / depth - far_inverse) / inverse_range).clamp(0.0, 1.0);
            Some((index, score - NEAR_DEPTH_PRIOR * near_fraction as f32))
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
) -> CostVolume {
    let mut finite_depths =
        inverse_depth_samples(options.near_depth, options.far_depth, options.planes);
    finite_depths.reverse();
    let labels = std::iter::once(None)
        .chain(finite_depths.into_iter().map(Some))
        .collect::<Vec<_>>();
    let label_count = labels.len();
    let node_count = columns * rows;
    let automatic_workers = thread::available_parallelism().map_or(1, usize::from);
    let worker_count = automatic_workers.clamp(1, rows);
    let rows_per_worker = rows.div_ceil(worker_count);
    let chunks = thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(rows_per_worker)
            .map(|first_row| {
                let last_row = (first_row + rows_per_worker).min(rows);
                let labels = &labels;
                scope.spawn(move || {
                    let chunk_nodes = (last_row - first_row) * columns;
                    let mut scores = Vec::with_capacity(chunk_nodes * label_count);
                    let mut costs = Vec::with_capacity(chunk_nodes * label_count);
                    let mut guidance = Vec::with_capacity(chunk_nodes);
                    let mut tested = Vec::with_capacity(chunk_nodes);
                    for row in first_row..last_row {
                        for column in 0..columns {
                            let p = [(column * step) as f64, (row * step) as f64];
                            guidance.push(reference_guidance(&inputs[reference_index], p));
                            let mut node_tested = false;
                            for (label, &depth) in labels.iter().enumerate() {
                                let views = score_views(
                                    inputs,
                                    reference_index,
                                    alignments,
                                    p,
                                    depth,
                                    options,
                                );
                                let score = aggregate(&views, options).unwrap_or(f32::NAN);
                                node_tested |= score.is_finite();
                                scores.push(score);
                                costs.push(if score.is_finite() {
                                    let near_prior = if label == 0 {
                                        0.0
                                    } else {
                                        NEAR_DEPTH_PRIOR * (label - 1) as f32
                                            / (label_count - 2) as f32
                                    };
                                    (1.0 - score).clamp(0.0, 2.0) + near_prior
                                } else {
                                    MISSING_COST
                                });
                            }
                            tested.push(node_tested);
                        }
                    }
                    (scores, costs, guidance, tested)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("depth cost worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut scores = Vec::with_capacity(node_count * label_count);
    let mut costs = Vec::with_capacity(node_count * label_count);
    let mut guidance = Vec::with_capacity(node_count);
    let mut tested = Vec::with_capacity(node_count);
    for (chunk_scores, chunk_costs, chunk_guidance, chunk_tested) in chunks {
        scores.extend(chunk_scores);
        costs.extend(chunk_costs);
        guidance.extend(chunk_guidance);
        tested.extend(chunk_tested);
    }
    CostVolume {
        labels,
        scores,
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
        // The global warp is a useful photometric baseline but not a depth
        // hypothesis: it already absorbs an unknown scene plane. Select among
        // finite calibrated planes and use the global score to reject only a
        // clearly worse reconstruction. This lets SGM infer depth throughout
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
        let improvement = if baseline.is_finite() {
            score - baseline
        } else {
            0.0
        };
        if baseline.is_finite() && improvement < -MAXIMUM_REGULARIZED_BASELINE_LOSS {
            continue;
        }
        let measured = score >= options.minimum_score
            && improvement >= options.minimum_improvement * 0.5
            && (margin >= options.minimum_margin || improvement >= options.minimum_improvement);
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

fn score_views(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    centre: Vec2,
    depth: Option<f64>,
    options: &DepthOptions,
) -> Vec<ViewScore> {
    (0..inputs.len())
        .filter(|&index| {
            index != reference_index
                && alignments[index].report.accepted
                && inputs[index].camera.is_some()
        })
        .filter_map(|index| {
            score_one_view(
                &inputs[reference_index],
                &inputs[index],
                &alignments[index].warp,
                centre,
                depth,
                options.patch_radius,
            )
        })
        .collect()
}

fn score_one_view(
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
        Some(depth) => map_at_depth(reference_camera, target_camera, global, centre, depth)?,
        None => [f64::from(global_centre[0]), f64::from(global_centre[1])],
    };
    let delta = [
        mapped_centre[0] - f64::from(global_centre[0]),
        mapped_centre[1] - f64::from(global_centre[1]),
    ];
    if delta[0].abs() > 128.0 || delta[1].abs() > 128.0 {
        return None;
    }
    Some(ViewScore {
        score: warped_patch_zncc(
            ViewPair {
                reference: reference.luminance,
                target: target.luminance,
                global,
            },
            centre,
            delta,
            patch_radius,
        )?,
    })
}

/// Refine the shared multiview depth for one target camera, then permit a
/// small image-space residual. The latter absorbs calibration interpolation
/// error and mild lens/model error, but is deliberately bounded so it cannot
/// turn into unconstrained optical flow or attach a foreground edge to an
/// unrelated background texture.
fn refine_one_view(
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
        let point = map_at_depth(
            reference_camera,
            target_camera,
            global,
            centre,
            1.0 / inverse,
        )?;
        let point = [point[0] as f32, point[1] as f32];
        let score = score_one_view_at_point(
            reference,
            target,
            global,
            centre,
            global_point,
            point,
            options.patch_radius,
        )?;
        // Prefer the shared multiview solution when photometric evidence is
        // effectively tied. This prevents independent cameras from drifting
        // to different repeated textures.
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
            let score = score_one_view_at_point(
                reference,
                target,
                global,
                centre,
                global_point,
                point,
                options.patch_radius,
            )?;
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

fn score_one_view_at_point(
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

fn map_at_depth(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    global: &Warp,
    pixel: Vec2,
    depth: f64,
) -> Option<Vec2> {
    let global_q = global.map(pixel[0] as f32, pixel[1] as f32)?;
    // Global alignment is composed in reference coordinates as M_inf(C(p)),
    // not as an image-space translation added after M_inf(p). Recover C(p)
    // by applying the inverse calibrated infinity mapping, then project that
    // corrected reference ray at the requested finite depth. The former
    // additive approximation becomes badly wrong for the 30--60 px global
    // corrections commonly needed by real L16 captures.
    let corrected_reference = reference_camera.map_from(
        target_camera,
        [f64::from(global_q[0]), f64::from(global_q[1])],
        INFINITY_DEPTH,
    )?;
    let q = target_camera.map_from(reference_camera, corrected_reference, depth)?;
    (q[0].is_finite() && q[1].is_finite()).then_some(q)
}

fn aggregate(scores: &[ViewScore], options: &DepthOptions) -> Option<f32> {
    if scores.len() < options.minimum_support {
        return None;
    }
    let mut values = scores.iter().map(|score| score.score).collect::<Vec<_>>();
    values.sort_by(|left, right| right.total_cmp(left));
    values.truncate(options.best_view_count.min(values.len()));
    (values.len() >= options.minimum_support)
        .then(|| values.iter().sum::<f32>() / values.len() as f32)
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
            // Differential parallax across this small patch is negligible.
            // Project its centre through the full calibration once, then
            // translate nearby global-warp samples by that displacement.
            let global_q = pair.global.map(p[0] as f32, p[1] as f32)?;
            let q = [
                f64::from(global_q[0]) + delta[0],
                f64::from(global_q[1]) + delta[1],
            ];
            let target_value = pair
                .target
                .sample(((q[0] - 0.5) * 0.5) as f32, ((q[1] - 0.5) * 0.5) as f32)?;
            // A larger direct support window can measure a flat interior from
            // texture elsewhere on the same surface. Reference-domain range
            // weighting prevents that support from simply crossing a visible
            // foreground/background edge.
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

fn valid_options(options: &DepthOptions) -> bool {
    options.enabled
        && options.grid_step > 0
        && options.near_depth.is_finite()
        && options.far_depth.is_finite()
        && options.near_depth > 0.0
        && options.far_depth > options.near_depth
        && options.planes >= 3
        && options.patch_radius > 0
        && options.minimum_support > 0
        && options.best_view_count >= options.minimum_support
        && options.minimum_neighbour_support > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_depth_planes_include_bounds_and_favour_near_resolution() {
        let depths = inverse_depth_samples(500.0, 100_000.0, 5);
        assert!((depths[0] - 500.0).abs() < 1e-9);
        assert!((depths[4] - 100_000.0).abs() < 1e-6);
        assert!(depths.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(depths[1] - depths[0] < depths[4] - depths[3]);
    }

    #[test]
    fn sublabel_fit_refines_inverse_depth_between_planes() {
        let labels = [None, Some(10_000.0), Some(5_000.0), Some(10_000.0 / 3.0)];
        let depth = sublabel_depth(&labels, &[9.0, 2.0, 0.0, 1.0], 2);
        assert!(depth < 5_000.0);
        assert!(depth > 10_000.0 / 3.0);
    }

    #[test]
    fn aggregation_requires_independent_view_support() {
        let options = DepthOptions::default();
        let one = [ViewScore { score: 0.9 }];
        assert_eq!(aggregate(&one, &options), None);
        let two = [ViewScore { score: 0.9 }, ViewScore { score: 0.7 }];
        assert!((aggregate(&two, &options).unwrap() - 0.8).abs() < 1.0e-6);
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
            NodeWarp::Global([10.0, 20.0])
        ));

        let mut isolated = vec![NodeWarp::Global([10.0, 20.0]); 9];
        isolated[4] = refined(2.0);
        enforce_warp_consensus(&mut isolated, 3, 3);
        assert!(matches!(isolated[4], NodeWarp::Global(_)));
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
            labels: vec![None, Some(10_000.0), Some(2_000.0)],
            scores: vec![0.9, 0.7, 0.6],
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
            labels: vec![None, Some(100_000.0), Some(10_000.0), Some(2_000.0)],
            scores: vec![0.72, 0.70, 0.68, 0.62],
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
    fn direct_depth_must_improve_on_the_global_warp() {
        let depths = [10_000.0, 9_000.0, 8_000.0, 7_000.0, 6_000.0];
        let scores = [Some(0.60), Some(0.65), Some(0.90), Some(0.65), Some(0.60)];
        let options = DepthOptions::default();

        // The finite winner is extremely clear relative to the other finite
        // labels, but it is indistinguishable from the global mapping.
        assert!(select_direct_depth(&depths, &scores, Some(0.899), false, &options).is_none());

        // The same unique label is accepted once it also improves on global.
        assert!(select_direct_depth(&depths, &scores, Some(0.89), false, &options).is_some());
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
