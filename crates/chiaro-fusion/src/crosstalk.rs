//! Capture-adaptive residual correction on top of the factory 4x4 CFA-phase
//! crosstalk mesh.
//!
//! The scene cannot identify sixteen independent coefficients at every mesh
//! node. This module therefore fits five small, strongly regularized modes
//! from smooth, aligned multi-camera observations. The modes operate in the
//! current white-balance domain and are conjugated back into RAW space before
//! they are composed with the factory matrix.

use std::{fmt, str::FromStr};

use chiaro_hotpixel_core::highlight::HighlightRecoveryState;
use serde::{Deserialize, Serialize};

use crate::{
    align::ModuleAlignment, calibration::CrosstalkMesh, image::Mosaic, synth::ModuleColor,
};

const FIT_COLUMNS: usize = 17;
const FIT_ROWS: usize = 13;
const MIN_SAMPLES: usize = 96;
const MIN_CELLS: usize = 24;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CrosstalkMode {
    None,
    Factory,
    #[default]
    Adaptive,
}

impl CrosstalkMode {
    pub const ALL: [Self; 3] = [Self::None, Self::Factory, Self::Adaptive];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Factory => "Factory",
            Self::Adaptive => "Adaptive",
        }
    }
}

impl fmt::Display for CrosstalkMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::Factory => "factory",
            Self::Adaptive => "adaptive",
        })
    }
}

impl FromStr for CrosstalkMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "factory" => Ok(Self::Factory),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(format!(
                "unknown crosstalk mode {value:?}; expected none, factory, or adaptive"
            )),
        }
    }
}

/// Five identifiable residual modes. Red and blue modes pull their corrected
/// channel toward or away from the mean green; radial terms vary that action
/// from optical centre to corner. Green balance mixes only the two green CFA
/// phases and preserves their mean.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CrosstalkResidual {
    pub red_global: f32,
    pub blue_global: f32,
    pub green_balance: f32,
    pub red_radial: f32,
    pub blue_radial: f32,
}

impl CrosstalkResidual {
    fn as_array(self) -> [f32; 5] {
        [
            self.red_global,
            self.blue_global,
            self.green_balance,
            self.red_radial,
            self.blue_radial,
        ]
    }

    fn from_array(values: [f32; 5]) -> Self {
        Self {
            red_global: values[0],
            blue_global: values[1],
            green_balance: values[2],
            red_radial: values[3],
            blue_radial: values[4],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AdaptiveCrosstalkReport {
    pub mode: CrosstalkMode,
    pub factory_available: bool,
    pub adapted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub samples: usize,
    pub fit_samples: usize,
    pub validation_samples: usize,
    pub observed_grid_cells: usize,
    pub grid: [usize; 2],
    pub before_log_chroma_rmse: f32,
    pub after_log_chroma_rmse: f32,
    pub residual: CrosstalkResidual,
    pub awb_gains: [f32; 3],
    pub capture_gain: f32,
    pub exposure_ns: u64,
}

pub struct CrosstalkFitSource<'a> {
    pub mosaic: &'a Mosaic,
    pub highlight: &'a HighlightRecoveryState,
    pub alignment: &'a ModuleAlignment,
    pub color: ModuleColor,
    pub capture_gain: f32,
    pub exposure_ns: u64,
}

pub struct CrosstalkFit {
    pub mesh: Option<CrosstalkMesh>,
    pub report: AdaptiveCrosstalkReport,
}

#[derive(Clone, Copy)]
struct Observation {
    phases: [f32; 4],
    reference_chroma: [f32; 2],
    radial: f32,
    weight: f32,
    cell: usize,
    validation: bool,
}

/// Fit one conservative residual per non-reference camera. The reference is
/// intentionally kept at its factory calibration: a single capture can
/// measure inter-camera disagreement, but cannot distinguish an absolute
/// reference-camera colour error from the colour of the scene.
pub fn fit_adaptive_crosstalk(
    sources: &[CrosstalkFitSource<'_>],
    reference_index: usize,
    mode: CrosstalkMode,
    reference_width: usize,
    reference_height: usize,
) -> Vec<CrosstalkFit> {
    sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let factory = source.mosaic.crosstalk.clone();
            let mut report = AdaptiveCrosstalkReport {
                mode,
                factory_available: factory.is_some(),
                adapted: false,
                reason: None,
                samples: 0,
                fit_samples: 0,
                validation_samples: 0,
                observed_grid_cells: 0,
                grid: [FIT_COLUMNS, FIT_ROWS],
                before_log_chroma_rmse: 0.0,
                after_log_chroma_rmse: 0.0,
                residual: CrosstalkResidual::default(),
                awb_gains: source.color.wb_gains,
                capture_gain: source.capture_gain,
                exposure_ns: source.exposure_ns,
            };
            if mode == CrosstalkMode::None {
                report.reason = Some("disabled".to_owned());
                return CrosstalkFit { mesh: None, report };
            }
            let Some(factory) = factory else {
                report.reason = Some("factory crosstalk mesh unavailable".to_owned());
                return CrosstalkFit { mesh: None, report };
            };
            if mode == CrosstalkMode::Factory {
                report.reason = Some("factory-only mode".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }
            if index == reference_index {
                report.reason = Some("reference camera retained as the colour anchor".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }
            if reference_index >= sources.len()
                || !source.alignment.report.accepted
                || !sources[reference_index].alignment.report.accepted
                || !source.color.calibrated
                || !sources[reference_index].color.calibrated
            {
                report.reason = Some("no accepted calibrated reference overlap".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }

            let observations = collect_observations(
                &sources[reference_index],
                source,
                reference_width,
                reference_height,
            );
            report.samples = observations.len();
            report.observed_grid_cells = observed_cells(&observations);
            let fit_observations = observations
                .iter()
                .copied()
                .filter(|observation| !observation.validation)
                .collect::<Vec<_>>();
            let validation_observations = observations
                .iter()
                .copied()
                .filter(|observation| observation.validation)
                .collect::<Vec<_>>();
            report.fit_samples = fit_observations.len();
            report.validation_samples = validation_observations.len();
            if report.samples < MIN_SAMPLES || report.observed_grid_cells < MIN_CELLS {
                report.reason = Some("insufficient smooth overlap".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }
            if fit_observations.len() < 64 || validation_observations.len() < 24 {
                report.reason = Some("insufficient fit/validation split".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }

            let residual = optimize_residual(&fit_observations, source.color);
            let before = chroma_rmse(
                &validation_observations,
                source.color,
                CrosstalkResidual::default(),
            );
            let after = chroma_rmse(&validation_observations, source.color, residual);
            report.before_log_chroma_rmse = before;
            report.after_log_chroma_rmse = after;
            report.residual = residual;
            let improvement = if before > 1e-6 {
                (before - after) / before
            } else {
                0.0
            };
            if improvement < 0.002 {
                report.reason = Some("residual did not improve validation observations".to_owned());
                return CrosstalkFit {
                    mesh: Some(factory),
                    report,
                };
            }
            report.adapted = true;
            CrosstalkFit {
                mesh: Some(compose_residual(&factory, residual, source.color.wb_gains)),
                report,
            }
        })
        .collect()
}

fn collect_observations(
    reference: &CrosstalkFitSource<'_>,
    target: &CrosstalkFitSource<'_>,
    width: usize,
    height: usize,
) -> Vec<Observation> {
    let step = (width.min(height) / 96).clamp(24, 48);
    let margin = step * 2;
    let mut observations = Vec::new();
    for y in (margin..height.saturating_sub(margin)).step_by(step) {
        for x in (margin..width.saturating_sub(margin)).step_by(step) {
            let (x, y) = (x as f32, y as f32);
            if reference.alignment.warp.confidence(x, y) < 0.75
                || target.alignment.warp.confidence(x, y) < 0.75
            {
                continue;
            }
            let (Some(rq), Some(tq)) = (
                reference.alignment.warp.map(x, y),
                target.alignment.warp.map(x, y),
            ) else {
                continue;
            };
            let Some((reference_phases, reference_weight)) = smooth_sample(reference, rq) else {
                continue;
            };
            let Some((target_phases, target_weight)) = smooth_sample(target, tq) else {
                continue;
            };
            let Some(reference_chroma) =
                log_chroma(reference.color.to_xyz(phases_to_rgb(reference_phases)))
            else {
                continue;
            };
            let nx = tq[0] / (target.mosaic.width - 1) as f32 - 0.5;
            let ny = tq[1] / (target.mosaic.height - 1) as f32 - 0.5;
            let radial = (2.0 * (nx * nx + ny * ny)).clamp(0.0, 1.0);
            let column = ((tq[0] / target.mosaic.width as f32) * FIT_COLUMNS as f32)
                .floor()
                .clamp(0.0, (FIT_COLUMNS - 1) as f32) as usize;
            let row = ((tq[1] / target.mosaic.height as f32) * FIT_ROWS as f32)
                .floor()
                .clamp(0.0, (FIT_ROWS - 1) as f32) as usize;
            observations.push(Observation {
                phases: target_phases,
                reference_chroma,
                radial,
                weight: reference_weight.min(target_weight),
                cell: row * FIT_COLUMNS + column,
                validation: (((x as usize / step) + 3 * (y as usize / step)) & 3) == 0,
            });
        }
    }
    observations
}

fn smooth_sample(source: &CrosstalkFitSource<'_>, point: [f32; 2]) -> Option<([f32; 4], f32)> {
    let (centre, confidence) =
        source
            .mosaic
            .sample_factory_phases(point[0], point[1], source.highlight)?;
    if confidence != 255
        || centre.iter().any(|value| !value.is_finite())
        || centre.iter().copied().fold(0.0f32, f32::max) >= 0.90
    {
        return None;
    }
    let green = ((centre[1] + centre[2]) * 0.5).max(1e-5);
    if green < 0.03 {
        return None;
    }
    let centre_chroma = [
        (centre[0].max(1e-5) / green).ln(),
        (centre[3].max(1e-5) / green).ln(),
    ];
    let mut variation = 0.0f32;
    for (dx, dy) in [(-4.0, 0.0), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0)] {
        let (neighbour, neighbour_confidence) =
            source
                .mosaic
                .sample_factory_phases(point[0] + dx, point[1] + dy, source.highlight)?;
        if neighbour_confidence != 255 {
            return None;
        }
        let neighbour_green = ((neighbour[1] + neighbour[2]) * 0.5).max(1e-5);
        let neighbour_chroma = [
            (neighbour[0].max(1e-5) / neighbour_green).ln(),
            (neighbour[3].max(1e-5) / neighbour_green).ln(),
        ];
        variation = variation
            .max((neighbour_green / green).ln().abs())
            .max((neighbour_chroma[0] - centre_chroma[0]).abs())
            .max((neighbour_chroma[1] - centre_chroma[1]).abs());
    }
    if variation >= 0.10 {
        return None;
    }
    let smooth = smoothstep((0.10 - variation) / 0.08);
    let midtone = smoothstep((green - 0.03) / 0.12) * smoothstep((0.90 - green) / 0.20);
    Some((centre, smooth * midtone))
}

fn phases_to_rgb(phases: [f32; 4]) -> [f32; 3] {
    [phases[0], (phases[1] + phases[2]) * 0.5, phases[3]]
}

fn log_chroma(xyz: [f32; 3]) -> Option<[f32; 2]> {
    if xyz.iter().any(|value| !value.is_finite()) || xyz[1] <= 1e-5 {
        return None;
    }
    let x = xyz[0] / xyz[1];
    let z = xyz[2] / xyz[1];
    (x > 1e-5 && z > 1e-5).then(|| [x.ln(), z.ln()])
}

fn observed_cells(observations: &[Observation]) -> usize {
    let mut cells = [false; FIT_COLUMNS * FIT_ROWS];
    for observation in observations {
        cells[observation.cell] = true;
    }
    cells.into_iter().filter(|value| *value).count()
}

fn cell_counts(observations: &[Observation]) -> [usize; FIT_COLUMNS * FIT_ROWS] {
    let mut counts = [0usize; FIT_COLUMNS * FIT_ROWS];
    for observation in observations {
        counts[observation.cell] += 1;
    }
    counts
}

fn optimize_residual(observations: &[Observation], color: ModuleColor) -> CrosstalkResidual {
    let counts = cell_counts(observations);
    let mut values = [0.0f32; 5];
    let mut steps = [0.012, 0.012, 0.015, 0.010, 0.010];
    let bounds = [0.08, 0.08, 0.10, 0.06, 0.06];
    let mut best = objective(observations, &counts, color, CrosstalkResidual::default());
    for _ in 0..8 {
        for parameter in 0..values.len() {
            let original = values[parameter];
            for direction in [-1.0f32, 1.0] {
                let mut candidate = values;
                candidate[parameter] = (original + direction * steps[parameter])
                    .clamp(-bounds[parameter], bounds[parameter]);
                let loss = objective(
                    observations,
                    &counts,
                    color,
                    CrosstalkResidual::from_array(candidate),
                );
                if loss < best {
                    values = candidate;
                    best = loss;
                }
            }
        }
        for step in &mut steps {
            *step *= 0.55;
        }
    }
    CrosstalkResidual::from_array(values)
}

fn objective(
    observations: &[Observation],
    counts: &[usize; FIT_COLUMNS * FIT_ROWS],
    color: ModuleColor,
    residual: CrosstalkResidual,
) -> f32 {
    let mut sum = 0.0;
    let mut total = 0.0;
    for observation in observations {
        let phases = apply_residual(
            observation.phases,
            residual,
            observation.radial,
            color.wb_gains,
        );
        let Some(chroma) = log_chroma(color.to_xyz(phases_to_rgb(phases))) else {
            continue;
        };
        let weight = observation.weight / counts[observation.cell].max(1) as f32;
        let error = [
            chroma[0] - observation.reference_chroma[0],
            chroma[1] - observation.reference_chroma[1],
        ];
        sum += weight * (huber(error[0], 0.08) + huber(error[1], 0.08));
        let balanced_green = [phases[1] * color.wb_gains[1], phases[2] * color.wb_gains[1]];
        if balanced_green[0] > 1e-5 && balanced_green[1] > 1e-5 {
            sum += weight * 0.15 * huber((balanced_green[0] / balanced_green[1]).ln(), 0.05);
        }
        total += weight;
    }
    let values = residual.as_array();
    let sigma = [0.04, 0.04, 0.05, 0.03, 0.03];
    let regularization = values
        .into_iter()
        .zip(sigma)
        .map(|(value, sigma)| (value / sigma).powi(2))
        .sum::<f32>();
    sum / total.max(1e-6) + regularization * 0.0005
}

fn chroma_rmse(
    observations: &[Observation],
    color: ModuleColor,
    residual: CrosstalkResidual,
) -> f32 {
    let mut squared = 0.0;
    let mut count = 0usize;
    for observation in observations {
        let phases = apply_residual(
            observation.phases,
            residual,
            observation.radial,
            color.wb_gains,
        );
        if let Some(chroma) = log_chroma(color.to_xyz(phases_to_rgb(phases))) {
            squared += (chroma[0] - observation.reference_chroma[0]).powi(2)
                + (chroma[1] - observation.reference_chroma[1]).powi(2);
            count += 2;
        }
    }
    (squared / count.max(1) as f32).sqrt()
}

fn apply_residual(
    phases: [f32; 4],
    residual: CrosstalkResidual,
    radial: f32,
    wb: [f32; 3],
) -> [f32; 4] {
    let gains = [wb[0], wb[1], wb[1], wb[2]].map(|value| value.max(1e-3));
    let mut balanced: [f32; 4] = std::array::from_fn(|phase| phases[phase] * gains[phase]);
    let input = balanced;
    let mean_green = (input[1] + input[2]) * 0.5;
    let red = (residual.red_global + residual.red_radial * radial).clamp(-0.12, 0.12);
    let blue = (residual.blue_global + residual.blue_radial * radial).clamp(-0.12, 0.12);
    let green = residual.green_balance.clamp(-0.12, 0.12);
    balanced[0] = input[0] + red * (mean_green - input[0]);
    balanced[3] = input[3] + blue * (mean_green - input[3]);
    balanced[1] = input[1] + green * (input[2] - input[1]);
    balanced[2] = input[2] + green * (input[1] - input[2]);
    std::array::from_fn(|phase| balanced[phase] / gains[phase])
}

fn compose_residual(
    factory: &CrosstalkMesh,
    residual: CrosstalkResidual,
    wb: [f32; 3],
) -> CrosstalkMesh {
    let gains = [wb[0], wb[1], wb[1], wb[2]].map(|value| value.max(1e-3));
    let mut matrices = Vec::with_capacity(factory.matrices.len());
    for row in 0..factory.rows {
        for column in 0..factory.columns {
            let nx = column as f32 / (factory.columns - 1).max(1) as f32 - 0.5;
            let ny = row as f32 / (factory.rows - 1).max(1) as f32 - 0.5;
            let radial = (2.0 * (nx * nx + ny * ny)).clamp(0.0, 1.0);
            let red = (residual.red_global + residual.red_radial * radial).clamp(-0.12, 0.12);
            let blue = (residual.blue_global + residual.blue_radial * radial).clamp(-0.12, 0.12);
            let green = residual.green_balance.clamp(-0.12, 0.12);
            let mut balanced = [0.0f32; 16];
            for diagonal in 0..4 {
                balanced[diagonal * 4 + diagonal] = 1.0;
            }
            balanced[0] = 1.0 - red;
            balanced[1] = red * 0.5;
            balanced[2] = red * 0.5;
            balanced[15] = 1.0 - blue;
            balanced[13] = blue * 0.5;
            balanced[14] = blue * 0.5;
            balanced[5] = 1.0 - green;
            balanced[6] = green;
            balanced[9] = green;
            balanced[10] = 1.0 - green;

            let mut raw_operator = [0.0f32; 16];
            for output in 0..4 {
                for input in 0..4 {
                    raw_operator[output * 4 + input] =
                        balanced[output * 4 + input] * gains[input] / gains[output];
                }
            }
            let factory_node = &factory.matrices[(row * factory.columns + column) * 16..][..16];
            for output in 0..4 {
                for input in 0..4 {
                    let value = (0..4)
                        .map(|middle| {
                            raw_operator[output * 4 + middle] * factory_node[middle * 4 + input]
                        })
                        .sum();
                    matrices.push(value);
                }
            }
        }
    }
    CrosstalkMesh {
        columns: factory.columns,
        rows: factory.rows,
        matrices,
    }
}

fn huber(error: f32, delta: f32) -> f32 {
    let absolute = error.abs();
    if absolute <= delta {
        0.5 * error * error
    } else {
        delta * (absolute - 0.5 * delta)
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_mesh() -> CrosstalkMesh {
        let mut matrix = vec![0.0; 16];
        for diagonal in 0..4 {
            matrix[diagonal * 4 + diagonal] = 1.0;
        }
        CrosstalkMesh {
            columns: 2,
            rows: 2,
            matrices: matrix.repeat(4),
        }
    }

    #[test]
    fn white_balance_conjugation_matches_direct_balanced_application() {
        let residual = CrosstalkResidual {
            red_global: 0.04,
            blue_global: -0.02,
            green_balance: 0.03,
            red_radial: 0.01,
            blue_radial: -0.01,
        };
        let wb = [2.0, 1.0, 1.4];
        let phases = [0.2, 0.35, 0.36, 0.25];
        let expected = apply_residual(phases, residual, 1.0, wb);
        let mesh = compose_residual(&identity_mesh(), residual, wb);
        let matrix = mesh.matrix(0.0, 0.0, 100, 100);
        let actual: [f32; 4] = std::array::from_fn(|row| {
            (0..4)
                .map(|column| matrix[row * 4 + column] * phases[column])
                .sum::<f32>()
        });
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn residual_fit_recovers_red_and_blue_modes() {
        let color = ModuleColor {
            calibrated: true,
            ..ModuleColor::default()
        };
        let planted = CrosstalkResidual {
            red_global: 0.035,
            blue_global: -0.025,
            ..CrosstalkResidual::default()
        };
        let observations = (0..FIT_COLUMNS * FIT_ROWS)
            .map(|cell| {
                let phases = [0.20, 0.35, 0.35, 0.25];
                let corrected = apply_residual(phases, planted, 0.5, color.wb_gains);
                Observation {
                    phases,
                    reference_chroma: log_chroma(color.to_xyz(phases_to_rgb(corrected))).unwrap(),
                    radial: 0.5,
                    weight: 1.0,
                    cell,
                    validation: false,
                }
            })
            .collect::<Vec<_>>();
        let fit = optimize_residual(&observations, color);
        assert!(fit.red_global + fit.red_radial * 0.5 > 0.01, "{fit:?}");
        assert!(fit.blue_global + fit.blue_radial * 0.5 < -0.003, "{fit:?}");
        assert!(
            chroma_rmse(&observations, color, fit)
                < chroma_rmse(&observations, color, CrosstalkResidual::default()),
            "{fit:?}"
        );
    }
}
