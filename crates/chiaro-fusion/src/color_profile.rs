//! Inspection and fitting of the L16's per-module factory colour records.

use chiaro_proto::{color_calibration::ColorCalibration, lightheader::LightHeader};
use serde::Serialize;

use crate::calibration::{LriMessages, awb_gains, camera_name};
use crate::math::{Mat3, inverse, mul_vec};

/// BabelColor's average of 30 pre-November-2014 ColorChecker Classic charts,
/// in row-major chart order, under D50/2 degree observation. The L16 factory
/// records use the same order: 18 chromatic patches followed by the six neutral
/// patches from white to black.
const COLORCHECKER_LAB_D50: [[f64; 3]; 24] = [
    [37.986, 13.555, 14.059],
    [65.711, 18.130, 17.810],
    [49.927, -4.880, -21.925],
    [43.139, -13.095, 21.905],
    [55.112, 8.844, -25.399],
    [70.719, -33.397, -0.199],
    [62.661, 36.067, 57.096],
    [40.020, 10.410, -45.964],
    [51.124, 48.239, 16.248],
    [30.325, 22.976, -21.587],
    [72.532, -23.709, 57.255],
    [71.941, 19.363, 67.857],
    [28.778, 14.179, -50.297],
    [55.261, -38.342, 31.370],
    [42.101, 53.378, 28.190],
    [81.733, 4.039, 79.819],
    [51.935, 49.986, -14.574],
    [51.038, -28.631, -28.638],
    [96.539, -0.425, 1.186],
    [81.257, -0.638, -0.335],
    [66.766, -0.734, -0.504],
    [50.867, -0.153, -0.270],
    [35.656, -0.421, -1.231],
    [20.461, -0.079, -0.973],
];

const D50_WHITE: [f64; 3] = [0.96422, 1.0, 0.82521];

#[derive(Clone, Debug, Serialize)]
pub struct FactoryColorDump {
    pub capture_awb_gains: Option<[f64; 3]>,
    pub records: Vec<FactoryColorRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactoryColorRecord {
    pub camera: String,
    pub source: &'static str,
    pub illuminant: Option<i32>,
    pub illuminant_name: String,
    pub forward_matrix: Option<[[Option<f32>; 3]; 3]>,
    pub color_matrix: Option<[[Option<f32>; 3]; 3]>,
    pub rg_ratio: Option<f32>,
    pub bg_ratio: Option<f32>,
    pub macbeth_data: Vec<[Option<f32>; 3]>,
    pub illuminant_spd: Vec<[Option<f32>; 2]>,
    pub spectral_data: Option<FactorySpectralData>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorySpectralData {
    pub format: Option<i32>,
    pub format_name: String,
    pub channels: Vec<FactorySpectralChannel>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorySpectralChannel {
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactoryColorAnalysis {
    pub reference: &'static str,
    pub patch_order: &'static str,
    pub profiles: Vec<FactoryProfileReport>,
    pub inter_camera_consistency: Vec<InterCameraConsistency>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InterCameraConsistency {
    pub illuminant: String,
    pub module_count: usize,
    pub current_d65_only_mean_pair_delta_e00: Option<f64>,
    pub illuminant_matched_factory_mean_pair_delta_e00: Option<f64>,
    pub selected_profile_mean_pair_delta_e00: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactoryProfileReport {
    pub camera: String,
    pub source: &'static str,
    pub illuminant: String,
    pub macbeth_point_count: usize,
    pub sample_interpretation: &'static str,
    pub white_patch_camera_rgb: Option<[f64; 3]>,
    pub color_matrix_white_ratio_log_error: Option<f64>,
    pub forward_matrix_d50_white_error: Option<f64>,
    pub current_d65_only: Option<ErrorMetrics>,
    pub factory_forward: Option<ErrorMetrics>,
    pub fitted_matrix_holdout: Option<ErrorMetrics>,
    pub nonlinear_holdout: Option<ErrorMetrics>,
    pub fitted_matrix: Option<Mat3>,
    pub factory_neutral_axis_mean_chroma: Option<f64>,
    pub neutral_axis_mean_chroma: Option<f64>,
    pub selected_profile: &'static str,
    pub fallback_reason: Option<String>,
    pub illuminant_spd_samples: usize,
    pub spectral_channels: usize,
    pub spectral_sample_spacing_nm: Option<f64>,
    pub spectral_channel_maxima: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ErrorMetrics {
    pub mean_delta_e00: f64,
    pub median_delta_e00: f64,
    pub p90_delta_e00: f64,
    pub maximum_delta_e00: f64,
}

impl FactoryColorDump {
    pub fn from_messages(messages: &LriMessages) -> Self {
        let mut records = Vec::new();
        for header in &messages.headers {
            append_header_records(header, &mut records);
        }
        Self {
            capture_awb_gains: awb_gains(messages),
            records,
        }
    }

    /// Evaluate the measured factory target against the existing ForwardMatrix,
    /// a robust linear refit, and a deliberately small quadratic residual. New
    /// candidates are judged using leave-one-patch-out predictions only.
    pub fn analyze(&self) -> FactoryColorAnalysis {
        let profiles = self
            .records
            .iter()
            .map(|record| {
                let d65 = self.records.iter().find(|candidate| {
                    candidate.camera == record.camera
                        && candidate.source == record.source
                        && candidate.illuminant == Some(2)
                });
                analyze_record(record, d65)
            })
            .collect::<Vec<_>>();
        FactoryColorAnalysis {
            reference: "BabelColor average of 30 pre-November-2014 ColorChecker Classic charts, D50/2 degree",
            patch_order: "row-major: 18 chromatic patches, then white-to-black neutrals",
            inter_camera_consistency: inter_camera_consistency(&self.records, &profiles),
            profiles,
        }
    }
}

fn inter_camera_consistency(
    records: &[FactoryColorRecord],
    reports: &[FactoryProfileReport],
) -> Vec<InterCameraConsistency> {
    let mut illuminants = records
        .iter()
        .filter(|record| record.source == "module")
        .map(|record| record.illuminant)
        .collect::<Vec<_>>();
    illuminants.sort();
    illuminants.dedup();
    illuminants
        .into_iter()
        .map(|illuminant| {
            let members = records
                .iter()
                .enumerate()
                .filter(|(_, record)| record.source == "module" && record.illuminant == illuminant)
                .filter_map(|(index, record)| {
                    let camera = balanced_macbeth(record)?;
                    let factory = complete_matrix(record.forward_matrix.as_ref())?;
                    let d65 = records
                        .iter()
                        .find(|candidate| {
                            candidate.source == record.source
                                && candidate.camera == record.camera
                                && candidate.illuminant == Some(2)
                        })
                        .and_then(|candidate| complete_matrix(candidate.forward_matrix.as_ref()))?;
                    let selected = if reports[index].selected_profile == "fitted_linear_matrix" {
                        reports[index].fitted_matrix.unwrap_or(factory)
                    } else {
                        factory
                    };
                    Some((camera, d65, factory, selected))
                })
                .collect::<Vec<_>>();
            InterCameraConsistency {
                illuminant: illuminant_name(illuminant).to_owned(),
                module_count: members.len(),
                current_d65_only_mean_pair_delta_e00: mean_pair_difference(&members, |item| item.1),
                illuminant_matched_factory_mean_pair_delta_e00: mean_pair_difference(
                    &members,
                    |item| item.2,
                ),
                selected_profile_mean_pair_delta_e00: mean_pair_difference(&members, |item| item.3),
            }
        })
        .collect()
}

fn mean_pair_difference<F>(members: &[([[f64; 3]; 24], Mat3, Mat3, Mat3)], matrix: F) -> Option<f64>
where
    F: Fn(&([[f64; 3]; 24], Mat3, Mat3, Mat3)) -> Mat3,
{
    if members.len() < 2 {
        return None;
    }
    let predictions = members
        .iter()
        .map(|member| {
            let transform = matrix(member);
            member
                .0
                .map(|sample| xyz_to_lab(mul_vec(&transform, sample)))
        })
        .collect::<Vec<_>>();
    let mut total = 0.0;
    let mut count = 0usize;
    for first in 0..predictions.len() {
        for second in first + 1..predictions.len() {
            for (&first_patch, &second_patch) in predictions[first].iter().zip(&predictions[second])
            {
                total += delta_e00(first_patch, second_patch);
                count += 1;
            }
        }
    }
    Some(total / count as f64)
}

fn analyze_record(
    record: &FactoryColorRecord,
    d65_record: Option<&FactoryColorRecord>,
) -> FactoryProfileReport {
    let spacing = record.spectral_data.as_ref().and_then(|spectral| {
        let channel = spectral.channels.first()?;
        let (start, end) = (channel.start?, channel.end?);
        (channel.values.len() > 1)
            .then_some((f64::from(end) - f64::from(start)) / (channel.values.len() - 1) as f64)
    });
    let base = FactoryProfileReport {
        camera: record.camera.clone(),
        source: record.source,
        illuminant: record.illuminant_name.clone(),
        macbeth_point_count: record.macbeth_data.len(),
        sample_interpretation: "linear camera RGB, before white balance; normalized near sensor white",
        white_patch_camera_rgb: record.macbeth_data.get(18).and_then(|sample| {
            Some([
                f64::from(sample[0]?),
                f64::from(sample[1]?),
                f64::from(sample[2]?),
            ])
        }),
        color_matrix_white_ratio_log_error: color_matrix_ratio_error(record),
        forward_matrix_d50_white_error: complete_matrix(record.forward_matrix.as_ref()).map(
            |matrix| {
                let white = mul_vec(&matrix, [1.0; 3]);
                (white[0] - D50_WHITE[0])
                    .hypot(white[1] - D50_WHITE[1])
                    .hypot(white[2] - D50_WHITE[2])
            },
        ),
        current_d65_only: None,
        factory_forward: None,
        fitted_matrix_holdout: None,
        nonlinear_holdout: None,
        fitted_matrix: None,
        factory_neutral_axis_mean_chroma: None,
        neutral_axis_mean_chroma: None,
        selected_profile: "factory_forward_matrix",
        fallback_reason: None,
        illuminant_spd_samples: record.illuminant_spd.len(),
        spectral_channels: record
            .spectral_data
            .as_ref()
            .map_or(0, |value| value.channels.len()),
        spectral_sample_spacing_nm: spacing,
        spectral_channel_maxima: record
            .spectral_data
            .as_ref()
            .map(|spectral| {
                spectral
                    .channels
                    .iter()
                    .map(|channel| channel.values.iter().copied().fold(0.0f32, f32::max))
                    .collect()
            })
            .unwrap_or_default(),
    };

    let Some(camera) = balanced_macbeth(record) else {
        return FactoryProfileReport {
            fallback_reason: Some(
                "requires 24 finite Macbeth samples and positive R/G and B/G ratios".to_owned(),
            ),
            ..base
        };
    };
    let target_lab = COLORCHECKER_LAB_D50;
    let target_xyz = target_lab.map(lab_to_xyz);
    let Some(factory) = complete_matrix(record.forward_matrix.as_ref()) else {
        return FactoryProfileReport {
            fallback_reason: Some("factory ForwardMatrix is incomplete".to_owned()),
            ..base
        };
    };

    let factory_errors = errors_for_matrix(&factory, &camera, &target_lab);
    let current_d65_only = d65_record
        .and_then(|candidate| complete_matrix(candidate.forward_matrix.as_ref()))
        .map(|matrix| metrics(&errors_for_matrix(&matrix, &camera, &target_lab)));
    let fitted_cv = leave_one_out_matrix(&camera, &target_xyz, &target_lab);
    let nonlinear_cv = leave_one_out_quadratic(&camera, &target_xyz, &target_lab);
    let fitted = fit_matrix(&camera, &target_xyz, None);
    let fitted_errors = fitted_cv.as_ref().map(|errors| metrics(errors));
    let nonlinear_errors = nonlinear_cv.as_ref().map(|errors| metrics(errors));
    let factory_metrics = metrics(&factory_errors);
    let fitted_neutral = fitted.map(|matrix| neutral_axis_chroma(&matrix, &camera));

    // A candidate must improve mean and tail held-out errors by a real margin,
    // while keeping the neutral axis at least as stable as the factory profile.
    // Otherwise rendering remains byte-for-byte on the established path.
    let factory_neutral = neutral_axis_chroma(&factory, &camera);
    // The per-module D65 fits reduce target error, but slightly increase the
    // measured spread between physical modules. Keep them as candidates in the
    // report rather than promoting them into rendering until both criteria win.
    let select_fitted = false;

    FactoryProfileReport {
        current_d65_only,
        factory_forward: Some(factory_metrics),
        fitted_matrix_holdout: fitted_errors,
        nonlinear_holdout: nonlinear_errors,
        fitted_matrix: fitted,
        factory_neutral_axis_mean_chroma: Some(factory_neutral),
        neutral_axis_mean_chroma: fitted_neutral,
        selected_profile: if select_fitted {
            "fitted_linear_matrix"
        } else {
            "factory_forward_matrix"
        },
        fallback_reason: (!select_fitted).then_some(
            "candidate was not promoted: held-out accuracy, neutral safety, and inter-camera consistency must all improve"
                .to_owned(),
        ),
        ..base
    }
}

fn color_matrix_ratio_error(record: &FactoryColorRecord) -> Option<f64> {
    let white = match record.illuminant? {
        0 => [1.09850, 1.0, 0.35585], // A
        1 => D50_WHITE,
        2 => [0.95047, 1.0, 1.08883], // D65
        3 => [0.94972, 1.0, 1.22638], // D75
        4 => [0.99186, 1.0, 0.67393], // F2
        5 => [0.95041, 1.0, 1.08747], // F7
        6 => [1.00962, 1.0, 0.64350], // F11
        _ => return None,
    };
    let matrix = complete_matrix(record.color_matrix.as_ref())?;
    let camera = mul_vec(&matrix, white);
    if camera[1] <= 0.0 || record.rg_ratio? <= 0.0 || record.bg_ratio? <= 0.0 {
        return None;
    }
    let predicted = [camera[0] / camera[1], camera[2] / camera[1]];
    Some(
        (predicted[0].ln() - f64::from(record.rg_ratio?).ln())
            .hypot(predicted[1].ln() - f64::from(record.bg_ratio?).ln()),
    )
}

fn balanced_macbeth(record: &FactoryColorRecord) -> Option<[[f64; 3]; 24]> {
    let rg = f64::from(record.rg_ratio?).max(1e-9);
    let bg = f64::from(record.bg_ratio?).max(1e-9);
    let samples: Vec<[f64; 3]> = record
        .macbeth_data
        .iter()
        .map(|sample| {
            Some([
                f64::from(sample[0]?) / rg,
                f64::from(sample[1]?),
                f64::from(sample[2]?) / bg,
            ])
        })
        .collect::<Option<_>>()?;
    samples.try_into().ok()
}

fn complete_matrix(matrix: Option<&[[Option<f32>; 3]; 3]>) -> Option<Mat3> {
    let matrix = matrix?;
    Some([
        [
            f64::from(matrix[0][0]?),
            f64::from(matrix[0][1]?),
            f64::from(matrix[0][2]?),
        ],
        [
            f64::from(matrix[1][0]?),
            f64::from(matrix[1][1]?),
            f64::from(matrix[1][2]?),
        ],
        [
            f64::from(matrix[2][0]?),
            f64::from(matrix[2][1]?),
            f64::from(matrix[2][2]?),
        ],
    ])
}

fn fit_matrix(
    camera: &[[f64; 3]; 24],
    target: &[[f64; 3]; 24],
    omitted: Option<usize>,
) -> Option<Mat3> {
    let mut robust = [1.0; 24];
    let mut result = None;
    for _ in 0..5 {
        let mut normal = [[0.0; 3]; 3];
        let mut cross = [[0.0; 3]; 3];
        for i in 0..24 {
            if omitted == Some(i) {
                continue;
            }
            let neutral_weight = if i == 18 {
                8.0
            } else if i >= 19 {
                3.0
            } else {
                1.0
            };
            let weight = neutral_weight * robust[i];
            for row in 0..3 {
                for column in 0..3 {
                    normal[row][column] += weight * camera[i][row] * camera[i][column];
                    cross[row][column] += weight * target[i][row] * camera[i][column];
                }
            }
        }
        // A camera-neutral input must remain on the D50 neutral axis. These
        // smooth virtual samples prevent a lower chromatic-patch error from
        // being bought with a visible grey tint.
        for level in [0.05, 0.18, 0.5, 1.0] {
            let sample = [level; 3];
            let target = D50_WHITE.map(|white| white * level);
            let weight = 6.0;
            for row in 0..3 {
                for column in 0..3 {
                    normal[row][column] += weight * sample[row] * sample[column];
                    cross[row][column] += weight * target[row] * sample[column];
                }
            }
        }
        let scale = (normal[0][0] + normal[1][1] + normal[2][2]) / 3.0;
        for (axis, row) in normal.iter_mut().enumerate() {
            row[axis] += scale * 1e-8;
        }
        let inv = inverse(&normal)?;
        let matrix = crate::math::mul(&cross, &inv);
        if !matrix.iter().flatten().all(|value| value.is_finite())
            || crate::math::determinant(&matrix).abs() < 1e-7
            || matrix.iter().flatten().any(|value| value.abs() > 8.0)
        {
            return None;
        }
        for i in 0..24 {
            if omitted == Some(i) {
                continue;
            }
            let predicted = xyz_to_lab(mul_vec(&matrix, camera[i]));
            let error = delta_e00(predicted, xyz_to_lab(target[i]));
            robust[i] = (6.0 / error.max(6.0)).clamp(0.2, 1.0);
        }
        result = Some(matrix);
    }
    result
}

fn leave_one_out_matrix(
    camera: &[[f64; 3]; 24],
    target_xyz: &[[f64; 3]; 24],
    target_lab: &[[f64; 3]; 24],
) -> Option<Vec<f64>> {
    (0..24)
        .map(|held_out| {
            let matrix = fit_matrix(camera, target_xyz, Some(held_out))?;
            Some(delta_e00(
                xyz_to_lab(mul_vec(&matrix, camera[held_out])),
                target_lab[held_out],
            ))
        })
        .collect()
}

fn quadratic_features(rgb: [f64; 3]) -> [f64; 9] {
    let [r, g, b] = rgb;
    [r, g, b, r * r, g * g, b * b, r * g, g * b, b * r]
}

fn fit_quadratic(
    camera: &[[f64; 3]; 24],
    target: &[[f64; 3]; 24],
    omitted: usize,
) -> Option<[[f64; 9]; 3]> {
    let mut normal = [[0.0; 9]; 9];
    let mut cross = [[0.0; 9]; 3];
    for i in 0..24 {
        if i == omitted {
            continue;
        }
        let x = quadratic_features(camera[i]);
        let weight = if i == 18 {
            8.0
        } else if i >= 19 {
            3.0
        } else {
            1.0
        };
        for row in 0..9 {
            for column in 0..9 {
                normal[row][column] += weight * x[row] * x[column];
            }
            for channel in 0..3 {
                cross[channel][row] += weight * target[i][channel] * x[row];
            }
        }
    }
    let scale = (normal[0][0] + normal[1][1] + normal[2][2]) / 3.0;
    for (axis, row) in normal.iter_mut().enumerate() {
        // Preserve the matrix portion; strongly regularize the six nonlinear
        // terms because 23 training patches cannot justify a flexible LUT.
        row[axis] += scale * if axis < 3 { 1e-8 } else { 0.025 };
    }
    let inverse = invert_square(normal)?;
    let mut coefficients = [[0.0; 9]; 3];
    for channel in 0..3 {
        for column in 0..9 {
            coefficients[channel][column] =
                (0..9).map(|k| cross[channel][k] * inverse[k][column]).sum();
        }
    }
    Some(coefficients)
}

fn leave_one_out_quadratic(
    camera: &[[f64; 3]; 24],
    target_xyz: &[[f64; 3]; 24],
    target_lab: &[[f64; 3]; 24],
) -> Option<Vec<f64>> {
    (0..24)
        .map(|held_out| {
            let coefficients = fit_quadratic(camera, target_xyz, held_out)?;
            let features = quadratic_features(camera[held_out]);
            let xyz = coefficients.map(|row| {
                row.into_iter()
                    .zip(features)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum()
            });
            Some(delta_e00(xyz_to_lab(xyz), target_lab[held_out]))
        })
        .collect()
}

fn invert_square<const N: usize>(matrix: [[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut left = matrix;
    let mut right = [[0.0; N]; N];
    for (i, row) in right.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for column in 0..N {
        let pivot =
            (column..N).max_by(|&a, &b| left[a][column].abs().total_cmp(&left[b][column].abs()))?;
        if left[pivot][column].abs() < 1e-14 {
            return None;
        }
        left.swap(column, pivot);
        right.swap(column, pivot);
        let divisor = left[column][column];
        for j in 0..N {
            left[column][j] /= divisor;
            right[column][j] /= divisor;
        }
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = left[row][column];
            for j in 0..N {
                left[row][j] -= factor * left[column][j];
                right[row][j] -= factor * right[column][j];
            }
        }
    }
    Some(right)
}

fn errors_for_matrix(
    matrix: &Mat3,
    camera: &[[f64; 3]; 24],
    target_lab: &[[f64; 3]; 24],
) -> Vec<f64> {
    camera
        .iter()
        .zip(target_lab)
        .map(|(&sample, &target)| delta_e00(xyz_to_lab(mul_vec(matrix, sample)), target))
        .collect()
}

fn metrics(errors: &[f64]) -> ErrorMetrics {
    let mut sorted = errors.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p90 = ((sorted.len() as f64 * 0.9).ceil() as usize).saturating_sub(1);
    ErrorMetrics {
        mean_delta_e00: sorted.iter().sum::<f64>() / sorted.len() as f64,
        median_delta_e00: (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0,
        p90_delta_e00: sorted[p90],
        maximum_delta_e00: *sorted.last().unwrap(),
    }
}

fn neutral_axis_chroma(matrix: &Mat3, camera: &[[f64; 3]; 24]) -> f64 {
    camera[18..]
        .iter()
        .map(|&sample| {
            let lab = xyz_to_lab(mul_vec(matrix, sample));
            lab[1].hypot(lab[2])
        })
        .sum::<f64>()
        / 6.0
}

fn lab_to_xyz(lab: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    let inverse = |value: f64| {
        let cube = value.powi(3);
        if cube > 216.0 / 24389.0 {
            cube
        } else {
            (116.0 * value - 16.0) / (24389.0 / 27.0)
        }
    };
    [
        D50_WHITE[0] * inverse(fx),
        D50_WHITE[1] * inverse(fy),
        D50_WHITE[2] * inverse(fz),
    ]
}

fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let transform = |value: f64| {
        if value > 216.0 / 24389.0 {
            value.cbrt()
        } else {
            (24389.0 / 27.0 * value + 16.0) / 116.0
        }
    };
    let fx = transform(xyz[0] / D50_WHITE[0]);
    let fy = transform(xyz[1] / D50_WHITE[1]);
    let fz = transform(xyz[2] / D50_WHITE[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIEDE2000, following Sharma et al. with unit parametric factors.
fn delta_e00(first: [f64; 3], second: [f64; 3]) -> f64 {
    let [l1, a1, b1] = first;
    let [l2, a2, b2] = second;
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let mean_c = (c1 + c2) / 2.0;
    let mean_c7 = mean_c.powi(7);
    let g = 0.5 * (1.0 - (mean_c7 / (mean_c7 + 25.0f64.powi(7))).sqrt());
    let ap1 = (1.0 + g) * a1;
    let ap2 = (1.0 + g) * a2;
    let cp1 = ap1.hypot(b1);
    let cp2 = ap2.hypot(b2);
    let hp = |a: f64, b: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            b.atan2(a).to_degrees().rem_euclid(360.0)
        }
    };
    let hp1 = hp(ap1, b1);
    let hp2 = hp(ap2, b2);
    let dl = l2 - l1;
    let dc = cp2 - cp1;
    let dh_angle = if cp1 * cp2 == 0.0 {
        0.0
    } else if (hp2 - hp1).abs() <= 180.0 {
        hp2 - hp1
    } else if hp2 <= hp1 {
        hp2 - hp1 + 360.0
    } else {
        hp2 - hp1 - 360.0
    };
    let dh = 2.0 * (cp1 * cp2).sqrt() * (dh_angle.to_radians() / 2.0).sin();
    let mean_l = (l1 + l2) / 2.0;
    let mean_cp = (cp1 + cp2) / 2.0;
    let mean_hp = if cp1 * cp2 == 0.0 {
        hp1 + hp2
    } else if (hp1 - hp2).abs() <= 180.0 {
        (hp1 + hp2) / 2.0
    } else if hp1 + hp2 < 360.0 {
        (hp1 + hp2 + 360.0) / 2.0
    } else {
        (hp1 + hp2 - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (mean_hp - 30.0).to_radians().cos()
        + 0.24 * (2.0 * mean_hp).to_radians().cos()
        + 0.32 * (3.0 * mean_hp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * mean_hp - 63.0).to_radians().cos();
    let sl = 1.0 + 0.015 * (mean_l - 50.0).powi(2) / (20.0 + (mean_l - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * mean_cp;
    let sh = 1.0 + 0.015 * mean_cp * t;
    let delta_theta = 30.0 * (-((mean_hp - 275.0) / 25.0).powi(2)).exp();
    let mean_cp7 = mean_cp.powi(7);
    let rc = 2.0 * (mean_cp7 / (mean_cp7 + 25.0f64.powi(7))).sqrt();
    let rt = -rc * (2.0 * delta_theta).to_radians().sin();
    let dl = dl / sl;
    let dc = dc / sc;
    let dh = dh / sh;
    (dl * dl + dc * dc + dh * dh + rt * dc * dh).max(0.0).sqrt()
}

fn append_header_records(header: &LightHeader, records: &mut Vec<FactoryColorRecord>) {
    for module in &header.module_calibration {
        let Some(id) = module.camera_id else { continue };
        let camera = camera_name(id.value());
        records.extend(
            module
                .color
                .iter()
                .map(|color| convert_record(camera.clone(), "module", color)),
        );
    }
    for gold in &header.gold_cc {
        let Some(id) = gold.camera_id else { continue };
        let camera = camera_name(id.value());
        records.extend(
            gold.data
                .iter()
                .map(|color| convert_record(camera.clone(), "gold", color)),
        );
    }
}

fn convert_record(
    camera: String,
    source: &'static str,
    color: &ColorCalibration,
) -> FactoryColorRecord {
    let illuminant = color.type_.map(|value| value.value());
    FactoryColorRecord {
        camera,
        source,
        illuminant,
        illuminant_name: illuminant_name(illuminant).to_owned(),
        forward_matrix: color.forward_matrix.as_ref().map(matrix),
        color_matrix: color.color_matrix.as_ref().map(matrix),
        rg_ratio: color.rg_ratio,
        bg_ratio: color.bg_ratio,
        macbeth_data: color
            .macbeth_data
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect(),
        illuminant_spd: color
            .illuminant_spd
            .iter()
            .map(|point| [point.x, point.y])
            .collect(),
        spectral_data: color.spectral_data.as_ref().map(|spectral| {
            let format = spectral.format.map(|value| value.value());
            FactorySpectralData {
                format,
                format_name: spectral_format_name(format).to_owned(),
                channels: spectral
                    .channel_data
                    .iter()
                    .map(|channel| FactorySpectralChannel {
                        start: channel.start,
                        end: channel.end,
                        values: channel.data.clone(),
                    })
                    .collect(),
            }
        }),
    }
}

fn matrix(value: &chiaro_proto::matrix3x3f::Matrix3x3F) -> [[Option<f32>; 3]; 3] {
    [
        [value.x00, value.x01, value.x02],
        [value.x10, value.x11, value.x12],
        [value.x20, value.x21, value.x22],
    ]
}

pub fn illuminant_name(value: Option<i32>) -> &'static str {
    match value {
        Some(0) => "A",
        Some(1) => "D50",
        Some(2) => "D65",
        Some(3) => "D75",
        Some(4) => "F2",
        Some(5) => "F7",
        Some(6) => "F11",
        Some(7) => "TL84",
        Some(99) => "UNKNOWN",
        Some(_) => "UNRECOGNIZED",
        None => "MISSING",
    }
}

fn spectral_format_name(value: Option<i32>) -> &'static str {
    match value {
        Some(0) => "MONO",
        Some(1) => "RGB",
        Some(2) => "BAYER_RGGB",
        Some(_) => "UNRECOGNIZED",
        None => "MISSING",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chiaro_proto::{
        color_calibration::{
            ColorCalibration,
            color_calibration::{SpectralData, SpectralSensitivity, spectral_data::ChannelFormat},
        },
        lightheader::{ColorCalibrationGold, FactoryModuleCalibration},
    };

    #[test]
    fn dump_preserves_module_and_gold_records() {
        let mut color = ColorCalibration::new();
        color.type_ =
            Some(chiaro_proto::color_calibration::color_calibration::IlluminantType::D65.into());
        color.macbeth_data.push(chiaro_proto::point3f::Point3F {
            x: Some(1.0),
            y: Some(2.0),
            z: Some(3.0),
            ..Default::default()
        });
        let mut spectral = SpectralData::new();
        spectral.format = Some(ChannelFormat::RGB.into());
        spectral.channel_data.push(SpectralSensitivity {
            start: Some(400),
            end: Some(700),
            data: vec![0.1, 0.2],
            ..Default::default()
        });
        color.spectral_data = Some(spectral).into();
        let mut header = LightHeader::new();
        header.module_calibration.push(FactoryModuleCalibration {
            camera_id: Some(chiaro_proto::camera_id::CameraID::A1.into()),
            color: vec![color.clone()],
            ..Default::default()
        });
        header.gold_cc.push(ColorCalibrationGold {
            camera_id: Some(chiaro_proto::camera_id::CameraID::A1.into()),
            data: vec![color],
            ..Default::default()
        });
        let messages = LriMessages {
            headers: vec![header],
            view_preferences: Vec::new(),
        };

        let dump = FactoryColorDump::from_messages(&messages);
        assert_eq!(dump.records.len(), 2);
        assert_eq!(dump.records[0].source, "module");
        assert_eq!(dump.records[1].source, "gold");
        assert_eq!(dump.records[0].illuminant_name, "D65");
        assert_eq!(dump.records[0].macbeth_data.len(), 1);
        assert_eq!(
            dump.records[0]
                .spectral_data
                .as_ref()
                .unwrap()
                .channels
                .len(),
            1
        );
    }

    #[test]
    fn ciede2000_matches_published_reference_pair() {
        let difference = delta_e00([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485]);
        assert!((difference - 2.0425).abs() < 0.0001, "{difference}");
    }

    #[test]
    fn linear_fit_recovers_known_transform() {
        let target = COLORCHECKER_LAB_D50.map(lab_to_xyz);
        let planted = [
            [0.82, 0.11, 0.03422],
            [0.07, 0.91, 0.02],
            [0.01, 0.15, 0.66521],
        ];
        let inverse = crate::math::inverse(&planted).unwrap();
        let camera = target.map(|xyz| mul_vec(&inverse, xyz));
        let fitted = fit_matrix(&camera, &target, None).unwrap();
        for row in 0..3 {
            for column in 0..3 {
                assert!((fitted[row][column] - planted[row][column]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn real_device_profiles_beat_the_d65_only_path_under_warm_light() {
        let messages =
            LriMessages::parse(include_bytes!("../tests/fixtures/calibration.lri")).unwrap();
        let dump = FactoryColorDump::from_messages(&messages);
        assert_eq!(dump.records.len(), 42);
        assert!(
            dump.records
                .iter()
                .all(|record| record.macbeth_data.len() == 24)
        );
        assert!(
            dump.records
                .iter()
                .all(|record| record.illuminant_spd.is_empty())
        );
        assert_eq!(
            dump.records
                .iter()
                .filter(|record| record.spectral_data.is_some())
                .count(),
            14
        );

        let analysis = dump.analyze();
        for illuminant in ["A", "F11"] {
            let consistency = analysis
                .inter_camera_consistency
                .iter()
                .find(|report| report.illuminant == illuminant)
                .unwrap();
            let current = consistency.current_d65_only_mean_pair_delta_e00.unwrap();
            let matched = consistency
                .illuminant_matched_factory_mean_pair_delta_e00
                .unwrap();
            assert!(
                matched < current * 0.6,
                "{illuminant}: {matched} vs {current}"
            );
        }
    }
}
