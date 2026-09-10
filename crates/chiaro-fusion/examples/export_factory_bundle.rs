use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chiaro::lri::{SensorPattern, parse_raw_layout};
use chiaro_fusion::{
    calibration::{
        CalibrationDatabase, CameraCalibration, IntrinsicsMode, LriMessages, ModuleState,
        module_states,
    },
    geometry::{CameraRefinement, ResolvedCamera},
    image::Mosaic,
};
use chiaro_hotpixel_core::pipeline::extract_raw_plane_threaded;
use chiaro_hotpixel_core::{hotpixel::HotpixelRec, pipeline::FramePipeline};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Serialize)]
struct ExportManifest {
    schema: &'static str,
    source_lri: String,
    device_id: Option<String>,
    image_coordinates: &'static str,
    image_correction: &'static str,
    pixel_coordinates: &'static str,
    distance_units: &'static str,
    intrinsics_resolution: &'static str,
    modules: Vec<Value>,
}

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let lri_path = arguments
        .next()
        .map(PathBuf::from)
        .context("usage: export_factory_bundle CAPTURE.lri OUTPUT [OVERLAY.lri ...]")?;
    let output = arguments
        .next()
        .map(PathBuf::from)
        .context("usage: export_factory_bundle CAPTURE.lri OUTPUT [OVERLAY.lri ...]")?;
    let sources = arguments.map(PathBuf::from).collect::<Vec<_>>();
    let hotpixel_path = sources.iter().find(|path| {
        path.file_name()
            .is_some_and(|name| name.eq_ignore_ascii_case("hotpixel.rec"))
    });
    let overlay_paths = sources
        .iter()
        .filter(|path| Some(*path) != hotpixel_path)
        .cloned()
        .collect::<Vec<_>>();
    if output.exists() {
        bail!("output already exists: {}", output.display());
    }

    let lri = fs::read(&lri_path).with_context(|| format!("read {}", lri_path.display()))?;
    let messages = LriMessages::parse(&lri)?;
    let overlays = overlay_paths
        .iter()
        .map(|path| {
            LriMessages::parse(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        })
        .collect::<Result<Vec<_>>>()?;
    let calibration = CalibrationDatabase::from_capture_and_overlays(&messages, &overlays);
    let hotpixel = hotpixel_path.map(HotpixelRec::open).transpose()?;
    let states = module_states(&messages)
        .into_iter()
        .map(|state| (state.name.clone(), state))
        .collect::<HashMap<_, _>>();
    let layout = parse_raw_layout(&lri, &HashMap::new()).map_err(anyhow::Error::msg)?;

    fs::create_dir_all(output.join("images"))?;
    fs::create_dir_all(output.join("factory-source"))?;
    let mut modules = Vec::new();
    for raw in &layout.cameras {
        let Some(state) = states.get(&raw.name) else {
            continue;
        };
        let Some(factory) = calibration.cameras.get(&raw.name) else {
            continue;
        };
        let resolved = ResolvedCamera::new(
            factory,
            state,
            IntrinsicsMode::LinearHall,
            &CameraRefinement::default(),
        )?;

        let samples = if let Some(hotpixel) = &hotpixel {
            let severity = hotpixel.load_rotated_map(raw.id, raw.width, raw.height)?;
            FramePipeline {
                threads: 0,
                ..Default::default()
            }
            .correct_lri(&lri, raw, &severity)
            .with_context(|| format!("extract/correct {}", raw.name))?
            .samples_q6
        } else {
            extract_raw_plane_threaded(&lri, raw, 0)
                .with_context(|| format!("extract {}", raw.name))?
                .into_iter()
                .map(|sample| sample << 6)
                .collect()
        };
        let mosaic = Mosaic::from_stream_q6(
            samples,
            raw.width,
            raw.height,
            raw.pattern,
            raw.black_level,
            raw.white_level,
        );
        let image_name = format!("{}.pgm", raw.name);
        write_linear_luminance(&output.join("images").join(&image_name), &mosaic)?;

        let ray_pixels = [
            [0.0, 0.0],
            [(raw.width - 1) as f64, 0.0],
            [0.0, (raw.height - 1) as f64],
            [(raw.width - 1) as f64, (raw.height - 1) as f64],
            [
                (raw.width as f64 - 1.0) * 0.5,
                (raw.height as f64 - 1.0) * 0.5,
            ],
        ];
        let validation_rays = ray_pixels
            .into_iter()
            .map(|pixel| {
                let ray = resolved.pixel_to_ray(pixel);
                json!({"pixel": pixel, "origin_world": ray.origin, "direction_world": ray.direction})
            })
            .collect::<Vec<_>>();
        modules.push(module_json(
            raw.id,
            raw.pattern,
            raw.black_level,
            raw.white_level,
            state,
            factory,
            &resolved,
            image_name,
            validation_rays,
        )?);
    }

    for path in &sources {
        let name = path
            .file_name()
            .context("calibration overlay has no file name")?;
        fs::copy(path, output.join("factory-source").join(name))?;
    }
    let manifest = ExportManifest {
        schema: "chiaro-factory-rig-bundle-v1",
        source_lri: lri_path.display().to_string(),
        device_id: messages.device_id().map(|id| id.to_string()),
        image_coordinates: "Every image is in Light calibration-raster orientation; no EXIF/display rotation. Origin is the centre of the top-left pixel, +x right, +y down.",
        image_correction: if hotpixel.is_some() {
            "Factory hot/dead-pixel correction applied in adaptive mode. No glow, generic temperature prior, learned cleanup, vignetting or geometric warp was applied."
        } else {
            "No hot/dead-pixel correction was applied."
        },
        pixel_coordinates: "Floating coordinates address pixel centres: integer (x,y) is the centre of sensor sample [x,y]. All K matrices and distortion centres use these native per-camera pixels.",
        distance_units: "Factory world coordinates and ray depths are believed to be millimetres.",
        intrinsics_resolution: "resolved_factory.k uses linear Hall interpolation/extrapolation at the capture lens_hall, matching chiaro-fuse's default.",
        modules,
    };
    fs::write(
        output.join("factory-calibration.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(output.join("README.md"), README)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn module_json(
    id: usize,
    pattern: SensorPattern,
    black_level: f32,
    white_level: f32,
    state: &ModuleState,
    factory: &CameraCalibration,
    resolved: &ResolvedCamera,
    image_name: String,
    validation_rays: Vec<Value>,
) -> Result<Value> {
    let canonical = factory.canonical_pose.as_ref().map(|pose| {
        json!({
            "rotation_world_to_camera": pose.rotation_wc,
            "translation_world_to_camera": pose.translation_wc,
            "center_world": pose.center_world(),
        })
    });
    let mirror = factory.mirror.as_ref().map(|mirror| {
        json!({
            "real_camera_location_world": mirror.real_camera_location,
            "real_camera_orientation_camera_to_world": mirror.real_camera_orientation_cw,
            "rotation_axis_world": mirror.rotation_axis,
            "point_on_rotation_axis_world": mirror.point_on_rotation_axis,
            "mirror_plane_distance": mirror.mirror_plane_distance,
            "mirror_normal_zero_world": mirror.mirror_normal_zero,
            "flip_image_around_x": mirror.flip_img_around_x,
            "capture_mirror_angle_degrees": mirror.actuator.angle_for_hall(state.mirror_hall).ok(),
            "actuator": {
                "mean_std_normalize": mirror.actuator.mean_std_normalize,
                "actuator_length_offset": mirror.actuator.actuator_length_offset,
                "actuator_length_scale": mirror.actuator.actuator_length_scale,
                "mirror_angle_offset": mirror.actuator.mirror_angle_offset,
                "mirror_angle_scale": mirror.actuator.mirror_angle_scale,
                "hall_angle_pairs": mirror.actuator.hall_angle_pairs,
                "quadratic_coefficients": mirror.actuator.quadratic_coeffs,
                "selected_quadratic_branch": mirror.actuator.selected_branch(),
            }
        })
    });
    let distortion = factory.distortion.as_ref().map(|distortion| {
        json!({
            "model": "Brown k1,k2,p1,p2,k3 in the explicit normalized frame",
            "center_px": distortion.center,
            "normalization_px": distortion.normalization,
            "coefficients": distortion.coeffs,
        })
    });
    Ok(json!({
        "name": state.name,
        "camera_id": id,
        "image": format!("images/{image_name}"),
        "jpeg": format!("images/{}.jpg", state.name),
        "width": state.width,
        "height": state.height,
        "sensor_pattern": pattern.as_str(),
        "black_level": black_level,
        "white_level": white_level,
        "capture_state": {
            "lens_hall": state.lens_hall,
            "mirror_hall": state.mirror_hall,
            "gain": state.gain,
            "exposure_ns": state.exposure_ns,
            "focus": {
                "achieved": state.focus.achieved,
                "disparity_distance": state.focus.disparity_distance,
                "contrast_distance": state.focus.contrast_distance,
                "roi": state.focus.roi,
                "lens_timeout": state.focus.lens_timeout,
                "mirror_timeout": state.focus.mirror_timeout,
            }
        },
        "factory": {
            "mirror_type": factory.mirror_type.map(|value| format!("{value:?}")),
            "intrinsics_bundles": factory.intrinsics.iter().map(|bundle| json!({
                "hall_code": bundle.hall_code,
                "focus_distance": bundle.focus_distance,
                "k": bundle.k,
            })).collect::<Vec<_>>(),
            "canonical_pose": canonical,
            "mirror_model": mirror,
            "distortion": distortion,
        },
        "resolved_factory": {
            "k": resolved.k,
            "center_world": resolved.center(),
            "focal_px": resolved.focal_px,
            "focus_distance": resolved.focus_distance,
            "validation_pixel_to_world_rays": validation_rays,
        }
    }))
}

fn write_linear_luminance(path: &Path, mosaic: &Mosaic) -> Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    write!(output, "P5\n{} {}\n65535\n", mosaic.width, mosaic.height)?;
    for y in 0..mosaic.height {
        for x in 0..mosaic.width {
            let rgb = mosaic.sample_rgb(x as f32, y as f32).unwrap_or_default();
            let luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).clamp(0.0, 1.0);
            output.write_all(&((luminance * 65535.0).round() as u16).to_be_bytes())?;
        }
    }
    output.flush()?;
    Ok(())
}

const README: &str = r#"# L16 factory-rig matching bundle

The lossless PGM and convenience JPEG images are in native per-camera
calibration-raster coordinates. Use the PGM files for final subpixel scoring;
JPEG compression is only for quick inspection and feature bootstrapping.

`factory-calibration.json` contains the capture Hall positions, every factory
intrinsics bundle, the resolved capture-time K matrix, canonical or movable-
mirror extrinsics, Brown distortion, and five reference rays per camera.
Those rays are fixtures for checking an independently implemented Python
camera model before optimizing anything.

For a reference pixel `p` and ray depth `d`, the corresponding target pixel is
`target.project(reference.ray(p).origin + d * reference.ray(p).direction)`.
Residuals must always be stated in the native pixel grid of the camera in
which they are measured. Also report reference-equivalent residuals by
dividing target-camera displacement by the local target/reference scale.

The copied `.lri` files are the exact cached factory sources used to supplement
the calibration embedded in the capture. `hotpixel.rec` is supplied for
provenance/re-extraction but is not part of the geometric camera model.
"#;
