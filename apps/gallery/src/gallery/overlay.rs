//! Compact embedded geometry used by the interactive Hugin-style preview.

use std::collections::HashMap;

use chiaro_fusion::{
    calibration::{CalibrationDatabase, IntrinsicsMode, LriMessages, module_states},
    geometry::{CameraRefinement, ResolvedCamera},
};

use super::calibration_cache;

/// Resolved capture-time camera models, completed from an identity-matched
/// persistent device calibration when available. Missing modules simply use
/// the UI's focal-group fallback.
#[derive(Clone, Debug)]
pub struct CaptureOverlayGeometry {
    pub cameras: HashMap<String, ResolvedCamera>,
}

impl CaptureOverlayGeometry {
    #[cfg(test)]
    pub fn from_lri(data: &[u8]) -> Result<Self, String> {
        Self::from_lri_and_overlays(data, &[])
    }

    /// Resolve embedded geometry plus persistent calibration from the same
    /// physical camera. A missing/corrupt cache degrades to embedded data.
    pub fn from_lri_with_cache(data: &[u8]) -> Result<Self, String> {
        let capture = LriMessages::parse(data).map_err(|error| error.to_string())?;
        let overlay_bytes = capture
            .device_id()
            .and_then(|device_id| {
                calibration_cache::load_for_device_id(device_id)
                    .ok()
                    .flatten()
            })
            .map(|cached| {
                ["calibration.lri", "zoom_calib_v0.lri"]
                    .iter()
                    .filter_map(|name| cached.files.get(*name))
                    .filter_map(|path| std::fs::read(path).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let overlays = overlay_bytes
            .iter()
            .filter_map(|bytes| LriMessages::parse(bytes).ok())
            .collect::<Vec<_>>();
        Self::from_messages(&capture, &overlays)
    }

    #[cfg(test)]
    fn from_lri_and_overlays(data: &[u8], overlay_bytes: &[Vec<u8>]) -> Result<Self, String> {
        let capture = LriMessages::parse(data).map_err(|error| error.to_string())?;
        let overlays = overlay_bytes
            .iter()
            .map(|bytes| LriMessages::parse(bytes).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_messages(&capture, &overlays)
    }

    fn from_messages(capture: &LriMessages, overlays: &[LriMessages]) -> Result<Self, String> {
        let calibration = CalibrationDatabase::from_capture_and_overlays(capture, overlays);
        let cameras = module_states(capture)
            .into_iter()
            .filter(|state| state.width > 1 && state.height > 1)
            .filter_map(|state| {
                let model = calibration.cameras.get(&state.name)?;
                let camera = ResolvedCamera::new(
                    model,
                    &state,
                    IntrinsicsMode::Clamp,
                    &CameraRefinement::default(),
                )
                .ok()?;
                Some((state.name, camera))
            })
            .collect();
        Ok(Self { cameras })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_embedded_geometry_resolves_usable_modules_without_overlays() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/chiaro-fusion/tests/fixtures/L16_04366_headers.lri");
        let bytes = std::fs::read(path).unwrap();
        let geometry = CaptureOverlayGeometry::from_lri(&bytes).unwrap();
        assert!(geometry.cameras.contains_key("A1"));
        assert!(!geometry.cameras.is_empty());
    }

    #[test]
    fn matching_device_overlays_resolve_every_captured_module() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/chiaro-fusion/tests/fixtures");
        let capture = std::fs::read(directory.join("L16_04366_headers.lri")).unwrap();
        let overlays = ["calibration.lri", "zoom_calib_v0.lri"]
            .iter()
            .map(|name| std::fs::read(directory.join(name)).unwrap())
            .collect::<Vec<_>>();
        let geometry = CaptureOverlayGeometry::from_lri_and_overlays(&capture, &overlays).unwrap();
        let messages = LriMessages::parse(&capture).unwrap();
        assert_eq!(geometry.cameras.len(), module_states(&messages).len());
    }
}
