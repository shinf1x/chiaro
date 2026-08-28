//! Compact embedded geometry used by the interactive Hugin-style preview.

use std::collections::HashMap;

use chiaro_fusion::{
    calibration::{CalibrationDatabase, IntrinsicsMode, LriMessages, module_states},
    geometry::{CameraRefinement, ResolvedCamera},
};

/// Resolved capture-time camera models. Missing modules simply use the UI's
/// focal-group fallback; embedded calibration is intentionally best-effort.
#[derive(Clone, Debug)]
pub struct CaptureOverlayGeometry {
    pub cameras: HashMap<String, ResolvedCamera>,
}

impl CaptureOverlayGeometry {
    pub fn from_lri(data: &[u8]) -> Result<Self, String> {
        let messages = LriMessages::parse(data).map_err(|error| error.to_string())?;
        let calibration = CalibrationDatabase::from_capture_and_overlays(&messages, &[]);
        let cameras = module_states(&messages)
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
}
