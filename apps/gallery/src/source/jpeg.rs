use super::*;

pub fn decode_jpeg_preview(data: &[u8], max_edge: usize) -> Result<PreviewImage, String> {
    let mut decoder = zune_jpeg::JpegDecoder::new(data);
    decoder
        .decode_headers()
        .map_err(|error| format!("JPEG header decode failed: {error}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| "JPEG has no image information".to_owned())?;
    let width = usize::from(info.width);
    let height = usize::from(info.height);
    let pixels = decoder
        .decode()
        .map_err(|error| format!("JPEG decode failed: {error}"))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| "JPEG dimensions overflow".to_owned())?;
    let channels = pixels.len().checked_div(pixel_count).unwrap_or(0);
    if !matches!(channels, 1 | 3 | 4) || pixels.len() != pixel_count * channels {
        return Err(format!("Unsupported JPEG output with {channels} channels"));
    }

    let scale = if max_edge == usize::MAX {
        1.0
    } else {
        (max_edge.max(1) as f64 / width.max(height).max(1) as f64).min(1.0)
    };
    let out_width = ((width as f64 * scale).round() as usize).max(1);
    let out_height = ((height as f64 * scale).round() as usize).max(1);
    let mut rgb = Vec::with_capacity(out_width * out_height * 3);
    for y in 0..out_height {
        let source_y = y * height / out_height;
        for x in 0..out_width {
            let source_x = x * width / out_width;
            let offset = (source_y * width + source_x) * channels;
            match channels {
                1 => rgb.extend_from_slice(&[pixels[offset]; 3]),
                3 | 4 => rgb.extend_from_slice(&pixels[offset..offset + 3]),
                _ => unreachable!(),
            }
        }
    }
    Ok(PreviewImage {
        size: [out_width, out_height],
        rgb,
        camera: "JPEG".to_owned(),
        color_calibrated: true,
        metadata: Default::default(),
    })
}

pub(super) fn lri_paths(folder: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(folder)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && has_extension(path, "lri"))
        .collect::<Vec<_>>();
    paths.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    });
    Ok(paths)
}

pub(super) fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

pub(super) fn file_stem_lower(path: &Path) -> Option<String> {
    Some(path.file_stem()?.to_string_lossy().to_lowercase())
}

pub(super) fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usb_product_ids_select_the_expected_mode() {
        assert_eq!(LIGHT_MTP_PRODUCT_ID, 0x0005);
        assert_eq!(LIGHT_PTP_PRODUCT_ID, 0x0007);
        assert_eq!(DeviceMode::Mtp.label(), "MTP");
        assert_eq!(DeviceMode::Ptp.label(), "PTP");
    }

    #[test]
    fn jpeg_orientation_rotates_pixels_and_dimensions() {
        let mut preview = PreviewImage {
            size: [2, 1],
            rgb: vec![255, 0, 0, 0, 255, 0],
            camera: "JPEG".to_owned(),
            color_calibrated: true,
            metadata: CaptureMetadata::default(),
        };
        apply_preview_orientation(&mut preview, 1);
        assert_eq!(preview.size, [1, 2]);
        assert_eq!(preview.rgb, vec![255, 0, 0, 0, 255, 0]);
    }
}
