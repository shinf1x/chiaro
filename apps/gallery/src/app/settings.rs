//! Persistent application settings and the Settings tab.
//!
//! Settings live in `$XDG_CONFIG_HOME/chiaro/gallery.json` (or the platform
//! equivalent) and are written whenever a value changes.

use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};

use super::*;
use crate::gallery::cache::{DEFAULT_LIMIT_BYTES, TYPICAL_ENTRY_BYTES};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub thumbnail_cache_enabled: bool,
    pub thumbnail_cache_limit_mb: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            thumbnail_cache_enabled: true,
            thumbnail_cache_limit_mb: DEFAULT_LIMIT_BYTES / 1_000_000,
        }
    }
}

/// `$XDG_CONFIG_HOME/chiaro/gallery.json` or the platform equivalent.
pub fn settings_path() -> Option<PathBuf> {
    let base = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("APPDATA")?)
    } else if cfg!(target_os = "macos") {
        std::env::home_dir()?.join("Library/Application Support")
    } else {
        std::env::home_dir()?.join(".config")
    };
    Some(base.join("chiaro").join("gallery.json"))
}

/// Inspectable SQLite catalog beside [`settings_path`].
pub fn database_path() -> Option<PathBuf> {
    settings_path().map(|path| path.with_file_name("gallery.sqlite3"))
}

impl Settings {
    pub fn load() -> Self {
        settings_path()
            .and_then(|path| fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = settings_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(self)?)
    }

    pub fn limit_bytes(&self) -> u64 {
        self.thumbnail_cache_limit_mb.saturating_mul(1_000_000)
    }
}

impl GalleryApp {
    /// Push the settings into the running thumbnail cache.
    pub(super) fn apply_settings(&self) {
        if let Some(cache) = self.loader.thumbnail_cache() {
            cache.set_enabled(self.settings.thumbnail_cache_enabled);
            cache.set_limit_bytes(self.settings.limit_bytes());
        }
    }

    pub(super) fn settings_view(&mut self, ui: &mut egui::Ui) {
        let before = self.settings.clone();
        ui.heading("Settings");
        ui.add_space(8.0);
        ui.label(RichText::new("Thumbnail cache").strong());
        match self.loader.thumbnail_cache().cloned() {
            None => {
                ui.label(
                    RichText::new("No cache directory could be determined on this system.")
                        .color(Color32::from_gray(150)),
                );
            }
            Some(cache) => {
                ui.label(
                    RichText::new(format!("Location: {}", cache.root().display()))
                        .color(Color32::from_gray(150))
                        .size(12.0),
                );
                if let Some(database) = self.loader.database() {
                    ui.label(
                        RichText::new(format!("Catalog: {}", database.path().display()))
                            .color(Color32::from_gray(150))
                            .size(12.0),
                    );
                }
                ui.checkbox(
                    &mut self.settings.thumbnail_cache_enabled,
                    "Keep decoded thumbnails on disk so cards load instantly next time",
                );
                ui.horizontal(|ui| {
                    ui.label("Size limit");
                    ui.add(
                        egui::DragValue::new(&mut self.settings.thumbnail_cache_limit_mb)
                            .range(50..=100_000)
                            .speed(10)
                            .suffix(" MB"),
                    );
                    let expected = self.settings.limit_bytes() / TYPICAL_ENTRY_BYTES;
                    ui.label(
                        RichText::new(format!("room for roughly {expected} captures"))
                            .color(Color32::from_gray(150)),
                    );
                });
                let usage = cache.usage();
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "In use: {} in {} thumbnails",
                        crate::export::format_size(usage.bytes),
                        usage.entries
                    ));
                    let fraction = (usage.bytes as f32 / self.settings.limit_bytes().max(1) as f32)
                        .clamp(0.0, 1.0);
                    ui.add(egui::ProgressBar::new(fraction).desired_width(200.0));
                    if ui.button("Clear cache").clicked()
                        && let Err(error) = cache.clear()
                    {
                        self.current_view.status = Some(format!("Could not clear cache: {error}"));
                    }
                });
                ui.label(
                    RichText::new(
                        "Oldest unused thumbnails are removed first when the limit is reached.",
                    )
                    .color(Color32::from_gray(150))
                    .size(12.0),
                );
            }
        }
        if self.settings != before {
            self.apply_settings();
            if let Err(error) = self.settings.save() {
                self.current_view.status = Some(format!("Could not save settings: {error}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_default_limit() {
        let settings = Settings::default();
        assert_eq!(settings.limit_bytes(), 500 * 1_000_000);
        let json = serde_json::to_string(&settings).unwrap();
        assert_eq!(serde_json::from_str::<Settings>(&json).unwrap(), settings);
        assert!(
            settings_path().is_none_or(|p| p.ends_with(Path::new("chiaro").join("gallery.json")))
        );
        assert!(
            database_path()
                .is_none_or(|p| { p.ends_with(Path::new("chiaro").join("gallery.sqlite3")) })
        );
    }
}
