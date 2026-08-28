//! Reusable, UI-independent hot-pixel, corner-glow, and defect correction for
//! Light L16 RAW frames.
//!
//! The crate is the processing core behind the `chiaro-hotpixel` command-line
//! tool and is designed to be embedded in other Chiaro applications. It builds
//! on the shared [`chiaro::lri`] parser for capture layout and metadata; no
//! second LRI implementation lives here.
//!
//! Typical use:
//!
//! ```no_run
//! use chiaro::lri::parse_raw_layout;
//! use chiaro_hotpixel_core::hotpixel::HotpixelRec;
//! use chiaro_hotpixel_core::pipeline::{FramePipeline, OutputMode};
//! use chiaro_hotpixel_core::scan::mmap_file;
//! use chiaro_hotpixel_core::thermal::ThermalProfile;
//! use chiaro_hotpixel_core::universal_hotpixel::UniversalHotpixelProfile;
//!
//! # fn main() -> anyhow::Result<()> {
//! let factory = HotpixelRec::open("/camera/hotpixel.rec")?;
//! let universal = UniversalHotpixelProfile::bundled()?;
//! let glow = ThermalProfile::bundled()?;
//! let pipeline = FramePipeline {
//!     universal_hotpixel: Some(&universal),
//!     thermal: Some(&glow),
//!     ..FramePipeline::default()
//! };
//!
//! let lri = mmap_file("/captures/L16_04480.lri".as_ref())?;
//! let layout = parse_raw_layout(&lri, &Default::default())?;
//! for camera in &layout.cameras {
//!     let map = factory.load_rotated_map(camera.id, camera.width, camera.height)?;
//!     let frame = pipeline.correct_lri(&lri, camera, &map)?;
//!     frame.write_png(format!("/frames/{}.png", camera.name).as_ref(), OutputMode::Rgb)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Module overview:
//!
//! - [`hotpixel`]: the factory `hotpixel.rec` container and per-camera maps;
//! - [`raw10`]: Light's reversed packed-RAW10 decoding;
//! - [`correct`]: factory-guided same-colour interpolation and demosaicing;
//! - [`universal_hotpixel`]: the bundled coordinate-free activity prior;
//! - [`thermal`]: the bundled sensor-family corner-glow model;
//! - [`cleanup`]: optional camera-specific defect/line profiles and their trainer;
//! - [`parallel`]: row-band threading used inside every per-frame kernel;
//! - [`simd`]: runtime AVX2 dispatch for the auto-vectorised kernels;
//! - [`pipeline`]: the validated stage order packaged for applications;
//! - [`png16`]: atomic linear 16-bit PNG writers;
//! - [`scan`]: `.lri` discovery, memory mapping, and pattern overrides.

pub mod cleanup;
pub mod correct;
pub mod hotpixel;
pub mod parallel;
pub mod pipeline;
pub mod png16;
pub mod raw10;
pub mod scan;
pub mod simd;
pub mod thermal;
pub mod universal_hotpixel;

pub use chiaro::lri;
pub use pipeline::{CleanupStage, CorrectedFrame, FramePipeline, OutputMode, extract_raw_plane};
