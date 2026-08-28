//! Multi-camera alignment and high-resolution frame synthesis for Light L16
//! captures: the processing pipeline behind the gallery's fused export.
//!
//! Stages (see [`pipeline`]):
//!
//! 1. hot-pixel removal per module (`chiaro-hotpixel-core`);
//! 2. alignment: the factory camera model ([`calibration`], [`geometry`])
//!    predicts where every module lands in the reference frame, then a
//!    coarse-to-fine correlation search refines one homography per module
//!    ([`align`]);
//! 3. synthesis: every module is resampled onto a common high-resolution
//!    canvas with resolution-aware, feathered weights and written as a linear
//!    or display-referred 16-bit PNG ([`synth`]).
//!
//! The alignment model is a per-module homography, which is exact for distant
//! scenes (pure rotation between bearings) and degrades gracefully into
//! ghosting for near objects. Depth-aware warps are a planned extension and
//! the stage boundaries are designed so they can replace the homography
//! without touching hot-pixel removal or synthesis.

pub mod align;
pub mod calibration;
pub mod geometry;
pub mod image;
pub mod math;
pub mod pipeline;
pub mod synth;
