//! Multi-camera alignment and high-resolution frame synthesis for Light L16
//! captures: the processing pipeline behind the gallery's fused export.
//!
//! Stages (see [`pipeline`]):
//!
//! 1. hot-pixel removal per module (`chiaro-hotpixel-core`);
//! 2. alignment: the factory camera model ([`calibration`], [`geometry`])
//!    predicts where every module lands in the reference frame, then a
//!    coarse-to-fine correlation search refines a global homography and a
//!    dense calibrated inverse-depth reconstruction corrects parallax
//!    ([`align`], [`depth`]);
//! 3. synthesis: every module is resampled onto a common high-resolution
//!    canvas with resolution-aware, feathered weights and written as a linear
//!    or display-referred 16-bit PNG ([`synth`]).
//!
//! Ambiguous depth and occlusion estimates reduce the affected module's local
//! synthesis confidence, allowing the reference image to remain authoritative
//! instead of averaging incompatible surfaces.

pub mod align;
pub mod calibration;
pub mod depth;
pub mod geometry;
pub mod image;
pub mod math;
pub mod pipeline;
pub mod synth;
