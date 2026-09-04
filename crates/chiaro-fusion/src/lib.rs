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
//! 3. factory colour selection: the original D65 profile remains the default;
//!    experimental modes use capture CCT or sparse, reliable aligned overlap
//!    to choose an A/F11/D65 blend ([`array_color`]);
//! 4. capture-adaptive crosstalk: a constrained residual over the factory
//!    four-phase mesh is fitted from smooth aligned overlap and accepted only
//!    after held-out validation ([`crosstalk`]);
//! 5. synthesis: every module is resampled onto a common high-resolution
//!    canvas with resolution-aware, feathered weights and written as a linear
//!    or display-referred 16-bit PNG ([`synth`]).
//!
//! Ambiguous depth and occlusion estimates reduce the affected module's local
//! synthesis confidence, allowing the reference image to remain authoritative
//! instead of averaging incompatible surfaces.

pub mod align;
pub mod array_color;
pub mod calibration;
pub mod color_profile;
pub mod crosstalk;
pub mod depth;
pub mod geometry;
pub mod image;
pub mod math;
pub mod pipeline;
pub mod resolution;
pub mod synth;
