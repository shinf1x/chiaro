# chiaro-stack

Reusable motion-aware temporal and multi-camera RAW fusion for Light L16 night
captures. It exposes gyro-seeded reference alignment, shared rig motion,
gain-interpolated sensor-noise weighting, CFA-domain robust merging, calibrated
module synthesis, and diagnostics to the CLI and Gallery. Capture-embedded
noise tables take priority, while device-matched calibration overlays fill
missing sensor families or gain points. Monochrome modules use the factory
panchromatic model rather than a Bayer colour channel.

The final mosaics use the shared selectable Simple, AMaZE, RCD, LMMSE, or IGV
reconstruction. LMMSE is the night-stack default; IGV is the other
noise-tolerant stack option, while RCD is intended for individual night-sky
exposures.

Merged Bayer mosaics pass through confidence-tracked local and multiscale RAW
highlight reconstruction before demosaicing. All-module fusion can additionally
use consistent unclipped donor measurements from aligned cameras.
Before demosaicing, it also defaults to capture-adaptive crosstalk: a small
white-balance-aware residual is fitted over the factory 17x13 four-phase mesh
using smooth aligned overlap and accepted only when held-out measurements
improve. Individual cameras fall back to their factory mesh when evidence is
insufficient.

All-module synthesis also supports the shared classical resolution stage. It
locally verifies cross-camera registration at common bandwidth, selects the
finest optical tier, then combines noise-weighted multiscale coefficients
through compact edge-aligned kernels. Equal-resolution sources require distinct
subpixel phases; no learned model or GPU is required.
