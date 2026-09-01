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
