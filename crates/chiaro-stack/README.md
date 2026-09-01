# chiaro-stack

Reusable motion-aware temporal and multi-camera RAW fusion for Light L16 night
captures. It exposes gyro-seeded reference alignment, shared rig motion,
embedded sensor-noise weighting, CFA-domain robust merging, calibrated module
synthesis, and diagnostics to the CLI and future Gallery integrations.

The final mosaics use the shared selectable Simple, AMaZE, RCD, LMMSE, or IGV
reconstruction. LMMSE is the night-stack default; IGV is the other
noise-tolerant stack option, while RCD is intended for individual night-sky
exposures.
