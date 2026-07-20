# Chiaro library

Reusable, UI-independent support for Light L16 `.lri` captures.

The crate owns the shared LELR framing, typed metadata conversion, sparse
reference-camera reads, RAW10/Bayer-JPEG previews, color calibration, and
processing-oriented RAW layout information. Wire messages come from the
workspace's `chiaro-proto` bindings. Applications should use this
crate rather than introducing another LRI parser.

Useful APIs include:

- `lri::inspect_capture` and `lri::inspect_capture_bytes`;
- `lri::decode_reference_preview` and `lri::decode_camera_preview`;
- `lri::decode_camera_frame_preview` for repeated night-mode frames;
- `lri::inspect_lelr_block_header` for sparse transports;
- `lri::parse_raw_layout` for processing applications.

## Example

```bash
cargo run -p chiaro --release --example inspect_lri -- /path/to/capture.lri [camera] [max-edge]
```

The example prints capture metadata and decodes either the reference camera or
an explicitly named camera. Its preview remains in memory; it writes no image.

## Format research attribution

The 32-byte `LELR` framing, packed 10-bit stream interpretation, and recovered
metadata were cross-checked against:

- [`ookami125/lri-cpp`](https://github.com/ookami125/lri-cpp);
- [`gennyble/lri-rs`](https://github.com/gennyble/lri-rs); and
- [`dllu/lri-rs`](https://github.com/dllu/lri-rs), which recovered the original
  Lumen Protocol Buffer definitions.

The recovered schemas are provided by `chiaro-proto`; no external checkout or
runtime schema compiler is required.
