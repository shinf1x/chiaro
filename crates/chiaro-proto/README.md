# Chiaro Proto

Typed Rust bindings for the recovered Light L16 protobuf schemas.

The original schemas were recovered from Lumen by
[`@dllu`](https://github.com/dllu) for
[`dllu/lri-rs`](https://github.com/dllu/lri-rs). They are shared here so
Chiaro Gallery, Chiaro Hotpixel, and future applications do not define Light
metadata independently. Generated Rust files are committed, so normal workspace
builds do not require `protoc` or a protobuf code generator.

Real L16 captures sometimes omit fields declared `required` by the recovered
proto2 schemas. The active schemas therefore use optional fields for tolerant,
read-only decoding.

Most applications should depend on the higher-level `chiaro` crate,
which validates container ranges and converts these wire messages into stable
capture and processing types.
