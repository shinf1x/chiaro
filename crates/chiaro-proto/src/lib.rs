//! Typed Light L16 protobuf bindings generated from the recovered schemas.
//!
//! Applications should normally use the higher-level `chiaro` crate.
//! This crate is the shared wire-schema boundary for parsers and future tools.

mod proto;

pub use proto::*;
pub use protobuf::Message;
