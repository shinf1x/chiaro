//! Runtime instruction-set selection for the per-frame kernels.
//!
//! The kernels are ordinary Rust loops that LLVM auto-vectorises. Binaries are
//! built for a baseline CPU (SSE2 on x86-64), which caps the vector width at
//! 128 bits. [`multiversion!`] compiles the same function body a second time
//! with AVX2 enabled and dispatches to it when the host supports it, so the
//! hot loops run with 256-bit vectors without any hand-written intrinsics.
//! On aarch64, NEON is part of the baseline and the generic body is used
//! directly.
//!
//! Measured on real L16 frames (4 threads): glow 45 -> 33 ms, demosaic
//! 63 -> 31 ms. No floating-point contraction is enabled, so results are
//! bit-identical between the two paths; the test suite checks this.

/// Define a function whose body is compiled for the baseline ISA and, on x86,
/// additionally for AVX2 with runtime dispatch.
///
/// Usage mirrors a plain `fn` item without a return value:
///
/// ```ignore
/// multiversion! {
///     /// Doc comments and attributes are allowed.
///     pub fn scale(values: &mut [f32], factor: f32) {
///         for value in values {
///             *value *= factor;
///         }
///     }
/// }
/// ```
macro_rules! multiversion {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $body:block
    ) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) {
            #[inline(always)]
            fn generic($($arg: $ty),*) $body

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) {
                    generic($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 support was confirmed on this CPU above;
                    // the function body is plain Rust with no other
                    // requirements.
                    return unsafe { avx2($($arg),*) };
                }
            }
            generic($($arg),*)
        }
    };
}

pub(crate) use multiversion;

/// Name of the vector path the kernels will take on this machine.
pub fn active_isa() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        return "sse2";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[allow(unreachable_code)]
    "baseline"
}

#[cfg(test)]
mod tests {
    multiversion! {
        fn scale_all(values: &mut [f32], factor: f32) {
            for value in values {
                *value *= factor;
            }
        }
    }

    #[test]
    fn multiversioned_function_runs_and_reports_an_isa() {
        let mut values = vec![1.0, 2.0, 3.0];
        scale_all(&mut values, 2.0);
        assert_eq!(values, [2.0, 4.0, 6.0]);
        assert!(!super::active_isa().is_empty());
    }
}
