//! Guards the emulated-CPU CI jobs: when `HEXHAMMING_EXPECT_X86_FEATURES` is set
//! (for example `avx2,popcnt,!avx512f`), every listed feature must be detected and
//! every `!`-prefixed feature must be absent, so a misconfigured emulator cannot
//! silently skip the SIMD backends it is meant to exercise.

#[cfg(target_arch = "x86_64")]
fn detected(feature: &str) -> bool {
    match feature {
        "sse4.1" => std::arch::is_x86_feature_detected!("sse4.1"),
        "popcnt" => std::arch::is_x86_feature_detected!("popcnt"),
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
        "avx512bw" => std::arch::is_x86_feature_detected!("avx512bw"),
        "avx512bitalg" => std::arch::is_x86_feature_detected!("avx512bitalg"),
        "avx512vbmi" => std::arch::is_x86_feature_detected!("avx512vbmi"),
        "avx512vpopcntdq" => std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        other => panic!("unknown feature {other:?} in HEXHAMMING_EXPECT_X86_FEATURES"),
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn expected_x86_features_are_detected() {
    let Ok(spec) = std::env::var("HEXHAMMING_EXPECT_X86_FEATURES") else {
        return;
    };
    for item in spec
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let (expected, feature) = match item.strip_prefix('!') {
            Some(feature) => (false, feature),
            None => (true, item),
        };
        assert_eq!(
            detected(feature),
            expected,
            "feature {feature}: expected detected={expected}"
        );
    }
}
