use crate::{hex_hamming_distance, bytes_hamming_distance};

#[test]
fn test_basic_hamming() {
    assert_eq!(hex_hamming_distance("deadbeef", "00000000").unwrap(), 24);
    assert_eq!(hex_hamming_distance("ffff", "0000").unwrap(), 16);
    assert_eq!(hex_hamming_distance("0000", "0000").unwrap(), 0);
    assert_eq!(hex_hamming_distance("f", "0").unwrap(), 4);
}

#[test]
fn test_mixed_case() {
    assert_eq!(hex_hamming_distance("DEADBEEF", "deadbeef").unwrap(), 0);
    assert_eq!(hex_hamming_distance("AbCdEf", "abcdef").unwrap(), 0);
    assert_eq!(hex_hamming_distance("aAbBcC", "AABBCC").unwrap(), 0);
}

#[test]
fn test_long_strings_32plus() {
    // 32 chars — exercises the SSE pack/32-char loop
    let a32 = "f".repeat(32);
    let b32 = "0".repeat(32);
    assert_eq!(hex_hamming_distance(&a32, &b32).unwrap(), 128);

    // 64 chars — exercises the AVX2 64-char loop
    let a64 = "f".repeat(64);
    let b64 = "0".repeat(64);
    assert_eq!(hex_hamming_distance(&a64, &b64).unwrap(), 256);

    // 128 chars — multiple AVX2 iterations
    let a128 = "f".repeat(128);
    let b128 = "0".repeat(128);
    assert_eq!(hex_hamming_distance(&a128, &b128).unwrap(), 512);

    // 254 chars — AVX2 loop + SSE tail + scalar tail
    let a254 = "f".repeat(254);
    let b254 = "0".repeat(254);
    assert_eq!(hex_hamming_distance(&a254, &b254).unwrap(), 1016);
}

#[test]
fn test_long_mixed_content() {
    // Mixed hex chars to exercise all parse paths across SIMD lanes
    let a = "0123456789abcdef".repeat(8); // 128 chars
    let b = "fedcba9876543210".repeat(8);
    let result = hex_hamming_distance(&a, &b).unwrap();
    // Each pair: 0^f=f(4), 1^e=f(4), 2^d=f(4), 3^c=f(4),
    //            4^b=f(4), 5^a=f(4), 6^9=f(4), 7^8=f(4),
    //            8^7=f(4), 9^6=f(4), a^5=f(4), b^4=f(4),
    //            c^3=f(4), d^2=f(4), e^1=f(4), f^0=f(4) = 64 per 16 chars
    assert_eq!(result, 64 * 8);

    // Mixed case in long string
    let a_mixed = "AaBbCcDdEeFf0011".repeat(4); // 64 chars
    let b_mixed = "aAbBcCdDeEfF0011".repeat(4);
    assert_eq!(hex_hamming_distance(&a_mixed, &b_mixed).unwrap(), 0);
}

#[test]
fn test_invalid_chars() {
    assert!(hex_hamming_distance("zz", "00").is_err());
    assert!(hex_hamming_distance("gg", "00").is_err());
    assert!(hex_hamming_distance("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "00000000000000000000000000000000ff").is_err());
    assert!(hex_hamming_distance("``````````````````````````````````", "00000000000000000000000000000000ff").is_err());
}

#[test]
fn test_length_mismatch() {
    assert!(hex_hamming_distance("ff", "f").is_err());
}

#[test]
fn test_empty() {
    assert_eq!(hex_hamming_distance("", "").unwrap(), 0);
}

#[test]
fn test_bytes_basic() {
    assert_eq!(bytes_hamming_distance(b"\xff", b"\x00").unwrap(), 8);
    assert_eq!(bytes_hamming_distance(b"\xde\xad\xbe\xef", b"\x00\x00\x00\x00").unwrap(), 24);
}
