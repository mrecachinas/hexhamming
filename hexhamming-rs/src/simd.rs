//! SIMD optimized implementations for x86_64

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
static mut CPU_FEATURES: CpuFeatures = CpuFeatures {
    has_sse41: false,
    has_avx2: false,
    initialized: false,
};

#[cfg(target_arch = "x86_64")]
struct CpuFeatures {
    has_sse41: bool,
    has_avx2: bool,
    initialized: bool,
}

#[cfg(target_arch = "x86_64")]
fn init_cpu_features() {
    unsafe {
        if CPU_FEATURES.initialized {
            return;
        }
        
        if is_x86_feature_detected!("sse4.1") {
            CPU_FEATURES.has_sse41 = true;
        }
        
        if is_x86_feature_detected!("avx2") {
            CPU_FEATURES.has_avx2 = true;
        }
        
        CPU_FEATURES.initialized = true;
    }
}

#[cfg(target_arch = "x86_64")]
pub fn has_sse41() -> bool {
    init_cpu_features();
    unsafe { CPU_FEATURES.has_sse41 }
}

#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    init_cpu_features();
    unsafe { CPU_FEATURES.has_avx2 }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_sse41() -> bool { false }

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool { false }

/// Classic popcount for fallback
#[inline]
fn popcount_classic(mut x: u64) -> u64 {
    const M1: u64 = 0x5555555555555555;
    const M2: u64 = 0x3333333333333333;
    const M4: u64 = 0x0F0F0F0F0F0F0F0F;
    const H01: u64 = 0x0101010101010101;
    
    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    (x * H01) >> 56
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn hamming_distance_bytes_sse41_impl(a: &[u8], b: &[u8]) -> u64 {
    let mut result = 0u64;
    let len = a.len();
    let mut i = 0;
    
    // Process in 16-byte chunks
    while i + 16 <= len {
        let a_chunk = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let b_chunk = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        let xor_result = _mm_xor_si128(a_chunk, b_chunk);
        
        // Extract both 64-bit parts and count bits
        let low = _mm_cvtsi128_si64(xor_result) as u64;
        let high = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xor_result, xor_result)) as u64;
        
        result += low.count_ones() as u64;
        result += high.count_ones() as u64;
        
        i += 16;
    }
    
    // Process remaining bytes
    while i < len {
        result += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
pub fn hamming_distance_bytes_sse41(a: &[u8], b: &[u8]) -> u64 {
    if has_sse41() {
        unsafe { hamming_distance_bytes_sse41_impl(a, b) }
    } else {
        // Fallback to classic implementation
        let mut result = 0u64;
        let len = a.len();
        let mut i = 0;
        
        while i + 8 <= len {
            let a_chunk = u64::from_le_bytes([
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            ]);
            let b_chunk = u64::from_le_bytes([
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            ]);
            result += popcount_classic(a_chunk ^ b_chunk);
            i += 8;
        }
        
        while i < len {
            result += popcount_classic((a[i] ^ b[i]) as u64);
            i += 1;
        }
        
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_bytes_avx2_impl(a: &[u8], b: &[u8]) -> u64 {
    let mut result = 0u64;
    let len = a.len();
    let mut i = 0;
    
    // Process in 32-byte chunks with AVX2
    while i + 32 <= len {
        let a_chunk = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b_chunk = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let xor_result = _mm256_xor_si256(a_chunk, b_chunk);
        
        // Use the lookup table approach for popcount
        let lookup1 = _mm256_setr_epi8(
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
        );
        
        let lookup2 = _mm256_setr_epi8(
            4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
            4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
        );
        
        let low_mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(xor_result, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xor_result, 4), low_mask);
        
        let popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
        let popcnt2 = _mm256_shuffle_epi8(lookup2, hi);
        let r = _mm256_sad_epu8(popcnt1, popcnt2);
        
        // Extract all four 64-bit results
        result += _mm256_extract_epi64(r, 0) as u64;
        result += _mm256_extract_epi64(r, 1) as u64;
        result += _mm256_extract_epi64(r, 2) as u64;
        result += _mm256_extract_epi64(r, 3) as u64;
        
        i += 32;
    }
    
    // Process remaining bytes
    while i < len {
        result += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
pub fn hamming_distance_bytes_avx2(a: &[u8], b: &[u8]) -> u64 {
    if has_avx2() {
        unsafe { hamming_distance_bytes_avx2_impl(a, b) }
    } else {
        // Fallback to SSE4.1 or classic
        hamming_distance_bytes_sse41(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn hamming_distance_bytes_sse41(_a: &[u8], _b: &[u8]) -> u64 {
    unimplemented!("SSE4.1 not available on this architecture")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn hamming_distance_bytes_avx2(_a: &[u8], _b: &[u8]) -> u64 {
    unimplemented!("AVX2 not available on this architecture")
}

/// Convert hex characters to nibbles using SIMD (SSE4.1)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
unsafe fn hamming_distance_string_sse41_impl(a: &str, b: &str) -> Result<u64, &'static str> {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let len = a_bytes.len();
    let mut result = 0u64;
    let mut i = 0;
    
    // SIMD lookup table for hex conversion
    const LOOKUP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
    
    // Process 16 characters at a time when possible
    while i + 16 <= len {
        // Load 16 characters
        let _a_chunk = _mm_loadu_si128(a_bytes.as_ptr().add(i) as *const __m128i);
        let _b_chunk = _mm_loadu_si128(b_bytes.as_ptr().add(i) as *const __m128i);
        
        // Convert to nibbles and compute XOR
        let mut local_result = 0u64;
        for j in 0..16 {
            let a_char = a_bytes[i + j];
            let b_char = b_bytes[i + j];
            
            // Convert hex char to nibble
            let a_nibble = match a_char {
                b'0'..=b'9' => a_char - b'0',
                b'A'..=b'F' => a_char - b'A' + 10,
                b'a'..=b'f' => a_char - b'a' + 10,
                _ => return Err("hex string contains invalid char"),
            };
            
            let b_nibble = match b_char {
                b'0'..=b'9' => b_char - b'0',
                b'A'..=b'F' => b_char - b'A' + 10,
                b'a'..=b'f' => b_char - b'a' + 10,
                _ => return Err("hex string contains invalid char"),
            };
            
            local_result += LOOKUP[(a_nibble ^ b_nibble) as usize] as u64;
        }
        
        result += local_result;
        i += 16;
    }
    
    // Process remaining characters
    while i < len {
        let a_char = a_bytes[i];
        let b_char = b_bytes[i];
        
        let a_nibble = match a_char {
            b'0'..=b'9' => a_char - b'0',
            b'A'..=b'F' => a_char - b'A' + 10,
            b'a'..=b'f' => a_char - b'a' + 10,
            _ => return Err("hex string contains invalid char"),
        };
        
        let b_nibble = match b_char {
            b'0'..=b'9' => b_char - b'0',
            b'A'..=b'F' => b_char - b'A' + 10,
            b'a'..=b'f' => b_char - b'a' + 10,
            _ => return Err("hex string contains invalid char"),
        };
        
        result += LOOKUP[(a_nibble ^ b_nibble) as usize] as u64;
        i += 1;
    }
    
    Ok(result)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn hamming_distance_string_sse41(a: &str, b: &str) -> Result<u64, &'static str> {
    if has_sse41() {
        unsafe { hamming_distance_string_sse41_impl(a, b) }
    } else {
        // Fallback - convert to classic
        Err("SSE4.1 not available, use fallback")
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn hamming_distance_string_sse41(_a: &str, _b: &str) -> Result<u64, &'static str> {
    Err("SSE4.1 not available on this architecture")
}