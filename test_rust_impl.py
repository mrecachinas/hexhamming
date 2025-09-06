#!/usr/bin/env python3
"""
Test the Rust implementation with the existing test suite to ensure compatibility
"""
import sys
import os

# Add current directory to path to import both modules
sys.path.insert(0, '/home/runner/work/hexhamming/hexhamming')

import pytest
import hexhamming_rs
from platform import machine

# Test data and expected results from the original test suite
test_cases_string = [
    ("abc", "abc", 0),
    ("000", "001", 1),
    ("ABCDEF", "000001", 16),
    ("", "", 0),
    ("f" * 64, "0" * 64, 256),
    ("f" * 64, "f" * 64, 0),
    ("0" * 64, "0" * 64, 0),
    ("f" * 10000, "0" * 10000, 40000),
    ("f" * 10000, "f" * 10000, 0),
]

test_cases_bytes = [
    (b"\xab\x0c", b"\xab\x0c", 0),
    (b"\x00", b"\x01", 1),
    (b"\xAB\xCD\xEF", b"\x00\x00\x01", 16),
    (b"", b"", 0),
    (b"\xff" * 32, b"\x00" * 32, 256),
    (b"\xff" * 32, b"\xff" * 32, 0),
    (b"\x00" * 32, b"\x00" * 32, 0),
    (b"\xff" * 5000, b"\x00" * 5000, 40000),
    (b"\xff" * 5000, b"\xff" * 5000, 0),
]

within_dist_cases = [
    ("000abcdef", "011abcdef", 3, True),
    ("1f0abcdef", "011abcdef", 3, False),
    ("011abcdef", "011abcdef", 1000, True),
]

def test_string_hamming_distance():
    """Test string hamming distance function"""
    print("Testing hamming_distance_string...")
    for hex1, hex2, expected in test_cases_string:
        result = hexhamming_rs.hamming_distance_string(hex1, hex2)
        assert result == expected, f"Failed for {hex1}, {hex2}: expected {expected}, got {result}"
        print(f"  ✓ '{hex1[:10]}...' ^ '{hex2[:10]}...' = {result}")
    print("All string tests passed!")

def test_bytes_hamming_distance():
    """Test bytes hamming distance function"""
    print("\nTesting hamming_distance_bytes...")
    for bytes1, bytes2, expected in test_cases_bytes:
        result = hexhamming_rs.hamming_distance_bytes(bytes1, bytes2)
        assert result == expected, f"Failed for {bytes1}, {bytes2}: expected {expected}, got {result}"
        print(f"  ✓ {len(bytes1)} bytes = {result}")
    print("All bytes tests passed!")

def test_within_distance():
    """Test within distance function"""
    print("\nTesting check_hexstrings_within_dist...")
    for hex1, hex2, max_dist, expected in within_dist_cases:
        result = hexhamming_rs.check_hexstrings_within_dist(hex1, hex2, max_dist)
        assert result == expected, f"Failed for {hex1}, {hex2}, {max_dist}: expected {expected}, got {result}"
        print(f"  ✓ '{hex1}' ~ '{hex2}' ≤ {max_dist} = {result}")
    print("All within distance tests passed!")

def test_error_handling():
    """Test error conditions"""
    print("\nTesting error handling...")
    
    # Different length strings
    try:
        hexhamming_rs.hamming_distance_string("abc", "ab")
        assert False, "Should have raised error for different lengths"
    except ValueError as e:
        assert "same length" in str(e)
        print("  ✓ Different length detection works")
    
    # Invalid hex characters
    try:
        hexhamming_rs.hamming_distance_string("abg", "abc")
        assert False, "Should have raised error for invalid hex"
    except ValueError as e:
        assert "invalid char" in str(e)
        print("  ✓ Invalid hex character detection works")
    
    # Different length bytes
    try:
        hexhamming_rs.hamming_distance_bytes(b"abc", b"ab")
        assert False, "Should have raised error for different lengths"
    except ValueError as e:
        assert "same length" in str(e)
        print("  ✓ Different byte length detection works")
    
    print("All error handling tests passed!")

def test_set_algo():
    """Test algorithm setting function (for compatibility)"""
    print("\nTesting set_algo compatibility...")
    
    # Valid algorithms should return empty string
    for algo in ["classic", "native", "sse41", "extra"]:
        result = hexhamming_rs.set_algo(algo)
        assert result == "", f"Expected empty string for {algo}, got '{result}'"
        print(f"  ✓ Algorithm '{algo}' accepted")
    
    # Invalid algorithm should return error message
    result = hexhamming_rs.set_algo("invalid")
    assert "without this algorithm" in result
    print("  ✓ Invalid algorithm rejection works")
    
    print("Algorithm compatibility tests passed!")

def run_mini_benchmark():
    """Run a mini benchmark to show performance"""
    import time
    
    print("\nMini Performance Test:")
    print("-" * 40)
    
    # Test cases for benchmarking
    test_data = [
        ("Short strings (64 chars)", "f" * 64, "0" * 64),
        ("Medium strings (1000 chars)", "f" * 1000, "0" * 1000),
        ("Large strings (10000 chars)", "f" * 10000, "0" * 10000),
    ]
    
    for name, hex1, hex2 in test_data:
        iterations = 1000 if len(hex1) <= 1000 else 100
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = hexhamming_rs.hamming_distance_string(hex1, hex2)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / iterations * 1e6  # microseconds
        print(f"{name}: {avg_time:.1f} μs/op (result: {result})")
    
    # Test bytes too
    byte_data = [
        ("Byte arrays (1000 bytes)", b"\xff" * 1000, b"\x00" * 1000),
        ("Byte arrays (10000 bytes)", b"\xff" * 10000, b"\x00" * 10000),
    ]
    
    for name, bytes1, bytes2 in byte_data:
        iterations = 1000 if len(bytes1) <= 1000 else 100
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = hexhamming_rs.hamming_distance_bytes(bytes1, bytes2)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / iterations * 1e6  # microseconds
        print(f"{name}: {avg_time:.1f} μs/op (result: {result})")

if __name__ == "__main__":
    print("=== Rust Hexhamming Implementation Test Suite ===\n")
    
    try:
        test_string_hamming_distance()
        test_bytes_hamming_distance()
        test_within_distance()
        test_error_handling()
        test_set_algo()
        run_mini_benchmark()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The Rust implementation is working correctly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)