#!/usr/bin/env python
"""
Run existing tests against the Rust implementation
"""
import sys
import os
sys.path.insert(0, '/home/runner/work/hexhamming/hexhamming')

from platform import machine
import pytest
import hexhamming_rs as hexhamming

# Now run the existing test function
def test_hamming_distance_string():
    test_cases = [
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
    
    for hex1, hex2, expected in test_cases:
        assert expected == hexhamming.hamming_distance_string(hex1, hex2)
        result = hexhamming.set_algo('classic')
        assert len(result) == 0
        assert expected == hexhamming.hamming_distance_string(hex1, hex2)

def test_hamming_distance_byte():
    test_cases = [
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
    
    algorithm_list = ['extra', 'native', 'classic']
    if machine().lower().startswith('x86'):
        algorithm_list.append('sse41')
    
    for hex1, hex2, expected in test_cases:
        for algorithm in algorithm_list:
            result = hexhamming.set_algo(algorithm)
            if len(result) > 0:
                print(f'Warning: Skipping {algorithm}, reason: {result}')
                continue
            assert expected == hexhamming.hamming_distance_bytes(hex1, hex2)

def test_check_hexstrings_within_dist():
    test_cases = [
        ("000abcdef", "011abcdef", 3, True),
        ("1f0abcdef", "011abcdef", 3, False),
        ("011abcdef", "011abcdef", 1000, True),
    ]
    
    algorithm_list = ['extra', 'native', 'classic']
    if machine().lower().startswith('x86'):
        algorithm_list.append('sse41')
    
    for hex1, hex2, max_dist, expected in test_cases:
        for algorithm in algorithm_list:
            result = hexhamming.set_algo(algorithm)
            if len(result) > 0:
                print(f'Warning: Skipping {algorithm}, reason: {result}')
                continue
            assert expected == hexhamming.check_hexstrings_within_dist(hex1, hex2, max_dist)

if __name__ == "__main__":
    print("Running original test cases against Rust implementation...")
    
    try:
        test_hamming_distance_string()
        print("✓ String hamming distance tests passed")
        
        test_hamming_distance_byte()
        print("✓ Byte hamming distance tests passed")
        
        test_check_hexstrings_within_dist()
        print("✓ Within distance tests passed")
        
        print("\n🎉 All original tests pass with Rust implementation!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)