#!/usr/bin/env python3
"""
Benchmark comparison between C++ and Rust implementations of hexhamming
"""
import time
import random
import string

# Import both implementations
import hexhamming  # C++ version
import hexhamming_rs  # Rust version

def generate_hex_string(length):
    """Generate a random hex string of given length"""
    return ''.join(random.choices('0123456789abcdef', k=length))

def generate_byte_array(length):
    """Generate a random byte array of given length"""
    return bytes([random.randint(0, 255) for _ in range(length)])

def benchmark_function(func, *args, iterations=10000):
    """Benchmark a function call"""
    start_time = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time * 1e9, result  # Return time in nanoseconds

def run_comparison():
    """Run comprehensive performance comparison"""
    print("=== Hexhamming Rust vs C++ Performance Comparison ===\n")
    
    # Test sizes
    test_sizes = [3, 64, 256, 1000, 1024]
    
    print("String Hamming Distance Tests:")
    print("-" * 80)
    print(f"{'Size':<8} {'C++ (ns)':<15} {'Rust (ns)':<15} {'Speedup':<10} {'Validation'}")
    print("-" * 80)
    
    for size in test_sizes:
        # Generate test data
        hex1 = generate_hex_string(size)
        hex2 = generate_hex_string(size)
        
        # Reduce iterations for larger strings to keep test time reasonable
        iterations = 10000 if size <= 256 else 1000
        
        # Benchmark C++ version
        cpp_time, cpp_result = benchmark_function(
            hexhamming.hamming_distance_string, hex1, hex2, iterations=iterations
        )
        
        # Benchmark Rust version  
        rust_time, rust_result = benchmark_function(
            hexhamming_rs.hamming_distance_string, hex1, hex2, iterations=iterations
        )
        
        # Calculate speedup
        speedup = cpp_time / rust_time if rust_time > 0 else float('inf')
        validation = "✓" if cpp_result == rust_result else "✗"
        
        print(f"{size:<8} {cpp_time:<15.1f} {rust_time:<15.1f} {speedup:<10.2f}x {validation}")
    
    print("\nByte Array Hamming Distance Tests:")
    print("-" * 80)
    print(f"{'Size':<8} {'C++ (ns)':<15} {'Rust (ns)':<15} {'Speedup':<10} {'Validation'}")
    print("-" * 80)
    
    for size in test_sizes:
        # Generate test data
        bytes1 = generate_byte_array(size)
        bytes2 = generate_byte_array(size)
        
        iterations = 10000 if size <= 256 else 1000
        
        # Benchmark C++ version
        cpp_time, cpp_result = benchmark_function(
            hexhamming.hamming_distance_bytes, bytes1, bytes2, iterations=iterations
        )
        
        # Benchmark Rust version
        rust_time, rust_result = benchmark_function(
            hexhamming_rs.hamming_distance_bytes, bytes1, bytes2, iterations=iterations
        )
        
        speedup = cpp_time / rust_time if rust_time > 0 else float('inf')
        validation = "✓" if cpp_result == rust_result else "✗"
        
        print(f"{size:<8} {cpp_time:<15.1f} {rust_time:<15.1f} {speedup:<10.2f}x {validation}")
    
    # Test correctness with known values
    print("\nCorrectness Tests:")
    print("-" * 50)
    
    test_cases = [
        ("deadbeef", "00000000", 24),
        ("abc", "abc", 0),
        ("000", "001", 1),
        ("ABCDEF", "000001", 16),
        ("", "", 0),
        ("f" * 64, "0" * 64, 256),
    ]
    
    all_passed = True
    for hex1, hex2, expected in test_cases:
        cpp_result = hexhamming.hamming_distance_string(hex1, hex2)
        rust_result = hexhamming_rs.hamming_distance_string(hex1, hex2)
        
        cpp_correct = cpp_result == expected
        rust_correct = rust_result == expected
        match = cpp_result == rust_result
        
        status = "✓" if cpp_correct and rust_correct and match else "✗"
        if not (cpp_correct and rust_correct and match):
            all_passed = False
            
        print(f"'{hex1}' ^ '{hex2}' = {expected}: C++={cpp_result}, Rust={rust_result} {status}")
    
    print(f"\nOverall correctness: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    
    # Test within distance functions
    print("\nWithin Distance Tests:")
    print("-" * 50)
    
    within_tests = [
        ("000abcdef", "011abcdef", 3, True),
        ("1f0abcdef", "011abcdef", 3, False),
        ("011abcdef", "011abcdef", 1000, True),
    ]
    
    for hex1, hex2, max_dist, expected in within_tests:
        cpp_result = hexhamming.check_hexstrings_within_dist(hex1, hex2, max_dist)
        rust_result = hexhamming_rs.check_hexstrings_within_dist(hex1, hex2, max_dist)
        
        match = cpp_result == rust_result == expected
        status = "✓" if match else "✗"
        
        print(f"'{hex1}' ~ '{hex2}' ≤ {max_dist}: Expected={expected}, C++={cpp_result}, Rust={rust_result} {status}")

if __name__ == "__main__":
    run_comparison()