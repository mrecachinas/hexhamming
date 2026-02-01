# Re-export all functions from the compiled Rust extension
from hexhamming.hexhamming import (
    hamming_distance_string,
    hamming_distance_bytes,
    check_hexstrings_within_dist,
    check_bytes_arrays_within_dist,
    set_algo,
    __version__,
)

__all__ = [
    "hamming_distance_string",
    "hamming_distance_bytes",
    "check_hexstrings_within_dist",
    "check_bytes_arrays_within_dist",
    "set_algo",
    "__version__",
]
