use crate::HEX_LOOKUP;

/// Branchless hex character to nibble conversion using lookup table
/// Returns 0xFF for invalid characters
#[inline(always)]
pub(crate) fn hex_char_to_nibble(c: u8) -> u8 {
    // SAFETY: c is u8, so always in bounds of 256-element table
    unsafe { *HEX_LOOKUP.get_unchecked(c as usize) }
}

/// Convert a hex character to its numeric value (0-15)
/// Returns None if the character is not a valid hex digit
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn hex_char_to_val(c: u8) -> Option<u8> {
    let val = hex_char_to_nibble(c);
    if val == 0xFF {
        None
    } else {
        Some(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_digits() {
        for (i, c) in b"0123456789".iter().enumerate() {
            assert_eq!(hex_char_to_nibble(*c), i as u8);
        }
    }

    #[test]
    fn nibble_upper() {
        for (i, c) in b"ABCDEF".iter().enumerate() {
            assert_eq!(hex_char_to_nibble(*c), 10 + i as u8);
        }
    }

    #[test]
    fn nibble_lower() {
        for (i, c) in b"abcdef".iter().enumerate() {
            assert_eq!(hex_char_to_nibble(*c), 10 + i as u8);
        }
    }

    #[test]
    fn nibble_invalid() {
        assert_eq!(hex_char_to_nibble(b'g'), 0xFF);
        assert_eq!(hex_char_to_nibble(b'z'), 0xFF);
        assert_eq!(hex_char_to_nibble(b'@'), 0xFF);
        assert_eq!(hex_char_to_nibble(b' '), 0xFF);
        assert_eq!(hex_char_to_nibble(b'/'), 0xFF);
        assert_eq!(hex_char_to_nibble(b':'), 0xFF); // just past '9'
        assert_eq!(hex_char_to_nibble(b'G'), 0xFF); // just past 'F'
    }

    #[test]
    fn hex_val_valid() {
        assert_eq!(hex_char_to_val(b'0'), Some(0));
        assert_eq!(hex_char_to_val(b'f'), Some(15));
        assert_eq!(hex_char_to_val(b'A'), Some(10));
    }

    #[test]
    fn hex_val_invalid() {
        assert_eq!(hex_char_to_val(b'z'), None);
        assert_eq!(hex_char_to_val(b' '), None);
    }
}
