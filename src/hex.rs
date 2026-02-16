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
    if val == 0xFF { None } else { Some(val) }
}
