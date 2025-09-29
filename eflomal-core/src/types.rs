use core::cmp::{max, min};

pub type Link = u16;
pub type Token = u32;

#[cfg(feature = "double-precision")]
pub type Count = f64;
#[cfg(not(feature = "double-precision"))]
pub type Count = f32;

pub const NULL_LINK: Link = 0xffff;

pub const JUMP_ARRAY_LEN: usize = 0x800;
pub const JUMP_SUM: usize = JUMP_ARRAY_LEN - 1;
pub const JUMP_MAX_EST: Count = 100.0;

pub const FERT_ARRAY_LEN: usize = 0x08;
pub const MAX_SENT_LEN: usize = 0x400;

pub const JUMP_ALPHA: Count = 0.5;
pub const FERT_ALPHA: Count = 0.5;
pub const LEX_ALPHA: Count = 0.001;
pub const NULL_ALPHA: Count = 0.001;

#[inline]
pub fn get_jump_index(i: isize, j: isize, len: usize) -> usize {
    let z = j - i + (JUMP_ARRAY_LEN as isize) / 2;
    let z = max(0, min((JUMP_ARRAY_LEN as isize) - 1, z));
    z as usize
}

#[inline]
pub fn get_fert_index(e: Token, fert: usize) -> usize {
    let k = core::cmp::min(fert, FERT_ARRAY_LEN - 1);
    (e as usize) * FERT_ARRAY_LEN + k
}