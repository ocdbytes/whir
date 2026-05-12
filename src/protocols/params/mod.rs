// This module contains the parameter selection and security target logic.

pub(crate) mod bounds;
pub mod code_switch;
pub mod irs_commit;
pub mod mask_proximity;
pub mod plan;
pub mod spec;
pub mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;
