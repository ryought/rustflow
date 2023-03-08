//!
//! FlowRateLike trait for generics of Flow amount
//!
//! usize and f64 implements FlowRateLike
//!

use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

///
/// generic FlowRate
///
pub trait FlowRateLike:
    Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + Sum
    + Default
    + std::fmt::Debug
    + std::fmt::Display
{
    /// zero value = 0
    fn zero() -> Self;
    /// unit value = 1
    fn unit() -> Self;
    /// cast to f64
    fn to_f64(self) -> f64;
    /// cast to usize (by flooring)
    fn to_usize(self) -> usize;
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn large_const() -> Self;
    /// similary equal
    fn sim_eq(self, rhs: Self) -> bool;
    /// difference allowed to be regarded as a same value
    fn eps() -> Self;
}

impl FlowRateLike for usize {
    fn zero() -> usize {
        0
    }
    fn unit() -> usize {
        1
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn to_usize(self) -> usize {
        self
    }
    fn wrapping_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
    fn large_const() -> Self {
        100
    }
    fn sim_eq(self, rhs: Self) -> bool {
        // integer type does not need to consider the floating error
        self == rhs
    }
    fn eps() -> Self {
        0
    }
}

impl FlowRateLike for f64 {
    fn zero() -> Self {
        0.0
    }
    fn unit() -> Self {
        1.0
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn to_usize(self) -> usize {
        // flooring
        self as usize
    }
    fn wrapping_add(self, rhs: Self) -> Self {
        // no overflow
        self + rhs
    }
    fn wrapping_sub(self, rhs: Self) -> Self {
        // no overflow
        self - rhs
    }
    fn large_const() -> Self {
        100.0
    }
    fn sim_eq(self, rhs: Self) -> bool {
        (self - rhs).abs() <= Self::eps()
    }
    fn eps() -> Self {
        0.000000001
    }
}
