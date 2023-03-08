//!
//! FlowEdge and constant cost function `ConstCost` definition
//!
use super::{Cost, FlowRateLike};

/// Edge of FlowGraph
///
/// * `demand()`: demand `l(e)`
/// * `capacity()`: capacity `u(e)`
///
/// cost is either `ConstCost` or `ConvexCost`
///
/// `[l, u], c`
pub trait FlowEdge<F: FlowRateLike> {
    /// Demand of the edge, Lower limit of the flow
    fn demand(&self) -> F;
    /// Capacity of the edge, Upper limit of the flow
    fn capacity(&self) -> F;
}

/// Edge of FlowGraph with constant cost
///
/// * `cost()`: cost per unit flow `c(e)`
///
/// `[l, u], c`
pub trait ConstCost {
    /// constant Cost-per-unit-flow of the edge
    fn cost(&self) -> Cost;
}
