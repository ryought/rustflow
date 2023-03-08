//! # `FlowGraphRaw` and `FlowGraph`
//!
//! Basic example implementation of flow network with constant cost
//!
//! `FlowGraphRaw` is a DiGraph whose edge is `FlowEdgeRaw`.
//!
//! `FlowEdgeRaw` has
//! * `demand`
//! * `capacity`
//! * `cost` Cost per unit flow
//! and additional information about an edge in `info`
//!
use super::{ConstCost, Cost, FlowEdge, FlowRateLike};
use petgraph::graph::DiGraph;

/// FlowGraph definition
pub type FlowGraph<F> = DiGraph<(), FlowEdgeBase<F>>;
pub type FlowGraphRaw<F, T> = DiGraph<(), FlowEdgeRaw<F, T>>;

/// Edge attributes used in FlowGraph.
/// This is a minimal example of min-cost-flow problem definition.
///
/// It has
/// * Demand l(e)
/// * Capacity u(e)
/// * Cost per unit flow c(e)
///
/// It can contain additional information in T.
#[derive(Debug, Copy, Clone)]
pub struct FlowEdgeRaw<F: FlowRateLike, T> {
    /// demand (lower limit of flow) of the edge l(e)
    pub demand: F,
    /// capacity (upper limit of flow) of the edge u(e)
    pub capacity: F,
    /// cost per unit flow
    pub cost: Cost,
    /// auxiliary informations
    pub info: T,
}

pub type FlowEdgeBase<F> = FlowEdgeRaw<F, ()>;

impl<F: FlowRateLike> FlowEdgeBase<F> {
    pub fn new(demand: F, capacity: F, cost: Cost) -> FlowEdgeBase<F> {
        FlowEdgeBase {
            demand,
            capacity,
            cost,
            info: (),
        }
    }
}

impl<F: FlowRateLike + std::fmt::Display, T> std::fmt::Display for FlowEdgeRaw<F, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}] {}", self.demand, self.capacity, self.cost)
    }
}

impl<F: FlowRateLike, T> FlowEdge<F> for FlowEdgeRaw<F, T> {
    fn demand(&self) -> F {
        self.demand
    }
    fn capacity(&self) -> F {
        self.capacity
    }
}

impl<F: FlowRateLike, T> ConstCost for FlowEdgeRaw<F, T> {
    fn cost(&self) -> Cost {
        self.cost
    }
}
