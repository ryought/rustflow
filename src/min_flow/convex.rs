//!
//! Flow network definitions for convex cost.
//!
use super::flow::EdgeCost;
use petgraph::graph::{DiGraph, EdgeIndex};
use std::collections::HashMap;

///
///
///
#[derive(Debug, Copy, Clone)]
pub struct ConvexFlowEdge {
    /// demand (lower limit of flow) of the edge l(e)
    pub demand: u32,
    /// capacity (upper limit of flow) of the edge u(e)
    pub capacity: u32,
    /// cost function
    /// it is a convex function of the current flow
    pub convex_cost: fn(u32) -> f64,
}

impl ConvexFlowEdge {
    pub fn new(demand: u32, capacity: u32, convex_cost: fn(u32) -> f64) -> ConvexFlowEdge {
        ConvexFlowEdge {
            demand,
            capacity,
            convex_cost,
        }
    }
}

impl EdgeCost for ConvexFlowEdge {
    fn cost(&self, flow: u32) -> f64 {
        (self.convex_cost)(flow)
    }
}

pub type ConvexFlowGraph = DiGraph<(), ConvexFlowEdge>;

//
// conversion functions
//

///
/// convert convex flow graph to the (normal, static cost) flow graph
///
/// create an parallel edges
///
pub fn to_flow_graph(graph: &ConvexFlowGraph) {}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_flow_edge_new() {
        let e = ConvexFlowEdge::new(1, 5, |f| 10.0 * f as f64);
        assert_eq!(50.0, (e.convex_cost)(5));
        assert_eq!(50.0, e.cost(5));
        assert_eq!(1, e.demand);
        assert_eq!(5, e.capacity);
    }
}
