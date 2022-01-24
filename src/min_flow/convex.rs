//!
//! Flow network definitions for convex cost.
//!
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
    pub cost: fn(u32) -> f64,
}

impl ConvexFlowEdge {
    pub fn new(demand: u32, capacity: u32, cost: fn(u32) -> f64) -> ConvexFlowEdge {
        ConvexFlowEdge {
            demand,
            capacity,
            cost,
        }
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
