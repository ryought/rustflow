//!
//! Flow network definitions for convex cost.
//!
use super::flow::{EdgeCost, Flow, FlowEdgeRaw, FlowGraphRaw};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use std::collections::HashMap;

///
/// Edge attributes of ConvexFlowGraph
/// For each edge,
///   demand
///   capacity
///   convex cost function
/// are assigned.
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

///
///
///
#[derive(Debug, Copy, Clone)]
pub struct ConvexFlowEdgeInfo {
    /// This represents that the edge is created from the origin
    /// in ConvexFlowGraph
    origin: EdgeIndex,
    /// The FixedCostFlowEdge has demand=0 and capacity=1.
    /// if this edge has flow=1, the original edge in ConvexFlowGraph
    /// should have flow=flow_offset+1.
    flow_offset: u32,
}

pub type FixedCostFlowEdge = FlowEdgeRaw<ConvexFlowEdgeInfo>;

impl FixedCostFlowEdge {
    pub fn new(cost: f64, origin: EdgeIndex, flow_offset: u32) -> FixedCostFlowEdge {
        FixedCostFlowEdge {
            demand: 0,
            capacity: 1,
            cost,
            info: ConvexFlowEdgeInfo {
                origin,
                flow_offset,
            },
        }
    }
}

///
/// A special kind of flow graph, that will be converted from ConvexFlowGraph
/// Each edge has additional information of ConvexFlowEdgeInfo, that will be
/// used when convert it back to flow in the original ConvexFlowGraph.
///
pub type FixedCostFlowGraph = DiGraph<(), FixedCostFlowEdge>;

//
// conversion functions
//

///
/// convert convex flow graph to the (normal, fixed constant cost) flow graph
///
/// create an parallel edges
/// with static cost of `e.cost(f + 1) - e.cost(f)`
///
pub fn to_fixed_flow_graph(graph: &ConvexFlowGraph) -> Option<FixedCostFlowGraph> {
    // TODO assert that
    // (1) flow is actually convex
    // (2) capacity is finite

    let mut g: FixedCostFlowGraph = FixedCostFlowGraph::new();

    for e in graph.edge_indices() {
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();

        let edges: Vec<(NodeIndex, NodeIndex, FixedCostFlowEdge)> = (ew.demand..ew.capacity)
            .map(|f| {
                let cost = ew.cost(f + 1) - ew.cost(f);
                // should store the original edge information
                let fe = FixedCostFlowEdge::new(cost, e, f);
                (v, w, fe)
            })
            .collect();
        g.extend_with_edges(&edges);
    }

    Some(g)
}

pub fn restore_convex_flow(flow_in_fixed: &Flow) -> Flow {
    // TODO
    Flow::empty()
}

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
