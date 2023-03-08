//!
//! Solve minimum-cost flow problem defined on `petgraph::DiGraph`
//!
//! # Usage
//!
//!
//!
//! # Solvers
//!
//! ## minimum-cost flow problem
//! * min_cost_flow
//! * min_cost_flow_convex
//!
//! ## manual search of neighboring flow
//! * enumerate_neighboring_flows
//!
//! # Detail of module structure
//!
//! Flow amount F: FlowRateLike (usize or f64)
//! Cost f64
//! Cost function: constant or convex function from F to Cost
//! FlowEdge: demand and capacity for each edge
//!
//! * traits definitions for writing problems
//!     * const
//!     * convex
//!     * flow
//!     * flow_edge
//!     * flow_rate
//! * submodule of solvers
//!     * residue
//!     * utils
//!     * zero_demand
//! * tests
//!     * mocks
//!
pub mod base;
pub mod convex;
pub mod flow;
pub mod flow_edge;
pub mod flow_rate;
pub mod mocks;
pub mod residue;
pub mod utils;
pub mod zero_demand;

// problem definitions
pub use convex::ConvexCost;
pub use flow::{EdgeCost, Flow};
pub use flow_edge::{ConstCost, FlowEdge};
pub use flow_rate::FlowRateLike;

// solvers
use convex::{restore_convex_flow, to_fixed_flow_graph};
pub use flow::total_cost;
use flow::{assert_valid_flow, is_valid_flow};
use petgraph::graph::DiGraph;
use residue::{
    enumerate_neighboring_flows_in_residue, flow_to_residue_convex, improve_flow,
    improve_flow_convex, CycleDetectMethod, UpdateInfo,
};
use zero_demand::{find_initial_flow, is_zero_demand_flow_graph};
// use convex::is_convex_cost_flow_graph;
// use utils::draw_with_flow;

///
/// A cost (of edges passing some amount of flow) in min flow problem definition.
///
/// Type of Constant flow and return type of convex cost flow should be this type.
///
pub type Cost = f64;

//
// public solver functions
//

///
/// Find minimum cost flow on the FlowGraph
///
pub fn min_cost_flow<F, N, E>(graph: &DiGraph<N, E>) -> Option<Flow<F>>
where
    F: FlowRateLike,
    N: std::fmt::Debug,
    E: FlowEdge<F> + ConstCost + std::fmt::Debug,
{
    let init_flow = find_initial_flow(graph);

    match init_flow {
        Some(flow) => {
            // draw_with_flow(graph, &flow);
            Some(min_cost_flow_from(graph, &flow))
        }
        None => None,
    }
}

///
/// Find minimum cost flow on the ConvexFlowGraph
///
pub fn min_cost_flow_convex<F, N, E>(graph: &DiGraph<N, E>) -> Option<Flow<F>>
where
    F: FlowRateLike,
    N: std::fmt::Debug,
    E: FlowEdge<F> + ConvexCost<F> + std::fmt::Debug,
{
    // (1) convert to normal FlowGraph and find the min-cost-flow
    let fg = match to_fixed_flow_graph(graph) {
        Some(fg) => fg,
        None => return None,
    };

    let fg_flow = match min_cost_flow(&fg) {
        Some(fg_flow) => fg_flow,
        None => return None,
    };

    // (2) convert-back to the flow on the ConvexFlowGraph
    Some(restore_convex_flow(&fg_flow, &fg, &graph))
}

///
/// Find minimum cost flow on the Graph whose edge is ConvexFlowEdge.
/// This solver requires less memory.
///
pub fn min_cost_flow_convex_fast<F, N, E>(graph: &DiGraph<N, E>) -> Option<Flow<F>>
where
    F: FlowRateLike,
    N: std::fmt::Debug,
    E: FlowEdge<F> + ConvexCost<F> + std::fmt::Debug,
{
    // (1) find the initial flow, by assigning constant cost to the flow.
    let init_flow = find_initial_flow(graph);

    // (2) upgrade the flow, by finding a negative cycle in residue graph.
    match init_flow {
        Some(flow) => {
            // draw_with_flow(graph, &flow);
            Some(min_cost_flow_from_convex(graph, &flow))
        }
        None => None,
    }
}

//
// internal functions
//

///
/// Find minimum cost flow of the special FlowGraph, whose demand is always zero.
///
pub fn min_cost_flow_from_zero<F, N, E>(graph: &DiGraph<N, E>) -> Flow<F>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConstCost,
{
    assert!(is_zero_demand_flow_graph(&graph));
    let flow = Flow::new(graph.edge_count(), F::zero());
    min_cost_flow_from(graph, &flow)
}

///
/// Find minimum cost by starting from the specified flow values.
///
pub fn min_cost_flow_from<F, N, E>(graph: &DiGraph<N, E>, init_flow: &Flow<F>) -> Flow<F>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConstCost,
{
    let mut flow = init_flow.clone();

    loop {
        assert_valid_flow(&flow, &graph);
        // solution of const cost is independent of cycle detection method
        // use BellmanFord because it is fastest.
        match improve_flow(graph, &flow, CycleDetectMethod::BellmanFord) {
            Some(new_flow) => {
                flow = new_flow;
                continue;
            }
            None => {
                break;
            }
        };
    }

    flow
}

///
/// Find minimum cost by starting from the specified flow values in ConvexCost Flowgraph.
///
pub fn min_cost_flow_from_convex<F, N, E>(graph: &DiGraph<N, E>, init_flow: &Flow<F>) -> Flow<F>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let mut flow = init_flow.clone();

    // TODO assert graph edge has convex function?
    // assert!(is_convex_cost_flow_graph(graph));

    loop {
        assert!(is_valid_flow(&flow, &graph));
        // solution of convex cost is independent of cycle detection method
        // use BellmanFord because it is fastest.
        match improve_flow_convex(graph, &flow, CycleDetectMethod::BellmanFord) {
            Some(new_flow) => {
                flow = new_flow;
                continue;
            }
            None => {
                break;
            }
        };
    }

    flow
}

///
/// enumerate neighboring flows of current flow on MinFlowNetwork.
///
pub fn enumerate_neighboring_flows<F, N, E>(
    graph: &DiGraph<N, E>,
    flow: &Flow<F>,
    max_depth: Option<usize>,
) -> Vec<(Flow<F>, UpdateInfo)>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let rg = flow_to_residue_convex(graph, flow);
    enumerate_neighboring_flows_in_residue(&rg, flow, max_depth)
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_rate_like_float() {
        assert_eq!(10.1_f64.to_usize(), 10);
        assert_eq!(10.9_f64.to_usize(), 10);
    }
}
