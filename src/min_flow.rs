pub mod convex;
pub mod flow;
pub mod mocks;
pub mod residue;
pub mod utils;
pub mod zero_demand;

use convex::ConvexFlowGraph;
use flow::{Flow, FlowGraphRaw};
use residue::improve_flow;
use zero_demand::{find_initial_flow, is_zero_demand_flow_graph};

//
// public functions
//

///
/// Find minimum cost flow on the FlowGraph
///
pub fn min_cost_flow<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Option<Flow> {
    let init_flow = find_initial_flow(graph);
    match init_flow {
        Some(flow) => Some(min_cost_flow_from(graph, &flow)),
        None => None,
    }
}

///
/// Find minimum cost flow on the ConvexFlowGraph
///
pub fn min_cost_flow_convex(graph: &ConvexFlowGraph) {
    // (1) convert to normal FlowGraph and find the min-cost-flow
    // (2) convert-back to the flow on the ConvexFlowGraph
}

//
// internal functions
//

///
/// Find minimum cost flow of the special FlowGraph, whose demand is always zero.
///
fn min_cost_flow_from_zero<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Flow {
    assert!(is_zero_demand_flow_graph(&graph));
    let flow = Flow::zero(graph);
    min_cost_flow_from(graph, &flow)
}

///
/// Find minimum cost by starting from the specified flow values.
///
fn min_cost_flow_from<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>, init_flow: &Flow) -> Flow {
    let mut flow = init_flow.clone();
    loop {
        println!("current flow: {:?} {}", flow, flow.total_cost(graph));
        match improve_flow(graph, &flow) {
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
