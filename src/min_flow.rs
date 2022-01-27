pub mod convex;
pub mod flow;
pub mod mocks;
pub mod residue;
pub mod utils;
pub mod zero_demand;

use convex::{restore_convex_flow, to_fixed_flow_graph, ConvexFlowGraph};
use flow::{is_valid_flow, Flow, FlowGraphRaw};
use residue::improve_flow;
use utils::draw_with_flow;
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
        Some(flow) => {
            draw_with_flow(graph, &flow);
            Some(min_cost_flow_from(graph, &flow))
        }
        None => None,
    }
}

///
/// Find minimum cost flow on the ConvexFlowGraph
///
pub fn min_cost_flow_convex(graph: &ConvexFlowGraph) -> Option<Flow> {
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
        assert!(is_valid_flow(&flow, &graph));
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
