pub mod flow;
pub mod mocks;
pub mod residue;
pub mod utils;
pub mod zero_demand;

use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use std::fmt;

use flow::{Flow, FlowGraphRaw};
use residue::improve_flow;
use zero_demand::{find_initial_flow, is_zero_demand_flow_graph};

pub fn min_cost_flow<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Option<Flow> {
    let init_flow = find_initial_flow(graph);
    match init_flow {
        Some(flow) => Some(min_cost_flow_from(graph, &flow)),
        None => None,
    }
}

pub fn min_cost_flow_from_zero<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Flow {
    assert!(is_zero_demand_flow_graph(&graph));
    let flow = Flow::zero(graph);
    min_cost_flow_from(graph, &flow)
}

pub fn min_cost_flow_from<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>, init_flow: &Flow) -> Flow {
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
