//!
//! utils
//!
use super::mocks;
use super::{find_initial_flow, min_cost_flow};
use petgraph::dot::Dot;
use petgraph::graph::Graph;
use petgraph::EdgeType;

pub fn test() {
    let g = mocks::mock_flow_network2();
    draw(&g);

    let f = find_initial_flow(&g);
    println!("initia_flow={:?}", f);

    let f = min_cost_flow(&g);
    println!("{:?}", f);
}

pub fn draw<'a, N: 'a, E: 'a, Ty, Ix>(graph: &'a Graph<N, E, Ty, Ix>)
where
    E: std::fmt::Debug,
    N: std::fmt::Debug,
    Ty: EdgeType,
    Ix: petgraph::graph::IndexType,
{
    println!("{:?}", Dot::with_config(&graph, &[]));
}

// pub fn draw_with_flow
