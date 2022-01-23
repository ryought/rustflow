use petgraph::algo::bellman_ford;
use petgraph::algo::dijkstra;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Copy, Clone)]
struct FlowEdge {
    demand: u32,
    capacity: u32,
    cost: f64,
}

impl FlowEdge {
    fn new(demand: u32, capacity: u32, cost: f64) -> FlowEdge {
        FlowEdge {
            demand,
            capacity,
            cost,
        }
    }
}

impl fmt::Display for FlowEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{},{}] {}", self.demand, self.capacity, self.cost)
    }
}

type FlowGraph = DiGraph<(), FlowEdge>;

fn mock_flow_graph() -> FlowGraph {
    let mut graph: FlowGraph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, FlowEdge::new(0, 0, 0.0));
    graph.add_edge(a, c, FlowEdge::new(0, 0, 0.0));
    graph.add_edge(b, c, FlowEdge::new(0, 0, 0.0));
    graph
}

fn main() {}

fn main2() {
    let g = mock_flow_graph();
    println!("{:?}", Dot::with_config(&g, &[Config::NodeIndexLabel]));
}

fn main_old() {
    println!("Hello, world!");
    // let graph = DiGraph::<i32, ()>::from_edges(&[(1, 2), (2, 3), (3, 4), (1, 4)]);
    // let mut graph = DiGraph::new();
    let mut graph: Graph<(), (), Directed> = Graph::new();
    let a = graph.add_node(()); // node with no weight
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    let e = graph.add_node(());
    let f = graph.add_node(());
    let g = graph.add_node(());
    let h = graph.add_node(());
    let z = graph.add_node(());
    // graph.extend_with_edges(&[(0, 1, 2.0), (0, 1, 4.0), (0, 3, 4.0), (1, 2, 1.0)]);
    graph.extend_with_edges(&[
        (a, b),
        (b, c),
        (c, d),
        (d, a),
        (e, f),
        (b, e),
        (f, g),
        (g, h),
        (h, e),
    ]);
    println!("{:?}", Dot::with_config(&graph, &[Config::NodeIndexLabel]));

    // let path = bellman_ford(&graph, a);
    // let path = path.unwrap();
    // println!("{:?}", path.distances);
    // println!("{:?}", path.predecessors);

    let expected_res: HashMap<NodeIndex, usize> = [
        (a, 3),
        (b, 0),
        (c, 1),
        (d, 2),
        (e, 1),
        (f, 2),
        (g, 3),
        (h, 4),
    ]
    .iter()
    .cloned()
    .collect();
    let res = dijkstra(&graph, b, None, |_| 1);
    assert_eq!(res, expected_res);
    println!("{:?}", res);
}
