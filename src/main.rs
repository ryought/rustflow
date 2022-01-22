use petgraph::algo::bellman_ford;
use petgraph::algo::dijkstra;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use std::collections::HashMap;

fn main() {
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
