pub mod min_flow;

use crate::min_flow::*;
use petgraph::algo::bellman_ford;
use petgraph::algo::dijkstra;
use petgraph::algo::toposort;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use std::collections::HashMap;

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

fn show_toposort<'a>(g: &'a DiGraph<u8, ()>) {
    // should pass reference of graph
    match toposort(&g, None) {
        Ok(order) => {
            for node in order {
                g.node_weight(node).map(|w| {
                    println!("node:{:?} {:?}", node, w);
                    w
                });
            }
        }
        Err(err) => {
            let node = err.node_id();
            g.node_weight(node).map(|w| {
                println!("cycle node: {:?} {:?}", node, w);
            });
        }
    };
}

fn test_graph1() -> DiGraph<u8, ()> {
    let mut graph: DiGraph<u8, ()> = Graph::new();
    let v1 = graph.add_node(20);
    let v2 = graph.add_node(10);
    let v3 = graph.add_node(30);
    let e1 = graph.add_edge(v1, v2, ());
    let e2 = graph.add_edge(v2, v3, ());
    let e3 = graph.add_edge(v3, v1, ());
    graph
}

pub fn scenario1() {
    let mut g = Graph::<(), ()>::new();
    g.extend_with_edges(&[(0, 1), (1, 2), (1, 3), (3, 4)]);
    println!("{:?}", Dot::with_config(&g, &[]));

    println!("{}", g.node_count());
    println!("{}", g.edge_count());
}

pub fn scenario2() {
    let mut g = DiGraph::<u32, u32>::new();
    let a = g.add_node(111);
    let b = g.add_node(222);
    let c = g.add_node(333);
    let e1 = g.add_edge(a, b, 10);
    let e1 = g.add_edge(a, b, 100);
    let e2 = g.add_edge(a, c, 20);
    println!("{:?}", Dot::with_config(&g, &[]));

    // node -> edge
    // by two nodes
    println!("(a, b) in G?: {:?}", g.contains_edge(a, b));
    println!("(a, b): {:?}", g.find_edge(a, b));

    // by one nodes
    // list of nodes
    for v in g.neighbors_directed(a, Direction::Outgoing) {
        println!("neighbor of a: {:?}", v);
    }
    // list of edges
    for e in g.edges_directed(a, Direction::Outgoing) {
        println!("outgoing edge of a: {:?}", e);
    }

    // edge -> node
    println!("e1 = {:?}", g.edge_endpoints(e1));

    let x = g.node_weight(a).unwrap();
    println!("a={:?}", x);

    println!("node itering");
    for node in g.node_indices() {
        println!(
            "node {:?} {:?} {:?}",
            node,
            g.node_weight(node).unwrap(),
            node.index()
        );
    }

    // delete a
    g.remove_node(a);

    println!("c={:?}", c);
    // should panic
    // println!("{:?}", g.node_weight(c).unwrap());
    println!("c={:?}", c.index());

    // check nodes again
    println!("node itering");
    for node in g.node_indices() {
        println!(
            "node {:?} {:?} {:?}",
            node,
            g.node_weight(node).unwrap(),
            node.index()
        );
    }
}

fn main3() {
    let mut graph: DiGraph<u8, ()> = Graph::new();
    let v1 = graph.add_node(20);
    let v2 = graph.add_node(10);
    let v3 = graph.add_node(30);
    let e1 = graph.add_edge(v1, v2, ());
    let e1 = graph.add_edge(v1, v3, ());
    println!("{:?}", v1);
    println!("{:?}", v2);
    println!("{:?}", v3);
    println!("{:?}", e1);
    println!("{:?}", Dot::with_config(&graph, &[]));

    // mapping
    let g2 = graph.map(|_, vp| vp.clone() + 1, |_, ep| ep.clone());
    println!("{:?}", Dot::with_config(&g2, &[]));

    // removing
    graph.remove_node(v1);
    println!("{:?}", Dot::with_config(&graph, &[]));

    show_toposort(&g2);

    let g3 = test_graph1();
    println!("{:?}", Dot::with_config(&g3, &[]));
    show_toposort(&g3);
}

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