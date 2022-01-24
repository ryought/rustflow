use super::{FlowEdge, FlowGraph};
use petgraph::graph::Graph;

/// mock graph generation functions
pub fn mock_flow_network() -> FlowGraph {
    let mut graph: FlowGraph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let e1 = graph.add_edge(a, b, FlowEdge::new(0, 10, -1.0));
    let e2 = graph.add_edge(b, c, FlowEdge::new(0, 10, -2.0));
    let e3 = graph.add_edge(c, a, FlowEdge::new(0, 10, -2.0));
    let f = graph.edge_weight(e1).unwrap();
    graph
}

/// mock network cited from Genome-scale algorithm design p48
pub fn mock_flow_network2() -> FlowGraph {
    let mut graph: FlowGraph = Graph::new();
    let s = graph.add_node(());
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    let e = graph.add_node(());
    let f = graph.add_node(());
    let t = graph.add_node(());
    let INF = 100000;
    graph.add_edge(s, a, FlowEdge::new(0, INF, 0.0));
    graph.add_edge(a, b, FlowEdge::new(2, 4, 2.0));
    graph.add_edge(a, d, FlowEdge::new(9, 13, 1.0));
    graph.add_edge(b, c, FlowEdge::new(2, 3, 2.0));
    graph.add_edge(d, c, FlowEdge::new(0, 6, 1.0));
    graph.add_edge(c, t, FlowEdge::new(4, 8, 1.0));
    graph.add_edge(d, f, FlowEdge::new(0, 10, 3.0));
    graph.add_edge(s, e, FlowEdge::new(0, 6, 1.0));
    graph.add_edge(e, f, FlowEdge::new(0, 5, 1.0));
    graph.add_edge(f, t, FlowEdge::new(7, 13, 3.0));
    graph.add_edge(t, s, FlowEdge::new(17, 17, 0.0));
    graph
}

pub fn mock_flow_network3() -> FlowGraph {
    let mut graph: FlowGraph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, FlowEdge::new(0, 2, 1.0));
    graph.add_edge(a, c, FlowEdge::new(0, 2, -2.0));
    graph.add_edge(b, d, FlowEdge::new(0, 2, 3.0));
    graph.add_edge(c, d, FlowEdge::new(0, 1, 4.0));
    graph.add_edge(d, a, FlowEdge::new(2, 2, 0.0));
    graph
}
