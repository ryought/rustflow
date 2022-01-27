use super::flow::{Flow, FlowEdge, FlowGraph};
use petgraph::graph::{EdgeIndex, Graph};

/// mock graph generation functions
pub fn mock_flow_network() -> FlowGraph {
    let mut graph: FlowGraph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, FlowEdge::new(0, 10, -1.0));
    graph.add_edge(b, c, FlowEdge::new(0, 10, -2.0));
    graph.add_edge(c, a, FlowEdge::new(0, 10, -2.0));
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
    const INF: u32 = 100000;
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

pub fn mock_flow_network_parallel_edge1() -> (FlowGraph, Flow) {
    let mut graph: FlowGraph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let e0 = graph.add_edge(a, b, FlowEdge::new(0, 2, -1.0));
    let e1 = graph.add_edge(b, c, FlowEdge::new(0, 2, -1.0));
    let e2 = graph.add_edge(b, c, FlowEdge::new(0, 2, -2.0));
    let e3 = graph.add_edge(c, a, FlowEdge::new(0, 2, 0.0));

    let f = Flow::from_vec(&[(e0, 2), (e1, 0), (e2, 2), (e3, 2)]);
    (graph, f)
}

pub fn mock_flow_network_parallel_edge2() -> (FlowGraph, Flow) {
    let mut graph: FlowGraph = FlowGraph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let e1 = graph.add_edge(a, b, FlowEdge::new(0, 2, 1.0));
    let e2 = graph.add_edge(b, c, FlowEdge::new(0, 2, 1.0));
    let e3 = graph.add_edge(b, c, FlowEdge::new(0, 2, 2.0));
    let e4 = graph.add_edge(c, a, FlowEdge::new(2, 2, 0.0));

    let f = Flow::from_vec(&[(e1, 2), (e2, 2), (e3, 0), (e4, 2)]);
    (graph, f)
}

#[cfg(test)]
mod tests {
    use super::super::utils::{draw, draw_with_flow};
    use super::*;
    use crate::min_flow::min_cost_flow;
    use petgraph::graph::EdgeIndex;

    #[test]
    fn test_mock_flow_network_parallel_edge1() {
        let (g, f_true) = mock_flow_network_parallel_edge1();
        let f = min_cost_flow(&g).unwrap();
        assert!(f_true == f);
    }

    #[test]
    fn test_mock_flow_network_parallel_edge2() {
        let (g, f_true) = mock_flow_network_parallel_edge2();
        draw(&g);
        let f = min_cost_flow(&g).unwrap();
        draw_with_flow(&g, &f);

        println!("{:?}", f);
        println!("{:?}", f_true);
        assert!(f_true == f);
    }
}
