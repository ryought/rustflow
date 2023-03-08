use super::base::{FlowEdgeBase, FlowGraph};
use super::flow::Flow;
use petgraph::graph::{EdgeIndex, Graph};

///
/// very small flow network
/// with triangle circular topology
///
pub fn mock_flow_network1() -> (FlowGraph<usize>, Flow<usize>) {
    let mut graph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, FlowEdgeBase::new(0, 10, -1.0));
    graph.add_edge(b, c, FlowEdgeBase::new(0, 10, -2.0));
    graph.add_edge(c, a, FlowEdgeBase::new(0, 10, -2.0));
    let f = vec![10, 10, 10].into();
    (graph, f)
}

///
/// mock_flow_network1 version of f64
///
pub fn mock_flow_network1_float() -> (FlowGraph<f64>, Flow<f64>) {
    let mut graph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, FlowEdgeBase::new(0.0, 10.0, -1.0));
    graph.add_edge(b, c, FlowEdgeBase::new(0.0, 10.0, -2.0));
    graph.add_edge(c, a, FlowEdgeBase::new(0.0, 10.0, -2.0));
    let f = vec![10.0, 10.0, 10.0].into();
    (graph, f)
}

///
/// mock network cited from Genome-scale algorithm design p48
///
pub fn mock_flow_network2() -> (FlowGraph<usize>, Flow<usize>) {
    let mut g = Graph::new();
    let s = g.add_node(());
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let d = g.add_node(());
    let e = g.add_node(());
    let f = g.add_node(());
    let t = g.add_node(());
    const INF: usize = 100000;
    let e0 = g.add_edge(s, a, FlowEdgeBase::new(0, INF, 0.0));
    let e1 = g.add_edge(a, b, FlowEdgeBase::new(2, 4, 2.0));
    let e2 = g.add_edge(a, d, FlowEdgeBase::new(9, 13, 1.0));
    let e3 = g.add_edge(b, c, FlowEdgeBase::new(2, 3, 2.0));
    let e4 = g.add_edge(d, c, FlowEdgeBase::new(0, 6, 1.0));
    let e5 = g.add_edge(c, t, FlowEdgeBase::new(4, 8, 1.0));
    let e6 = g.add_edge(d, f, FlowEdgeBase::new(0, 10, 3.0));
    let e7 = g.add_edge(s, e, FlowEdgeBase::new(0, 6, 1.0));
    let e8 = g.add_edge(e, f, FlowEdgeBase::new(0, 5, 1.0));
    let e9 = g.add_edge(f, t, FlowEdgeBase::new(7, 13, 3.0));
    let ea = g.add_edge(t, s, FlowEdgeBase::new(17, 17, 0.0));

    let f = vec![12, 2, 10, 2, 6, 8, 4, 5, 5, 9, 17].into();
    (g, f)
}

pub fn mock_flow_network3() -> (FlowGraph<usize>, Flow<usize>) {
    let mut g = Graph::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let d = g.add_node(());
    let e0 = g.add_edge(a, b, FlowEdgeBase::new(0, 2, 1.0));
    let e1 = g.add_edge(a, c, FlowEdgeBase::new(0, 2, -2.0));
    let e2 = g.add_edge(b, d, FlowEdgeBase::new(0, 2, 3.0));
    let e3 = g.add_edge(c, d, FlowEdgeBase::new(0, 1, 4.0));
    let e4 = g.add_edge(d, a, FlowEdgeBase::new(2, 2, 0.0));

    let f = vec![1, 1, 1, 1, 2].into();
    (g, f)
}

pub fn mock_flow_network_parallel_edge1() -> (FlowGraph<usize>, Flow<usize>) {
    let mut graph = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let e0 = graph.add_edge(a, b, FlowEdgeBase::new(0, 2, -1.0));
    let e1 = graph.add_edge(b, c, FlowEdgeBase::new(0, 2, -1.0));
    let e2 = graph.add_edge(b, c, FlowEdgeBase::new(0, 2, -2.0));
    let e3 = graph.add_edge(c, a, FlowEdgeBase::new(0, 2, 0.0));

    let f = vec![2, 0, 2, 2].into();
    (graph, f)
}

pub fn mock_flow_network_parallel_edge2() -> (FlowGraph<usize>, Flow<usize>) {
    let mut graph = FlowGraph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let e1 = graph.add_edge(a, b, FlowEdgeBase::new(0, 2, 1.0));
    let e2 = graph.add_edge(b, c, FlowEdgeBase::new(0, 2, 1.0));
    let e3 = graph.add_edge(b, c, FlowEdgeBase::new(0, 2, 2.0));
    let e4 = graph.add_edge(c, a, FlowEdgeBase::new(2, 2, 0.0));

    let f = vec![2, 2, 0, 2].into();
    (graph, f)
}

#[cfg(test)]
mod tests {
    use super::super::min_cost_flow;
    use super::super::utils::{draw, draw_with_flow};
    use super::*;

    #[test]
    fn test_mock_flow_network1() {
        let (g, f_true) = mock_flow_network1();
        let f = min_cost_flow(&g).unwrap();
        assert!(f_true == f);
    }

    #[test]
    fn test_mock_flow_network1_float() {
        let (g, f_true) = mock_flow_network1_float();
        println!("graph");
        let f = min_cost_flow(&g).unwrap();
        assert!(f_true == f);
    }

    #[test]
    fn test_mock_flow_network2() {
        let (g, f_true) = mock_flow_network2();
        let f = min_cost_flow(&g).unwrap();
        draw_with_flow(&g, &f);
        println!("{:?}", f);
        assert!(f_true == f);
    }

    #[test]
    fn test_mock_flow_network3() {
        let (g, f_true) = mock_flow_network3();
        let f = min_cost_flow(&g).unwrap();
        draw_with_flow(&g, &f);
        println!("{:?}", f);
        assert!(f_true == f);
    }

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
