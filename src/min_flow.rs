use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::collections::HashMap;
use std::fmt;

// Flow graph definitions

/// Edge attributes used in FlowGraph
/// It has
/// - demand l
/// - capacity u
/// - cost per flow c
/// [l, u], c
#[derive(Debug, Copy, Clone)]
pub struct FlowEdge {
    demand: u32,
    capacity: u32,
    cost: f64,
}

impl FlowEdge {
    pub fn new(demand: u32, capacity: u32, cost: f64) -> FlowEdge {
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

/// FlowGraph definition
pub type FlowGraph = DiGraph<(), FlowEdge>;

// Residue graph definitions

/// Edge attributes used in ResidueGraph
#[derive(Debug, Copy, Clone)]
pub struct ResidueEdge {
    count: u32,
    weight: f64,
    direction: ResidueDirection,
}

impl ResidueEdge {
    pub fn new(count: u32, weight: f64, direction: ResidueDirection) -> ResidueEdge {
        ResidueEdge {
            count,
            weight,
            direction,
        }
    }
}

/// Residue direction enum
/// residue edge has two types
/// - Up edge: increase(+1) of flow
/// - Down edge: decrease(-1) of flow
#[derive(Debug, Copy, Clone)]
enum ResidueDirection {
    Up,
    Down,
}

/// ResidueGraph definition
pub type ResidueGraph = DiGraph<(), ResidueEdge>;

/// Flow definitions
///
/// Flow f is a mapping of u32 f(e) to each edge e
/// TODO how to use EdgeId as a concrete type?
#[derive(Debug)]
pub struct Flow(HashMap<EdgeIndex, u32>);

impl Flow {
    pub fn zero(graph: &FlowGraph) -> Flow {
        let mut hm = HashMap::new();
        for e in graph.edge_indices() {
            // let w = graph.edge_weight(e).unwrap();
            hm.insert(e, 0);
        }
        Flow(hm)
    }
    pub fn from(hm: HashMap<EdgeIndex, u32>) -> Flow {
        Flow(hm)
    }
}

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

/// Convert FlowGraph and Flow
///  e = ([l,u],c), f
///
/// into
///
/// ResidueGraph
///  e1 = (u-f, +c) if u-f>0
///  e2 = (f-l, -c) if f-l>0
pub fn flow_to_residue(graph: &FlowGraph, flow: &Flow) -> ResidueGraph {
    let mut rg: ResidueGraph = Graph::new();
    // TODO how to convert with same node id?
    for node in graph.node_indices() {
        rg.add_node(());
    }
    rg
}

pub fn test() {
    let g = mock_flow_network();
    draw(&g);
    let f = Flow::zero(&g);
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
