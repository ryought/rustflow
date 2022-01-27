//! Flow graph definitions
//! - FlowEdge, FlowEdgeRaw<T>
//! - FlowGraph, FlowGraphRaw<T>
//! - Flow
use petgraph::graph::{DiGraph, EdgeIndex};
use std::collections::HashMap;

/// Edge attributes used in FlowGraph
/// It has
/// - demand l
/// - capacity u
/// - cost per flow c
/// [l, u], c
///
/// it can contain additional information in T.
#[derive(Debug, Copy, Clone)]
pub struct FlowEdgeRaw<T> {
    /// demand (lower limit of flow) of the edge l(e)
    pub demand: u32,
    /// capacity (upper limit of flow) of the edge u(e)
    pub capacity: u32,
    /// cost per unit flow
    pub cost: f64,
    /// auxiliary informations
    pub info: T,
}

pub type FlowEdge = FlowEdgeRaw<()>;

impl FlowEdge {
    pub fn new(demand: u32, capacity: u32, cost: f64) -> FlowEdge {
        FlowEdge {
            demand,
            capacity,
            cost,
            info: (),
        }
    }
}

impl<T> std::fmt::Display for FlowEdgeRaw<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}] {}", self.demand, self.capacity, self.cost)
    }
}

/// FlowGraph definition
pub type FlowGraph = DiGraph<(), FlowEdge>;
pub type FlowGraphRaw<T> = DiGraph<(), FlowEdgeRaw<T>>;

/// Flow definitions
///
/// Flow f is a mapping of u32 f(e) to each edge e
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Flow(HashMap<EdgeIndex, u32>);

impl Flow {
    pub fn empty() -> Flow {
        let hm = HashMap::new();
        Flow(hm)
    }
    pub fn zero<N, E>(graph: &DiGraph<N, E>) -> Flow {
        let mut hm = HashMap::new();
        for e in graph.edge_indices() {
            hm.insert(e, 0);
        }
        Flow(hm)
    }
    pub fn from(hm: HashMap<EdgeIndex, u32>) -> Flow {
        Flow(hm)
    }
    pub fn from_vec(vec: &[(EdgeIndex, u32)]) -> Flow {
        let mut hm = HashMap::new();
        for (e, f) in vec.iter() {
            hm.insert(*e, *f);
        }
        Flow(hm)
    }
    pub fn from_fn<N, E>(graph: &DiGraph<N, E>, f: fn(EdgeIndex) -> u32) -> Flow {
        let mut hm = HashMap::new();
        for e in graph.edge_indices() {
            hm.insert(e, f(e));
        }
        Flow(hm)
    }
    pub fn is_valid<N, E>(&self, _graph: &DiGraph<N, E>) -> bool {
        // TODO
        true
    }
    pub fn get(&self, e: EdgeIndex) -> Option<u32> {
        self.0.get(&e).cloned()
    }
    pub fn set(&mut self, e: EdgeIndex, v: u32) {
        self.0.insert(e, v);
    }
    pub fn has(&self, e: EdgeIndex) -> bool {
        self.0.contains_key(&e)
    }
    pub fn total_cost<N, E: EdgeCost>(&self, graph: &DiGraph<N, E>) -> f64 {
        graph
            .edge_indices()
            .map(|e| {
                let ew = graph.edge_weight(e).unwrap();
                let f = self.get(e).unwrap();
                ew.cost(f)
            })
            .sum()
    }
}

///
/// cost trait
///
pub trait EdgeCost {
    fn cost(&self, flow: u32) -> f64;
}

impl<T> EdgeCost for FlowEdgeRaw<T> {
    fn cost(&self, flow: u32) -> f64 {
        self.cost * flow as f64
    }
}
