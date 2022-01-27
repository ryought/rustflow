//! Flow graph definitions
//! - FlowEdge, FlowEdgeRaw<T>
//! - FlowGraph, FlowGraphRaw<T>
//! - Flow
use petgraph::graph::{DiGraph, EdgeIndex};
use petgraph::visit::EdgeRef; // for EdgeReference.id()
use petgraph::Direction;
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
/// Check if the flow is valid, i.e. it satisfies
/// - flows of all edges are defined
/// - demand and capacity constraint
/// - flow constraint
///
pub fn is_valid_flow<T>(flow: &Flow, graph: &FlowGraphRaw<T>) -> bool {
    is_defined_for_all_edges(flow, graph)
        && is_in_demand_and_capacity(flow, graph)
        && is_satisfying_flow_constraint(flow, graph)
}

///
/// Check if the flow contains all edges
///
pub fn is_defined_for_all_edges<T>(flow: &Flow, graph: &FlowGraphRaw<T>) -> bool {
    graph.edge_indices().all(|e| flow.get(e).is_some())
}

///
/// For each edge, the flow must satisfy `demand <= flow <= capacity`.
/// This function checks it
///
pub fn is_in_demand_and_capacity<T>(flow: &Flow, graph: &FlowGraphRaw<T>) -> bool {
    graph.edge_indices().all(|e| {
        let ew = graph.edge_weight(e).unwrap();
        match flow.get(e) {
            Some(f) => (ew.demand <= f) && (f <= ew.capacity),
            None => false,
        }
    })
}

///
/// For each node,
/// (the sum of out-going flows) should be equal to (the sum of in-coming flows).
///
pub fn is_satisfying_flow_constraint<T>(flow: &Flow, graph: &FlowGraphRaw<T>) -> bool {
    graph.node_indices().all(|v| {
        let in_flow: u32 = graph
            .edges_directed(v, Direction::Incoming)
            .map(|er| flow.get(er.id()).unwrap())
            .sum();
        let out_flow: u32 = graph
            .edges_directed(v, Direction::Outgoing)
            .map(|er| flow.get(er.id()).unwrap())
            .sum();
        in_flow == out_flow
    })
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

//
// tests
//
#[cfg(test)]
mod tests {
    use super::super::mocks::mock_flow_network1;
    use super::super::utils::draw;
    use super::*;

    #[test]
    fn flow_valid_tests() {
        let (g, _) = mock_flow_network1();
        draw(&g);

        // this is valid flow
        let f1 = Flow::from_vec(&[
            (EdgeIndex::new(0), 5),
            (EdgeIndex::new(1), 5),
            (EdgeIndex::new(2), 5),
        ]);
        assert!(is_defined_for_all_edges(&f1, &g));
        assert!(is_in_demand_and_capacity(&f1, &g));
        assert!(is_satisfying_flow_constraint(&f1, &g));
        assert!(is_valid_flow(&f1, &g));

        // this flow overs the capacity
        let f2 = Flow::from_vec(&[
            (EdgeIndex::new(0), 100),
            (EdgeIndex::new(1), 100),
            (EdgeIndex::new(2), 100),
        ]);
        assert!(is_defined_for_all_edges(&f2, &g));
        assert!(!is_in_demand_and_capacity(&f2, &g));
        assert!(is_satisfying_flow_constraint(&f2, &g));
        assert!(!is_valid_flow(&f2, &g));

        // this is a flow which not satisfies the flow constraint
        let f3 = Flow::from_vec(&[
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 5),
            (EdgeIndex::new(2), 1),
        ]);
        assert!(is_defined_for_all_edges(&f3, &g));
        assert!(is_in_demand_and_capacity(&f3, &g));
        assert!(!is_satisfying_flow_constraint(&f3, &g));
        assert!(!is_valid_flow(&f3, &g));

        // this is a partial flow
        let f4 = Flow::from_vec(&[(EdgeIndex::new(0), 1)]);
        assert!(!is_defined_for_all_edges(&f4, &g));
        assert!(!is_valid_flow(&f4, &g));
    }
}
