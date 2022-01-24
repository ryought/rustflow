pub mod mocks;
pub mod utils;

use itertools::Itertools;
use petgraph::algo::find_negative_cycle;
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

// Flow graph definitions

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
    demand: u32,
    /// capacity (upper limit of flow) of the edge u(e)
    capacity: u32,
    /// cost per unit flow
    cost: f64,
    /// auxiliary informations
    info: T,
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

impl<T> fmt::Display for FlowEdgeRaw<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{},{}] {}", self.demand, self.capacity, self.cost)
    }
}

/// FlowGraph definition
pub type FlowGraph = DiGraph<(), FlowEdge>;
pub type FlowGraphRaw<T> = DiGraph<(), FlowEdgeRaw<T>>;

// Residue graph definitions

/// Edge attributes used in ResidueGraph
#[derive(Debug, Default, Copy, Clone)]
pub struct ResidueEdge {
    /// The movable amount of the flow
    count: u32,
    /// Cost of the unit change of this flow
    weight: f64,
    /// Original edge index of the source graph
    target: EdgeIndex,
    /// +1 or -1
    direction: ResidueDirection,
}

impl ResidueEdge {
    pub fn new(
        count: u32,
        weight: f64,
        target: EdgeIndex,
        direction: ResidueDirection,
    ) -> ResidueEdge {
        ResidueEdge {
            count,
            weight,
            target,
            direction,
        }
    }
    pub fn only_weight(weight: f64) -> ResidueEdge {
        ResidueEdge {
            weight,
            // filled by default values
            target: EdgeIndex::new(0),
            count: 0,
            direction: ResidueDirection::Up,
        }
    }
}

// TODO implement FloatMeasure for ResidueEdge
// PartialOrd, Add<Self, Self>, FloatMeasure(zero, infinite)
impl PartialOrd for ResidueEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl PartialEq for ResidueEdge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl std::ops::Add for ResidueEdge {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ResidueEdge::only_weight(self.weight + other.weight)
    }
}

impl petgraph::algo::FloatMeasure for ResidueEdge {
    fn zero() -> Self {
        ResidueEdge::only_weight(0.)
    }
    fn infinite() -> Self {
        ResidueEdge::only_weight(1. / 0.)
    }
}

/// Residue direction enum
/// residue edge has two types
/// - Up edge: increase(+1) of flow
/// - Down edge: decrease(-1) of flow
#[derive(Debug, Copy, Clone)]
pub enum ResidueDirection {
    Up,
    Down,
}

impl Default for ResidueDirection {
    fn default() -> Self {
        ResidueDirection::Up
    }
}

/// ResidueGraph definition
pub type ResidueGraph = DiGraph<(), ResidueEdge>;

/// Flow definitions
///
/// Flow f is a mapping of u32 f(e) to each edge e
/// TODO how to use EdgeId as a concrete type?
#[derive(Debug, Clone)]
pub struct Flow(HashMap<EdgeIndex, u32>);

impl Flow {
    pub fn empty() -> Flow {
        let hm = HashMap::new();
        Flow(hm)
    }
    pub fn zero<T>(graph: &FlowGraphRaw<T>) -> Flow {
        let mut hm = HashMap::new();
        for e in graph.edge_indices() {
            hm.insert(e, 0);
        }
        Flow(hm)
    }
    pub fn from(hm: HashMap<EdgeIndex, u32>) -> Flow {
        Flow(hm)
    }
    pub fn from_fn<T>(graph: &FlowGraphRaw<T>, f: fn(EdgeIndex) -> u32) -> Flow {
        let mut hm = HashMap::new();
        for e in graph.edge_indices() {
            hm.insert(e, f(e));
        }
        Flow(hm)
    }
    pub fn is_valid<T>(&self, graph: &FlowGraphRaw<T>) -> bool {
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
    pub fn total_cost<T>(&self, graph: &FlowGraphRaw<T>) -> f64 {
        graph
            .edge_indices()
            .map(|e| {
                let ew = graph.edge_weight(e).unwrap();
                let c = ew.cost;
                let f = self.get(e).unwrap() as f64;
                c * f
            })
            .sum()
    }
}

/// Convert FlowGraph with Flow into ResidueGraph.
///
/// FlowGraph and Flow
/// v -> w
///  e = ([l,u],c), f
///
/// into
///
/// ResidueGraph
/// v -> w
///  e1 = (u-f, +c) if u-f>0
/// w -> v
///  e2 = (f-l, -c) if f-l>0
pub fn flow_to_residue<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>, flow: &Flow) -> ResidueGraph {
    let mut rg: ResidueGraph = Graph::new();

    // create two edges (Up and Down) for each edge
    for e in graph.edge_indices() {
        let f = flow.get(e).unwrap();
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();
        println!("{:?} {:?} {}", e, ew, f);

        let mut edges = Vec::new();
        if f < ew.capacity {
            // up movable
            edges.push((
                v,
                w,
                ResidueEdge::new(ew.capacity - f, ew.cost, e, ResidueDirection::Up),
            ));
        }
        if f > ew.demand {
            // down movable
            edges.push((
                w,
                v,
                ResidueEdge::new(f - ew.demand, -ew.cost, e, ResidueDirection::Down),
            ));
        }
        rg.extend_with_edges(&edges);
    }
    rg
}

fn pick_minimum_weight_edge(graph: &ResidueGraph, v: NodeIndex, w: NodeIndex) -> EdgeIndex {
    let er = graph
        .edges_connecting(v, w)
        .min_by(|e1, e2| {
            let w1 = e1.weight().weight;
            let w2 = e2.weight().weight;
            w1.partial_cmp(&w2).unwrap()
        })
        .unwrap();
    let e = er.id();
    println!("{:?} {:?} {:?}", v, w, e);
    e
}

/// Convert a cycle as nodes into as edges,
/// by choosing the minimum weight edge if there are parallel edges
fn node_list_to_edge_list(graph: &ResidueGraph, nodes: &[NodeIndex]) -> Vec<EdgeIndex> {
    let mut edges = Vec::new();

    // (1) nodes[i] and nodes[i+1] from i=0..n-1
    for (v, w) in nodes.iter().tuple_windows() {
        let edge = pick_minimum_weight_edge(graph, *v, *w);
        edges.push(edge);
    }

    // (2) tail and head, node[n-1] and node[0]
    let edge = pick_minimum_weight_edge(graph, nodes[nodes.len() - 1], nodes[0]);
    edges.push(edge);

    edges
}

fn apply_residual_edges_to_flow(flow: &Flow, rg: &ResidueGraph, edges: &[EdgeIndex]) -> Flow {
    let mut new_flow = flow.clone();

    // determine flow_change_amount
    // that is the minimum of ResidueEdge.count
    let flow_change_amount = edges
        .iter()
        .map(|&e| {
            let ew = rg.edge_weight(e).unwrap();
            ew.count
        })
        .min()
        .unwrap();

    for edge in edges {
        let ew = rg.edge_weight(*edge).unwrap();
        println!("{:?} {:?}", edge, ew);
        // convert back to the original edgeindex
        let original_edge = ew.target;
        let original_edge_flow = flow.get(original_edge).unwrap();

        let new_edge_flow = match ew.direction {
            ResidueDirection::Up => original_edge_flow + flow_change_amount,
            ResidueDirection::Down => original_edge_flow - flow_change_amount,
        };
        new_flow.set(original_edge, new_edge_flow);
    }

    new_flow
}

/// create a new improved flow from current flow
/// by upgrading along the negative weight cycle in the residual graph
pub fn improve_flow<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>, flow: &Flow) -> Option<Flow> {
    let rg = flow_to_residue(graph, flow);
    println!("improve_flow!");
    utils::draw(&rg);

    // find negative weight cycles
    let path = find_negative_cycle(&rg, NodeIndex::new(0));

    match path {
        Some(nodes) => {
            let edges = node_list_to_edge_list(&rg, &nodes);
            // apply these changes along the cycle to current flow
            let new_flow = apply_residual_edges_to_flow(&flow, &rg, &edges);
            println!("{:?}", new_flow);
            Some(new_flow)
        }
        None => None,
    }
}

pub fn min_cost_flow<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Option<Flow> {
    let init_flow = find_initial_flow(graph);
    match init_flow {
        Some(flow) => Some(min_cost_flow_from(graph, &flow)),
        None => None,
    }
}

pub fn min_cost_flow_from_zero<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Flow {
    // TODO check if graph is zero-demand
    // when non-zero demand graph
    let flow = Flow::zero(graph);
    min_cost_flow_from(graph, &flow)
}

pub fn min_cost_flow_from<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>, init_flow: &Flow) -> Flow {
    let mut flow = init_flow.clone();
    loop {
        println!("current flow: {:?} {}", flow, flow.total_cost(graph));
        match improve_flow(graph, &flow) {
            Some(new_flow) => {
                flow = new_flow;
                continue;
            }
            None => {
                break;
            }
        };
    }

    flow
}

//
// finding init valid flow
//
#[derive(Debug, Copy, Clone)]
pub struct ZeroDemandEdgeInfo {
    /// edge id in original graph
    origin: EdgeIndex,
    /// identifier of type-A or type-B
    kind: ZeroDemandEdgeKind,
}
#[derive(Debug, Copy, Clone)]
pub enum ZeroDemandEdgeKind {
    /// represents type-A edge
    /// e = ([0,l],-1)
    BelowDemand,
    /// represents type-B edge
    /// e = ([0,u-l],0)
    AboveDemand,
}
pub type ZeroDemandFlowEdge = FlowEdgeRaw<ZeroDemandEdgeInfo>;
impl ZeroDemandFlowEdge {
    pub fn new(
        demand: u32,
        capacity: u32,
        cost: f64,
        origin: EdgeIndex,
        kind: ZeroDemandEdgeKind,
    ) -> ZeroDemandFlowEdge {
        ZeroDemandFlowEdge {
            demand,
            capacity,
            cost,
            info: ZeroDemandEdgeInfo { origin, kind },
        }
    }
}
pub type ZeroDemandFlowGraph = DiGraph<(), ZeroDemandFlowEdge>;

/// To determine all-zero flow is valid or not
/// we should know whether the given graph is demand-less
/// that is all demands of the edges are 0.
fn is_zero_demand_flow_graph<T>(graph: &FlowGraphRaw<T>) -> bool {
    graph.edge_indices().all(|e| {
        let ew = graph.edge_weight(e).unwrap();
        ew.demand == 0
    })
}

/// sum of edge demand
fn sum_of_demand<T>(graph: &FlowGraphRaw<T>) -> u32 {
    graph
        .edge_indices()
        .map(|e| {
            let ew = graph.edge_weight(e).unwrap();
            ew.demand
        })
        .sum()
}

fn to_zero_demand_graph<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> ZeroDemandFlowGraph {
    let mut zdg: ZeroDemandFlowGraph = Graph::new();

    // create two edges (type-A and type-B) for each edge
    for e in graph.edge_indices() {
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();
        println!("{:?} {:?}", e, ew);

        zdg.extend_with_edges(&[
            (
                v,
                w,
                ZeroDemandFlowEdge::new(0, ew.demand, -1., e, ZeroDemandEdgeKind::BelowDemand),
            ),
            (
                v,
                w,
                ZeroDemandFlowEdge::new(
                    0,
                    ew.capacity - ew.demand,
                    0.,
                    e,
                    ZeroDemandEdgeKind::AboveDemand,
                ),
            ),
        ]);
    }
    zdg
}

/// Given the converted zero-demand graph and a flow on it (zd_graph, zd_flow),
/// this calculates the original flow on the original graph.
/// For each edge $e$ in the original graph,
/// there is
/// - type-A edge $ea$
/// - type-B edge $eb$
/// and the sum of the flows of the two is the flow of original edge $e$.
fn zero_demand_flow_to_original_flow(zd_flow: &Flow, zd_graph: &ZeroDemandFlowGraph) -> Flow {
    let mut flow = Flow::empty();
    for e in zd_graph.edge_indices() {
        let ew = zd_graph.edge_weight(e).unwrap();
        let zd_f = zd_flow.get(e).unwrap();
        let original_edge = ew.info.origin;

        if flow.has(original_edge) {
            let f = flow.get(original_edge).unwrap();
            flow.set(original_edge, f + zd_f);
        } else {
            flow.set(original_edge, zd_f);
        }
    }
    flow
}

pub fn find_initial_flow<T: std::fmt::Debug>(graph: &FlowGraphRaw<T>) -> Option<Flow> {
    let zdg = to_zero_demand_graph(graph);
    utils::draw(&zdg);
    let zd_flow = min_cost_flow_from_zero(&zdg);

    println!("sum_of_demand={:?}", sum_of_demand(&graph));
    if zd_flow.total_cost(&zdg) > sum_of_demand(&graph) as f64 {
        // valid flow does not exists
        None
    } else {
        // valid initial flow exists!
        let flow = zero_demand_flow_to_original_flow(&zd_flow, &zdg);
        Some(flow)
    }
}