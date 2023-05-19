//! Residue graph related definitions
//! - ResidueEdge
//! - ResidueGraph
//! - ResidueDirection
//!
use super::convex::ConvexCost;
use super::flow::{EdgeCost, Flow};
use super::utils::draw;
use super::{ConstCost, Cost, FlowEdge, FlowRateLike};
use itertools::Itertools; // for tuple_windows
use petgraph::algo::astar;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::visit::VisitMap;
use petgraph_algos::common::{
    edge_cycle_to_node_cycle, is_cycle, is_edge_simple, is_negative_cycle, node_list_to_edge_list,
    remove_all_parallel_edges, total_weight, FloatWeight,
};
use petgraph_algos::cycle_enumeration::{simple_cycles, simple_k_cycles_with_cond};
use petgraph_algos::min_mean_weight_cycle::edge_cond::find_negative_cycle_with_edge_cond;
use petgraph_algos::my_bellman_ford;
use std::cmp::Ordering;
use std::num::Saturating;

// basic definitions

/// Edge attributes used in ResidueGraph
#[derive(Debug, Default, Copy, Clone)]
pub struct ResidueEdge<F: FlowRateLike> {
    /// The movable amount of the flow
    pub count: F,
    /// Cost of the unit change of this flow
    pub weight: Cost,
    /// Original edge index of the source graph
    pub target: EdgeIndex,
    /// +1 or -1
    pub direction: ResidueDirection,
}

impl<F: FlowRateLike> ResidueEdge<F> {
    pub fn new(
        count: F,
        weight: Cost,
        target: EdgeIndex,
        direction: ResidueDirection,
    ) -> ResidueEdge<F> {
        ResidueEdge {
            count,
            weight,
            target,
            direction,
        }
    }
}

impl<F: FlowRateLike> FloatWeight for ResidueEdge<F> {
    fn float_weight(&self) -> f64 {
        self.weight
    }
    fn epsilon() -> f64 {
        0.00001
    }
}

/// Residue direction enum
/// residue edge has two types
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ResidueDirection {
    /// Up edge: it can increase(+1) flow
    Up,
    /// Down edge: it can decrease(-1) flow
    Down,
}

impl ResidueDirection {
    /// Map ResidueDirection into i32
    /// * Up   -> +1
    /// * Down -> -1
    pub fn int(&self) -> i32 {
        match *self {
            ResidueDirection::Up => 1,
            ResidueDirection::Down => -1,
        }
    }
}

impl std::fmt::Display for ResidueDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ResidueDirection::Up => write!(f, "+"),
            ResidueDirection::Down => write!(f, "-"),
        }
    }
}

impl std::str::FromStr for ResidueDirection {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(ResidueDirection::Up),
            "-" => Ok(ResidueDirection::Down),
            _ => Err(()),
        }
    }
}

///
/// List of ResidueDirection (n-times "+" and m-times "-") into total changes (n-m)
///
pub fn total_changes<I>(directions: I) -> i32
where
    I: Iterator<Item = ResidueDirection>,
{
    directions.map(|dir| dir.int()).sum()
}

impl Default for ResidueDirection {
    fn default() -> Self {
        ResidueDirection::Up
    }
}

/// ResidueGraph definition
pub type ResidueGraph<F> = DiGraph<(), ResidueEdge<F>>;

//
// conversion functions
//

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
pub fn flow_to_residue<F: FlowRateLike, N, E: FlowEdge<F> + ConstCost>(
    graph: &DiGraph<N, E>,
    flow: &Flow<F>,
) -> ResidueGraph<F> {
    assert_eq!(
        flow.len(),
        graph.edge_count(),
        "flow (len={}) does not match network (E={})",
        flow.len(),
        graph.edge_count()
    );

    let mut rg: ResidueGraph<F> = ResidueGraph::new();

    // create two edges (Up and Down) for each edge
    for e in graph.edge_indices() {
        let f = flow[e];
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();

        let mut edges = Vec::new();
        if f < ew.capacity() {
            // up movable
            edges.push((
                v,
                w,
                ResidueEdge::new(ew.capacity() - f, ew.cost(), e, ResidueDirection::Up),
            ));
        }
        if f > ew.demand() {
            // down movable
            edges.push((
                w,
                v,
                ResidueEdge::new(f - ew.demand(), -ew.cost(), e, ResidueDirection::Down),
            ));
        }
        rg.extend_with_edges(&edges);
    }
    rg
}

/// Convert FlowGraph with Flow with ConvexCost into ResidueGraph.
///
/// For each edge in FlowGraph with Flow
/// ```text
/// e(v -> w) = ([l,u],c), f
/// ```
///
/// create two edges in ResidueGraph
/// ```text
/// e1(v -> w) = (1, c(f+1) - c(f)) if u - f > 0
///
/// e2(w -> v) = (1, c(f-1) - c(f)) if f - l > 0
/// ```
pub fn flow_to_residue_convex<F, N, E>(graph: &DiGraph<N, E>, flow: &Flow<F>) -> ResidueGraph<F>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    assert_eq!(
        flow.len(),
        graph.edge_count(),
        "flow (len={}) does not match network (E={})",
        flow.len(),
        graph.edge_count()
    );

    let mut rg: ResidueGraph<F> = ResidueGraph::new();

    // create two edges (Up and Down) for each edge
    for e in graph.edge_indices() {
        let f = flow[e];
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();

        let mut edges = Vec::new();
        if f < ew.capacity() {
            // up movable
            // cost=cost(f+1)-cost(f)
            edges.push((
                v,
                w,
                ResidueEdge::new(F::unit(), ew.cost_diff(f), e, ResidueDirection::Up),
            ));
        }
        if f > ew.demand() {
            // down movable
            // cost=cost(f-1)-cost(f)
            edges.push((
                w,
                v,
                ResidueEdge::new(
                    F::unit(),
                    -ew.cost_diff(f - F::unit()),
                    e,
                    ResidueDirection::Down,
                ),
            ));
        }

        // if up/down movable,
        // self round loop (v->w and w->v) should not have negative weight.
        if f < ew.capacity() && f > ew.demand() {
            let cost_up = ew.cost_diff(f);
            let cost_down = -ew.cost_diff(f - F::unit());

            // TODO this assertion is valid only if the cost function is convex.
            // assert!(cost_up + cost_down >= 0.0);
        }

        rg.extend_with_edges(&edges);
    }
    rg
}

#[allow(dead_code)]
fn residue_to_float_weighted_graph<F: FlowRateLike>(graph: &ResidueGraph<F>) -> DiGraph<(), Cost> {
    graph.map(|_, _| (), |_, ew| ew.weight)
}

//
// internal functions to find a update of the flow
// (i.e. the negative cycle in ResidueGraph)
//

///
/// Change EdgeVec by `amount` along the edges of a cycle in residue graph
///
pub fn change_flow_along_edges<F: FlowRateLike>(
    flow: &Flow<F>,
    rg: &ResidueGraph<F>,
    edges: &[EdgeIndex],
    amount: F,
) -> Flow<F> {
    let mut new_flow = flow.clone();
    for edge in edges {
        let ew = rg.edge_weight(*edge).unwrap();
        // convert back to the original edgeindex
        let original_edge = ew.target;

        // use `wrapping_{add,sub}` because
        // in the some ordering of residue edges, applying -1 on a zero-flow edge can happen.
        // As long as the residue edges is valid (i.e. it makes cycle in the residue graph)
        // the final flow should satisfy the flow condition.
        new_flow[original_edge] = match ew.direction {
            ResidueDirection::Up => new_flow[original_edge].wrapping_add(amount),
            ResidueDirection::Down => new_flow[original_edge].wrapping_sub(amount),
        };
    }
    new_flow
}

///
/// Update the flow by a negative cycle on a residue graph.
///
fn apply_residual_edges_to_flow<F: FlowRateLike>(
    flow: &Flow<F>,
    rg: &ResidueGraph<F>,
    edges: &[EdgeIndex],
) -> Flow<F> {
    // (1) determine flow_change_amount
    // that is the minimum of ResidueEdge.count
    let flow_change_amount = edges
        .iter()
        .map(|&e| {
            let ew = rg.edge_weight(e).unwrap();
            ew.count
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // (2) apply these changes to the flow along the cycle
    change_flow_along_edges(flow, rg, edges, flow_change_amount)
}

///
///
///
pub fn residue_graph_cycle_to_flow<F: FlowRateLike>(
    flow: &Flow<F>,
    residue_graph: &ResidueGraph<F>,
    cycle_in_residue_graph: &[EdgeIndex],
) -> (Flow<F>, UpdateInfo) {
    (
        apply_residual_edges_to_flow(flow, residue_graph, cycle_in_residue_graph),
        cycle_in_residue_graph_into_update_info(residue_graph, cycle_in_residue_graph),
    )
}

///
/// Simple heuristic to avoid searching meaningless cycles on ResidueGraph.
///
/// Prohibiting move below:
///
/// ```text
///    +e
/// v ---> w
///   <---
///    -e
/// ```
///
pub fn is_meaningful_move_on_residue_graph<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    e_a: EdgeIndex,
    e_b: EdgeIndex,
) -> bool {
    let ew_a = rg.edge_weight(e_a).unwrap();
    let ew_b = rg.edge_weight(e_b).unwrap();
    let target_is_different = ew_a.target != ew_b.target;
    let dir_is_same = ew_a.direction == ew_b.direction;
    target_is_different || dir_is_same
}

///
/// `U`: ResidueDirection::Up, `D`: ResidueDirection::Down
///
/// | edges  | edge | n_flip |
/// | ------ | ---- | ------ |
/// |        | D    | 0      |
/// |        | U    | 0      |
/// | UUU    | U    | 0      |
/// | UUU    | D    | 1      |
/// | UUUDDD | D    | 1      |
/// | UUUDDD | U    | 2      |
/// | UUUDDD | D    | 1      |
/// | UUDDUU | U    | 2      |
/// | UUDDUU | D    | 3      |
///
pub fn is_within_max_flip<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    edges: &[EdgeIndex],
    edge: EdgeIndex,
    max_flip: Option<usize>,
) -> bool {
    if edges.is_empty() {
        return true;
    }
    match max_flip {
        None => true,
        Some(max_flip) => {
            let mut n_flip = 0;
            for i in 0..edges.len() {
                // if direction of edges[i-1]/edge[i] is different, increment n_flip
                let d0 = rg[edges[i]].direction;
                let d1 = if i == edges.len() - 1 {
                    rg[edge].direction
                } else {
                    rg[edges[i + 1]].direction
                };
                if d0 != d1 {
                    n_flip += 1;
                }
            }

            if n_flip <= max_flip {
                // current n_flip is below the max_flip
                true
            } else {
                false
            }
        }
    }
}

///
///
///
pub fn flow_diff_to_residue<F: FlowRateLike, N, E: FlowEdge<F> + ConstCost>(
    graph: &DiGraph<N, E>,
    flow_from: &Flow<F>,
    flow_to: &Flow<F>,
) -> ResidueGraph<F> {
    unimplemented!();
}

///
/// list up all neighboring flows
///
/// * max_cycle_size: maximum number of edges in a cycle
/// * max_flip: maximum number of flips (Down->Up or Up->Down) in a cycle
///
pub fn enumerate_neighboring_flows_in_residue<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    flow: &Flow<F>,
    max_cycle_size: Option<usize>,
    max_flip: Option<usize>,
) -> Vec<(Flow<F>, UpdateInfo)> {
    // println!("{:?}", petgraph::dot::Dot::with_config(&rg, &[]));
    let simple_cycles = match max_cycle_size {
        Some(k) => simple_k_cycles_with_cond(rg, k, |edges, edge| {
            if edges.len() == 0 {
                true
            } else {
                let last_edge = edges.last().copied().unwrap();
                is_meaningful_move_on_residue_graph(&rg, last_edge, edge)
                    && is_within_max_flip(&rg, edges, edge, max_flip)
            }
        }),
        // TODO Johnson algorithm does not support parallel edges
        // this cause problem when with compacted edbg
        None => simple_cycles(rg),
    };
    // for cycle in simple_cycles.iter() {
    //     println!("cycle = {}", cycle);
    // }
    // eprintln!("# n_simple_cycles={}", simple_cycles.len());
    let flows: Vec<_> = simple_cycles
        .into_iter()
        .map(|cycle| residue_graph_cycle_to_flow(flow, rg, cycle.edges()))
        .filter(|(new_flow, _update_info)| new_flow != flow)
        .collect();
    // eprintln!("# n_flows={}", flows.len());
    flows
}

///
/// Convert a path as nodes into a path as edges
///
/// Assume path is not a circular, so path[n-1] and path[0] will not be connected.
///
pub fn node_path_to_edge_path<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    node_path: &[NodeIndex],
    direction: ResidueDirection,
) -> Vec<EdgeIndex> {
    let mut edge_path = Vec::new();
    let n = node_path.len();

    // convert (nodes[i], nodes[i+1]) into an edge
    for i in 0..(n - 1) {
        let v = node_path[i];
        let w = node_path[i + 1];
        let e = rg
            .edges_connecting(v, w)
            .find(|e| e.weight().direction == direction)
            .map(|e| e.id())
            .expect("node path is invalid, no edge between p[i] and p[i+1]");
        edge_path.push(e);
    }

    edge_path
}

///
/// Find a neighboring flow by finding the minimum cycle of a single direction passing through
/// the `(edge, direction)` edge in the residual network using A* algorithm and update the flow accordingly.
///
/// weight `W: Fn(EdgeIndex) -> usize` is (non-negative integer) weight function of edges.
///
pub fn find_neighboring_flow_by_edge_change_in_residue<F, W>(
    rg: &ResidueGraph<F>,
    flow: &Flow<F>,
    edge: EdgeIndex,
    direction: ResidueDirection,
    weight: W,
) -> Option<(Flow<F>, UpdateInfo)>
where
    F: FlowRateLike,
    W: Fn(EdgeIndex) -> usize,
{
    // find the corresponding edge (edge, direction) in residue graph
    let e = rg
        .edge_references()
        .find(|e| e.weight().target == edge && e.weight().direction == direction)
        .map(|e| e.id());

    match e {
        Some(e) => {
            // find the minimum cycle passing through e = (v, w)
            //
            let (v, w) = rg.edge_endpoints(e).unwrap();
            //
            // find the shortest path from w to v
            //
            let path = astar(
                rg,
                w,
                |finish| finish == v,
                |e| {
                    if e.weight().direction == direction {
                        Saturating(weight(e.weight().target))
                    } else {
                        Saturating(usize::MAX)
                    }
                },
                |_| Saturating(0),
            )
            .map(|(_, nodes)| node_path_to_edge_path(rg, &nodes, direction));

            match path {
                Some(mut edges) => {
                    // join e (from node v into node w) and path (from node w into node v)
                    edges.insert(0, e);
                    assert!(is_cycle(&rg, &edges));
                    assert!(is_edge_simple(&rg, &edges));

                    // convert the cycle into flow and updateinfo
                    Some(residue_graph_cycle_to_flow(flow, rg, &edges))
                }
                None => None,
            }
        }
        None => None,
    }
}

#[derive(Clone, Debug, Copy)]
pub enum CycleDetectMethod {
    BellmanFord,
    MinMeanWeightCycle,
}

fn find_negative_cycle_in_whole_graph<F: FlowRateLike>(
    graph: &ResidueGraph<F>,
    method: CycleDetectMethod,
) -> Option<Vec<EdgeIndex>> {
    let mut node = NodeIndex::new(0);
    let mut dfs = Dfs::new(&graph, node);

    loop {
        // find negative cycle with prohibiting e1 -> e2 transition
        //
        //    e1 (+1 of e)
        // v --->
        //   <--- w
        //    e2 (-1 of e)
        //
        let path = match method {
            CycleDetectMethod::MinMeanWeightCycle => {
                find_negative_cycle_with_edge_cond(&graph, node, |e_a, e_b| {
                    is_meaningful_move_on_residue_graph(&graph, e_a, e_b)
                })
            }
            CycleDetectMethod::BellmanFord => find_negative_cycle_as_edges(&graph, node),
        };

        if path.is_some() {
            return path;
        }

        // search for alternative start point
        dfs.move_to(node);
        while let Some(_nx) = dfs.next(&graph) {}
        let unvisited_node = graph
            .node_indices()
            .find(|node| !dfs.discovered.is_visited(node));

        // if there is unvisited node, search again for negative cycle
        match unvisited_node {
            Some(n) => {
                node = n;
                continue;
            }
            None => break,
        }
    }
    return None;
}

///
/// Find a negative cycle by using Bellman ford algorithm.
/// Return cycle as edge list, not node list
///
pub fn find_negative_cycle_as_edges<N, E>(
    graph: &DiGraph<N, E>,
    source: NodeIndex,
) -> Option<Vec<EdgeIndex>>
where
    E: FloatWeight,
{
    my_bellman_ford::find_negative_cycle(graph, source)
        .map(|nodes| node_list_to_edge_list(graph, &nodes))
}

fn format_cycle<F: FlowRateLike>(rg: &ResidueGraph<F>, cycle: &[EdgeIndex]) -> String {
    cycle
        .iter()
        .map(|&edge| {
            let weight = rg.edge_weight(edge).unwrap();
            format!(
                "e{}({},{})w{}",
                edge.index(),
                weight.target.index(),
                weight.direction,
                weight.weight
            )
        })
        .join(",")
    // format!("{:?}", cycle)
}

///
/// Update residue graph by finding negative cycle
///
pub fn improve_residue_graph<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    method: CycleDetectMethod,
) -> Option<Vec<EdgeIndex>> {
    // find negative weight cycles
    draw(&rg);
    let path = find_negative_cycle_in_whole_graph(&rg, method);

    match path {
        Some(edges) => {
            // println!("improving.. {}", format_cycle(rg, &edges));
            // check if this is actually negative cycle
            assert!(
                is_cycle(&rg, &edges),
                "the cycle was not valid. edges={:?}",
                edges
            );
            assert!(
                is_edge_simple(&rg, &edges),
                "the cycle is not edge-simple. edges={:?}",
                edges
            );
            assert!(
                is_negative_cycle(&rg, &edges),
                "total weight of the found negative cycle is not negative. edges={:?} total_weight={}",
                edges,
                total_weight(&rg, &edges)
            );

            // TODO assert using is_meaningful_cycle?

            Some(edges)
        }
        None => None,
    }
}

///
/// WIP
///
fn is_meaningful_cycle<F: FlowRateLike>(rg: &ResidueGraph<F>, cycle: &[EdgeIndex]) -> bool {
    unimplemented!();
}

/// create a new improved flow from current flow
/// by upgrading along the negative weight cycle in the residual graph
fn update_flow_in_residue_graph<F: FlowRateLike>(
    flow: &Flow<F>,
    rg: &ResidueGraph<F>,
    method: CycleDetectMethod,
) -> Option<(Flow<F>, Vec<EdgeIndex>)> {
    match improve_residue_graph(rg, method) {
        Some(cycle) => {
            // apply these changes along the cycle to current flow
            let new_flow = apply_residual_edges_to_flow(&flow, &rg, &cycle);

            // if applying edges did not changed the flow (i.e. the edges was meaningless)
            // improve should fail.
            if &new_flow == flow {
                println!("meaningless cycle was found!");
                None
            } else {
                Some((new_flow, cycle))
            }
        }
        None => None,
    }
}

fn cycle_in_residue_graph_into_update_info<F: FlowRateLike>(
    rg: &ResidueGraph<F>,
    cycle: &[EdgeIndex],
) -> UpdateInfo {
    cycle
        .iter()
        .map(|&e| {
            let ew = rg.edge_weight(e).unwrap();
            (ew.target, ew.direction)
        })
        .collect()
}

//
// public functions
//

/// create a new improved flow from current flow
/// by upgrading along the negative weight cycle in the residual graph
pub fn improve_flow<F: FlowRateLike, N, E: FlowEdge<F> + ConstCost>(
    graph: &DiGraph<N, E>,
    flow: &Flow<F>,
    method: CycleDetectMethod,
) -> Option<Flow<F>> {
    let mut rg = flow_to_residue(graph, flow);
    remove_all_parallel_edges(&mut rg);
    match update_flow_in_residue_graph(flow, &rg, method) {
        Some((new_flow, _)) => Some(new_flow),
        None => None,
    }
}

///
/// `UpdateInfo` = `Vec<(EdgeIndex, ResidueDirection)>`
///
/// information of updating a edge of either direction?
///
pub type UpdateInfo = Vec<(EdgeIndex, ResidueDirection)>;

// ///
// /// summary of UpdateInfo
// ///
// pub type UpdateSummary = Vec<(Vec<EdgeIndex>, ResidueDirection)>;
//
// ///
// /// Convert a update cycle
// ///     [(1, +), (2, +), (3, +), (3, -), (2, -), (1, -)]
// ///     [(2, +), (3, +), (3, -), (2, -), (1, -), (1, +)]
// /// into a normalized summary
// ///     [([1,2,3], +), ([3,2,1], -)]
// ///
// fn to_contiguous_direction_list(
//     updates: &[(EdgeIndex, ResidueDirection)],
// ) -> Vec<(Vec<EdgeIndex>, ResidueDirection)> {
//     unimplemented!();
// }

/// create a new improved flow from current flow
/// by upgrading along the negative weight cycle in the residual graph
pub fn improve_flow_convex_with_update_info<F, N, E>(
    graph: &DiGraph<N, E>,
    flow: &Flow<F>,
    method: CycleDetectMethod,
) -> Option<(Flow<F>, UpdateInfo)>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let mut rg = flow_to_residue_convex(graph, flow);
    remove_all_parallel_edges(&mut rg);
    match update_flow_in_residue_graph(flow, &rg, method) {
        Some((new_flow, cycle)) => Some((
            new_flow,
            cycle_in_residue_graph_into_update_info(&rg, &cycle),
        )),
        None => None,
    }
}

/// create a new improved flow from current flow
/// by upgrading along the negative weight cycle in the residual graph
pub fn improve_flow_convex<F, N, E>(
    graph: &DiGraph<N, E>,
    flow: &Flow<F>,
    method: CycleDetectMethod,
) -> Option<Flow<F>>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let mut rg = flow_to_residue_convex(graph, flow);
    remove_all_parallel_edges(&mut rg);
    match update_flow_in_residue_graph(flow, &rg, method) {
        Some((new_flow, _)) => Some(new_flow),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph_algos::common::ei;
    use petgraph_algos::min_mean_weight_cycle::find_negative_cycle;

    #[test]
    fn residue_direction_basic() {
        let up = ResidueDirection::Up;
        assert_eq!(up.to_string(), "+");
        assert_eq!(up.int(), 1);
        assert_eq!("+".parse::<ResidueDirection>().unwrap(), up);

        let down = ResidueDirection::Down;
        assert_eq!(down.to_string(), "-");
        assert_eq!(down.int(), -1);
        assert_eq!("-".parse::<ResidueDirection>().unwrap(), down);

        let directions = vec![
            ResidueDirection::Down,
            ResidueDirection::Down,
            ResidueDirection::Down,
            ResidueDirection::Up,
        ];
        assert_eq!(total_changes(directions.into_iter()), -2);
    }

    #[test]
    fn petgraph_negative_cycle_test() {
        // small cycle test
        let mut g: DiGraph<(), f64> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        g.add_edge(a, b, -10.0);
        g.add_edge(b, a, 9.0);
        let path = find_negative_cycle(&g, NodeIndex::new(0));
        assert_eq!(path.is_some(), true);
        let nodes = path.unwrap();
        assert!(nodes.contains(&NodeIndex::new(0)));
        assert!(nodes.contains(&NodeIndex::new(1)));
    }

    #[test]
    fn petgraph_negative_cycle_test2() {
        // self loop test, it will work fine
        let mut g: DiGraph<(), f64> = Graph::new();
        let a = g.add_node(());
        g.add_edge(a, a, -10.0);
        let path = find_negative_cycle(&g, NodeIndex::new(0));
        assert_eq!(path.is_some(), true);
        let nodes = path.unwrap();
        assert!(nodes.contains(&NodeIndex::new(0)));
    }

    #[test]
    fn negative_cycle_in_whole() {
        let mut g = ResidueGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(
            a,
            b,
            ResidueEdge::new(1, 10.0, EdgeIndex::new(0), ResidueDirection::Up),
        );
        g.add_edge(
            b,
            a,
            ResidueEdge::new(1, -1.0, EdgeIndex::new(1), ResidueDirection::Up),
        );
        g.add_edge(
            c,
            c,
            ResidueEdge::new(1, -1.0, EdgeIndex::new(2), ResidueDirection::Up),
        );
        {
            let path =
                find_negative_cycle_in_whole_graph(&g, CycleDetectMethod::MinMeanWeightCycle);
            assert_eq!(path.is_some(), true);
            assert_eq!(path, Some(vec![ei(2)]));
        }

        {
            let path = find_negative_cycle_in_whole_graph(&g, CycleDetectMethod::BellmanFord);
            assert_eq!(path.is_some(), true);
            assert_eq!(path, Some(vec![ei(2)]));
        }
    }

    fn re(
        count: usize,
        weight: f64,
        target: EdgeIndex,
        direction: ResidueDirection,
    ) -> ResidueEdge<usize> {
        ResidueEdge::new(count, weight, target, direction)
    }

    #[test]
    fn negative_cycle_complex_case() {
        let up = ResidueDirection::Up;
        let down = ResidueDirection::Down;
        let mut rg = ResidueGraph::from_edges(&[
            (0, 17, re(1, 1.0, ei(43), up)),
            (17, 0, re(1, 1.0, ei(43), down)),
            (17, 3, re(1, 1.0, ei(24), up)),
            (3, 17, re(1, 1.0, ei(24), down)),
            (3, 2, re(1, 1.0, ei(28), up)),
            (2, 3, re(1, 1.0, ei(28), down)),
            (2, 31, re(1, 113.0, ei(26), up)),
            (31, 2, re(1, 138.0, ei(26), down)),
            (31, 7, re(1, 5.0, ei(25), up)),
            (7, 31, re(1, 50.0, ei(25), down)),
            (7, 12, re(1, 0.0, ei(34), up)),
            (12, 7, re(1, 1.0, ei(34), down)),
            (12, 10, re(1, 1.0, ei(42), up)),
            (10, 12, re(1, 54.0, ei(42), down)),
            (10, 26, re(1, 1165.0, ei(45), up)),
            (26, 10, re(1, 712.0, ei(45), down)),
            (26, 30, re(1, 163.0, ei(8), up)),
            (30, 26, re(1, 74.0, ei(8), down)),
            (30, 16, re(1, 604.0, ei(11), up)),
            (16, 30, re(1, 221.0, ei(11), down)),
            (16, 14, re(1, 581.0, ei(6), up)),
            (14, 16, re(1, -171.0, ei(6), down)),
            (14, 25, re(1, -35.0, ei(30), up)),
            (25, 14, re(1, -139.0, ei(32), down)),
            (25, 32, re(1, 0.0, ei(20), up)),
            (32, 25, re(1, -0.0, ei(20), down)),
        ]);
        remove_all_parallel_edges(&mut rg);
        println!("{:?}", petgraph::dot::Dot::with_config(&rg, &[]));
        let c = find_negative_cycle_in_whole_graph(&rg, CycleDetectMethod::BellmanFord);
        println!("c={:?}", c);
    }
}
