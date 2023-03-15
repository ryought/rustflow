//!
//! Flow network definitions for convex cost.
//!
pub mod fast;
use super::base::FlowEdgeRaw;
use super::flow::Flow;
use super::utils::{clamped_log, is_increasing, range};
use super::{Cost, FlowEdge, FlowRateLike};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};

/// Edge of FlowGraph with convex function cost
///
/// * `cost(f)`: cost per unit flow `c(e, f)`
///
pub trait ConvexCost<F: FlowRateLike>: FlowEdge<F> {
    ///
    /// cost function
    /// it is a convex function of the current flow
    fn convex_cost(&self, flow: F) -> Cost;
    ///
    /// cost_diff(f) = cost(f+1) - cost(f)
    ///
    /// This value will be used as weight (fixed cost) of residue edges.
    /// If cost(f+1)-cost(f) is numerically unstable (and analytically solvable)
    /// you should implement manually this function (and override it).
    ///
    fn cost_diff(&self, flow: F) -> Cost {
        self.convex_cost(flow + F::unit()) - self.convex_cost(flow)
    }
    ///
    /// check if this edge has a finite capacity (`capacity < 100`).
    fn is_finite_capacity(&self) -> bool {
        self.capacity() < F::large_const()
    }
    ///
    /// check if the cost function of the edge is actually convex function
    ///
    fn is_convex(&self) -> bool {
        // assert cost_diff is increasing function.
        // (<=> cost function is convex function)
        is_increasing(
            |flow: F| self.cost_diff(flow),
            self.demand(),
            self.capacity(),
        )
    }
}

///
/// Check if cost function of all edges is actually convex.
///
pub fn is_convex_cost_flow_graph<F, N, E>(graph: &DiGraph<N, E>) -> bool
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    graph.edge_indices().all(|e| {
        let ew = graph.edge_weight(e).unwrap();
        ew.is_convex()
    })
}

//
// Base implementations
//

///
/// Edge attributes of ConvexFlowGraph
/// For each edge,
///   demand
///   capacity
///   convex cost function
/// are assigned.
///
#[derive(Debug, Copy, Clone)]
pub struct BaseConvexFlowEdge<F: FlowRateLike> {
    /// demand (lower limit of flow) of the edge l(e)
    pub demand: F,
    /// capacity (upper limit of flow) of the edge u(e)
    pub capacity: F,
    /// cost function
    /// it is a convex function of the current flow
    pub convex_cost_fn: fn(F) -> Cost,
}

impl<F: FlowRateLike> FlowEdge<F> for BaseConvexFlowEdge<F> {
    fn demand(&self) -> F {
        self.demand
    }
    fn capacity(&self) -> F {
        self.capacity
    }
}

impl<F: FlowRateLike> ConvexCost<F> for BaseConvexFlowEdge<F> {
    fn convex_cost(&self, flow: F) -> Cost {
        (self.convex_cost_fn)(flow)
    }
}

impl<F: FlowRateLike> BaseConvexFlowEdge<F> {
    pub fn new(demand: F, capacity: F, convex_cost_fn: fn(F) -> Cost) -> BaseConvexFlowEdge<F> {
        BaseConvexFlowEdge {
            demand,
            capacity,
            convex_cost_fn,
        }
    }
}

/// short version of BaseConvexFlowEdge::new
fn cfe<F: FlowRateLike>(
    demand: F,
    capacity: F,
    convex_cost: fn(F) -> Cost,
) -> BaseConvexFlowEdge<F> {
    BaseConvexFlowEdge::new(demand, capacity, convex_cost)
}

pub type ConvexFlowGraph<F> = DiGraph<(), BaseConvexFlowEdge<F>>;

///
///
///
#[derive(Debug, Copy, Clone)]
pub struct ConvexFlowEdgeInfo<F: FlowRateLike> {
    /// This represents that the edge is created from the origin
    /// in ConvexFlowGraph
    origin: EdgeIndex,
    /// The FixedCostFlowEdge has demand=0 and capacity=1.
    /// if this edge has flow=1, the original edge in ConvexFlowGraph
    /// should have flow=flow_offset+1.
    flow_offset: F,
}

pub type FixedCostFlowEdge<F> = FlowEdgeRaw<F, ConvexFlowEdgeInfo<F>>;

impl<F: FlowRateLike> FixedCostFlowEdge<F> {
    pub fn new(
        demand: F,
        capacity: F,
        cost: Cost,
        origin: EdgeIndex,
        flow_offset: F,
    ) -> FixedCostFlowEdge<F> {
        FixedCostFlowEdge {
            demand,
            capacity,
            cost,
            info: ConvexFlowEdgeInfo {
                origin,
                flow_offset,
            },
        }
    }
}

///
/// A special kind of flow graph, that will be converted from ConvexFlowGraph
/// Each edge has additional information of ConvexFlowEdgeInfo, that will be
/// used when convert it back to flow in the original ConvexFlowGraph.
///
pub type FixedCostFlowGraph<F> = DiGraph<(), FixedCostFlowEdge<F>>;

//
// conversion functions
//

///
/// convert convex flow graph to the (normal, fixed constant cost) flow graph
///
/// create an parallel edges
/// with static cost of `e.cost(f + 1) - e.cost(f)`
///
pub fn to_fixed_flow_graph<F, N, E>(graph: &DiGraph<N, E>) -> Option<FixedCostFlowGraph<F>>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let mut g: FixedCostFlowGraph<F> = FixedCostFlowGraph::new();

    for e in graph.edge_indices() {
        let ew = graph.edge_weight(e).unwrap();
        let (v, w) = graph.edge_endpoints(e).unwrap();

        // assert that
        // (1) flow is actually convex
        if !ew.is_finite_capacity() {
            return None;
        }
        // (2) capacity is finite
        if !ew.is_convex() {
            return None;
        }

        // convert to the FixedFlowGraph
        // (1) if demand d > 0, add an edge of [demand,capacity]=[d,d]
        if ew.demand() > F::zero() {
            let demand = ew.demand();
            g.extend_with_edges(&[(
                v,
                w,
                FixedCostFlowEdge::new(demand, demand, 0.0, e, F::zero()),
            )]);
        }
        // (2) add aux edges of [demand,capacity]=[0,1]
        let edges: Vec<(NodeIndex, NodeIndex, FixedCostFlowEdge<F>)> =
            range(ew.demand(), ew.capacity())
                .into_iter()
                .map(|f| {
                    // set a (constant) cost to cost(f+1)-cost(f)
                    let cost = ew.cost_diff(f);
                    let fe = FixedCostFlowEdge::new(F::zero(), F::unit(), cost, e, f);
                    (v, w, fe)
                })
                .collect();
        g.extend_with_edges(&edges);
    }

    Some(g)
}

fn get_flow_in_fixed<F: FlowRateLike>(
    edge: EdgeIndex,
    fixed_flow: &Flow<F>,
    fixed_graph: &FixedCostFlowGraph<F>,
) -> F {
    // for original edge e (with EdgeIndex edge) in ConvexFlowGraph
    // the flow is the sum of the flow on the edges whose FixedCostFlowEdge.info.origin == edge
    fixed_graph
        .edge_indices()
        .filter_map(|fe| {
            let fe_weight = fixed_graph.edge_weight(fe).unwrap();
            if fe_weight.info.origin == edge {
                let flow = fixed_flow[fe];
                Some(flow)
            } else {
                None
            }
        })
        .sum()
}

pub fn restore_convex_flow<F, N, E>(
    fixed_flow: &Flow<F>,
    fixed_graph: &FixedCostFlowGraph<F>,
    graph: &DiGraph<N, E>,
) -> Flow<F>
where
    F: FlowRateLike,
    E: FlowEdge<F> + ConvexCost<F>,
{
    let mut flow = Flow::new(graph.edge_count(), F::zero());

    for e in graph.edge_indices() {
        let f = get_flow_in_fixed(e, fixed_flow, fixed_graph);
        flow[e] = f;
    }

    flow
}

//
// utils
//
#[allow(dead_code)]
fn mock_convex_flow_graph1() -> (ConvexFlowGraph<usize>, Flow<usize>) {
    let mut g = ConvexFlowGraph::new();
    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());
    let e0 = g.add_edge(a, b, cfe(0, 10, |f| (f as f64 - 5.0).powi(2)));
    let e1 = g.add_edge(b, c, cfe(0, 10, |f| (f as f64 - 5.0).powi(2)));
    let e2 = g.add_edge(c, a, cfe(0, 10, |f| (f as f64 - 5.0).powi(2)));

    let f = vec![5, 5, 5].into();
    (g, f)
}

#[allow(dead_code)]
fn mock_convex_flow_graph2() -> (ConvexFlowGraph<usize>, Flow<usize>) {
    let mut g = ConvexFlowGraph::new();
    let s = g.add_node(());
    let v1 = g.add_node(());
    let v2 = g.add_node(());
    let w1 = g.add_node(());
    let w2 = g.add_node(());
    let t = g.add_node(());
    // surrounding
    let e0 = g.add_edge(s, v1, cfe(2, 2, |_| 0.0));
    let e1 = g.add_edge(s, v2, cfe(4, 4, |_| 0.0));
    let e2 = g.add_edge(w1, t, cfe(1, 1, |_| 0.0));
    let e3 = g.add_edge(w2, t, cfe(5, 5, |_| 0.0));
    let e4 = g.add_edge(t, s, cfe(6, 6, |_| 0.0));

    // intersecting
    let e5 = g.add_edge(v1, w1, cfe(0, 6, |f| -10.0 * clamped_log(f)));
    let e6 = g.add_edge(v1, w2, cfe(0, 6, |f| -10.0 * clamped_log(f)));
    let e7 = g.add_edge(v2, w1, cfe(0, 6, |f| -10.0 * clamped_log(f)));
    let e8 = g.add_edge(v2, w2, cfe(0, 6, |f| -10.0 * clamped_log(f)));

    // true flow
    let f = vec![2, 4, 1, 5, 6, 0, 2, 1, 3].into();
    (g, f)
}

#[allow(dead_code)]
fn mock_convex_flow_graph3() -> ConvexFlowGraph<usize> {
    let mut g = ConvexFlowGraph::new();
    g.extend_with_edges(&[
        (0, 1, cfe(0, 10, |_| 0.0)),
        (1, 2, cfe(0, 10, |_| 0.0)),
        (2, 0, cfe(0, 10, |_| 0.0)),
        (1, 3, cfe(0, 10, |_| 0.0)),
        (3, 4, cfe(0, 10, |_| 0.0)),
        (4, 2, cfe(0, 10, |_| 0.0)),
    ]);
    g
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::super::utils::{draw, draw_with_flow};
    use super::super::{
        enumerate_neighboring_flows, min_cost_flow, min_cost_flow_convex,
        min_cost_flow_convex_fast, EdgeCost,
    };
    use super::*;

    #[test]
    fn convex_flow_edge_new() {
        let e = BaseConvexFlowEdge::new(1, 5, |f| 10.0 * f as f64);
        assert_eq!(50.0, e.convex_cost(5));
        assert_eq!(50.0, e.cost(5));
        assert_eq!(1, e.demand);
        assert_eq!(5, e.capacity);
        assert_eq!(true, e.is_convex());

        // TODO
        // let e = BaseConvexFlowEdge::new(1, 5, |f| 0.0001 * f as f64);
        // assert_eq!(true, e.is_convex());
    }

    #[test]
    fn convex_flow_graph_mock1() {
        let (g, f_true) = mock_convex_flow_graph1();
        let fg = to_fixed_flow_graph(&g).unwrap();
        let fg_flow = min_cost_flow(&fg).unwrap();
        let flow = restore_convex_flow(&fg_flow, &fg, &g);

        println!("{:?}", flow);
        println!("{:?}", f_true);
        assert!(flow == f_true);
    }

    #[test]
    fn convex_flow_graph_mock2() {
        let (g, f_true) = mock_convex_flow_graph2();
        draw(&g);
        let flow = min_cost_flow_convex(&g).unwrap();
        draw_with_flow(&g, &flow);
        println!("{:?}", flow);
        assert!(flow == f_true);
    }

    #[test]
    fn convex_flow_graph_mock1_fast() {
        let (g, f_true) = mock_convex_flow_graph1();
        let f = min_cost_flow_convex_fast(&g);
        println!("{:?}", f);
        println!("{:?}", f_true);
        assert!(f.is_some());
        assert!(f.unwrap() == f_true);
    }

    #[test]
    fn convex_flow_graph_mock2_fast() {
        let (g, f_true) = mock_convex_flow_graph2();
        let f = min_cost_flow_convex_fast(&g);
        println!("{:?}", f);
        println!("{:?}", f_true);
        assert!(f.is_some());
        assert!(f.unwrap() == f_true);
    }

    #[test]
    fn enumerate_graph3() {
        let g = mock_convex_flow_graph3();

        {
            let f0 = vec![0, 0, 0, 0, 0, 0].into();
            println!("f0={}", f0);
            let fs = enumerate_neighboring_flows(&g, &f0, Some(10), None);
            for (f, ui) in fs.iter() {
                println!("f={} ui={:?}", f, ui);
            }
            assert_eq!(fs.len(), 2);
            assert_eq!(fs[0].0, vec![1, 1, 1, 0, 0, 0].into());
            assert_eq!(fs[1].0, vec![1, 0, 1, 1, 1, 1].into());

            let fs = enumerate_neighboring_flows(&g, &f0, Some(2), None);
            assert_eq!(fs.len(), 0);

            let fs = enumerate_neighboring_flows(&g, &f0, Some(3), None);
            assert_eq!(fs.len(), 1);

            let fs = enumerate_neighboring_flows(&g, &f0, Some(5), None);
            assert_eq!(fs.len(), 2);

            let fs = enumerate_neighboring_flows(&g, &f0, Some(5), Some(0));
            assert_eq!(fs.len(), 2); // n_flip of all neighbors are 0
        }

        {
            let f0 = vec![2, 1, 2, 1, 1, 1].into();
            println!("f0={}", f0);
            let fs = enumerate_neighboring_flows(&g, &f0, Some(10), None);
            for (f, ui) in fs.iter() {
                println!("f={} ui={:?}", f, ui);
            }
            assert_eq!(fs.len(), 6);
            assert_eq!(fs[0].0, vec![1, 0, 1, 1, 1, 1].into());
            assert_eq!(fs[1].0, vec![3, 2, 3, 1, 1, 1].into());
            assert_eq!(fs[2].0, vec![2, 0, 2, 2, 2, 2].into());
            assert_eq!(fs[3].0, vec![2, 2, 2, 0, 0, 0].into());
            assert_eq!(fs[4].0, vec![1, 1, 1, 0, 0, 0].into());
            assert_eq!(fs[5].0, vec![3, 1, 3, 2, 2, 2].into());

            let fs = enumerate_neighboring_flows(&g, &f0, Some(5), Some(0));
            for (f, ui) in fs.iter() {
                println!("fa={} ui={:?}", f, ui);
            }
            assert_eq!(fs.len(), 4);
        }
    }
}
