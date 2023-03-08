//!
//! utils
//!
use super::flow::Flow;
use super::mocks;
use super::{find_initial_flow, min_cost_flow};
use super::{Cost, FlowRateLike};
use petgraph::dot::Dot;
use petgraph::graph::{DiGraph, Graph};
use petgraph::EdgeType;

///
/// check if the function `f` is convex or not
/// in the domain `[x_min, x_max]`
///
/// it will check `f(x + 1) - f(x)` is monotonically increasing
/// for increasing `x`s
///
pub fn is_convex<X: FlowRateLike, F: Fn(X) -> f64>(f: F, x_min: X, x_max: X) -> bool {
    is_increasing(|x| f(x + X::unit()) - f(x), x_min, x_max)
}

///
/// check if the function `f` is monotonically increasing in the domain `[x_min, x_max]`.
///
pub fn is_increasing<X: FlowRateLike, F: Fn(X) -> f64>(f: F, x_min: X, x_max: X) -> bool {
    let mut y_prev = f64::MIN;

    range(x_min, x_max).into_iter().map(|x| f(x)).all(|y| {
        // check if ys are increasing
        let is_increasing = y >= y_prev;
        y_prev = y;
        is_increasing
    })
}

///
/// return a (collected) vector of `x_min..x_max`.
///
pub fn range<X: FlowRateLike>(x_min: X, x_max: X) -> Vec<X> {
    assert!(x_min <= x_max);
    let mut ret = Vec::new();
    let mut x = x_min;

    while x < x_max {
        ret.push(x);
        x = x + X::unit();
    }

    ret
}

///
/// Avoid `ln(0) = -\infty`
/// by setting ln(0) = (default constant -1000.0)
///
pub fn clamped_log(x: usize) -> f64 {
    clamped_log_with(x, DEFAULT_CLAMP_VALUE)
}

pub const DEFAULT_CLAMP_VALUE: f64 = -1000.0;

///
/// Avoid `ln(0) = -\infty`
/// by setting `ln(0)` = (some constant `c < 0`)
///
pub fn clamped_log_with(x: usize, c: f64) -> f64 {
    if x == 0 {
        c
    } else {
        (x as f64).ln()
    }
}

pub fn test() {
    let (g, _) = mocks::mock_flow_network2();
    draw(&g);

    let f = find_initial_flow(&g);
    println!("initia_flow={:?}", f);

    let f = min_cost_flow(&g);
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

#[derive(Debug, Copy, Clone)]
struct EdgeWithFlow<T, F: FlowRateLike> {
    flow: F,
    info: T,
}

pub fn draw_with_flow<N, E, F>(graph: &DiGraph<N, E>, flow: &Flow<F>)
where
    F: FlowRateLike + std::fmt::Debug,
    N: std::fmt::Debug,
    E: std::fmt::Debug,
{
    let graph_with_flow = graph.map(
        |_, vw| vw,
        |e, ew| EdgeWithFlow {
            flow: flow[e],
            info: ew,
        },
    );
    println!("{:?}", Dot::with_config(&graph_with_flow, &[]));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_convex_test() {
        // f(x) = (x-10)^2 is convex
        assert!(is_convex(|x| (x as f64 - 10.0).powi(2), 0, 20));
        // f(x) = - (x-10)^2 is not convex
        assert!(!is_convex(|x| -(x as f64 - 10.0).powi(2), 0, 20));
        // f(x) = 0 (constant)
        assert!(is_convex(|_| 0.0, 0, 20));
        // f(x) = -c log(x)
        assert!(is_convex(|x| -10.0 * (x as f64).ln(), 1, 20));
        // f(x) = -c clamped_log(x)
        assert!(is_convex(|x| -10.0 * clamped_log(x), 0, 20));
        // f(x) = 10x
        assert!(is_convex(|x| 10.0 * x as f64, 0, 20));
        // f(x) = -10x
        assert!(is_convex(|x| -10.0 * x as f64, 0, 20));
        // f(x) = 0x
        assert!(is_convex(|x| 0.0 * x as f64, 0, 20));
        // TODO f(x) = 0.0001x
        // because of floating point calculation
        // some function (with small number) fails this convexity test.
        // assert!(is_convex(|x| 0.0001 * x as f64, 0, 20));
        // but by analytically solving f(x+1)-f(x) it passes the test.
        assert!(is_increasing(|_| 0.0001, 0, 20));
    }
    #[test]
    fn range_of_flowratelike() {
        let xs = range(0_usize, 10_usize);
        assert_eq!(xs, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let xs = range(0_usize, 1_usize);
        assert_eq!(xs, vec![0]);
        let xs = range(0_usize, 0_usize);
        assert_eq!(xs.len(), 0);

        let ys: Vec<usize> = (0..10).collect();
        println!("{:?}", ys);
        let ys: Vec<usize> = (0..0).collect();
        println!("{:?}", ys);

        // TODO assert for f64
    }
}
