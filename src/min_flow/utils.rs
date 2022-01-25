//!
//! utils
//!
use super::mocks;
use super::{find_initial_flow, min_cost_flow};
use petgraph::dot::Dot;
use petgraph::graph::Graph;
use petgraph::EdgeType;

///
/// check if the function `f` is convex or not
/// in the domain `[x_min, x_max]`
///
/// it will check `f(x + 1) - f(x)` is monotonically decreasing
/// for increasing `x`
///
pub fn is_convex(f: fn(u32) -> f64, x_min: u32, x_max: u32) -> bool {
    let mut y_prev = f64::MIN;

    (x_min..x_max).map(|x| f(x + 1) - f(x)).all(|y| {
        // check if ys are decresing
        let is_decreasing = y > y_prev;
        y_prev = y;
        is_decreasing
    })
}

pub fn test() {
    let g = mocks::mock_flow_network2();
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

// pub fn draw_with_flow
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_convex_test() {
        assert!(is_convex(|x| (x as f64 - 10.0).powi(2), 0, 20));
        assert!(!is_convex(|x| -(x as f64 - 10.0).powi(2), 0, 20));
    }
}
