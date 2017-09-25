use std::fmt;

use error::Error;
use root_system::{CartanMatrix, RootSystem};

/// The \\(B_{n}\\) infinite series of Lie groups.
///
/// The \\(B_{n}\\) (\\(n \geq 2\\)) series is also denoted as
/// \\(\mathrm{SO}(2n+1)\\) and corresponds to the [special orthogonal
/// group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of odd
/// dimension which is the group of \\((2n + 1) \times (2n + 1)\\) orthogonal
/// matrices with determinant 1.
///
/// ## Exceptional Isomorphisms
///
/// - \\(B_{1} \cong A_{1}\\).
#[derive(Debug)]
pub struct TypeB {
    rank: usize,
    cartan_matrix: CartanMatrix,
}

impl TypeB {
    /// Create new Lie group \\(B_{n}\\).
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeB;
    ///
    /// let b3 = TypeB::new(3).unwrap();
    ///
    /// assert_eq!(b3.rank(), 3);
    /// assert_eq!(b3.num_simple_roots(), 3);
    /// assert_eq!(b3.num_positive_roots(), 9);
    /// assert_eq!(b3.num_roots(), 21);
    ///
    /// println!("The roots of {} are:", b3);
    /// for r in &b3.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeB, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            1 => Err(Error::new(
                "B1 is isomorphic to A1.  Please use the latter.",
            )),
            rank => {
                Ok(TypeB {
                    rank,
                    cartan_matrix: Self::cartan_matrix(rank),
                })
            }
        }
    }

    /// Generate the Cartan matrix for the \\(B_{n}\\) group.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 => -1,
            (i, j) if i + 1 == j && j != rank - 1 => -1,
            (i, j) if i + 1 == j && j == rank - 1 => -2,
            _ => 0,
        })
    }
}

impl RootSystem for TypeB {
    fn rank(&self) -> usize {
        self.rank
    }

    fn cartan_matrix(&self) -> &CartanMatrix {
        &self.cartan_matrix
    }

    fn num_roots(&self) -> usize {
        2 * self.rank * self.rank + self.rank
    }

    fn num_positive_roots(&self) -> usize {
        self.rank * self.rank
    }
}

////////////////////////////////////////////////////////////////////////////////
// Trait implementations
////////////////////////////////////////////////////////////////////////////////

impl fmt::Display for TypeB {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("B{}", self.rank))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    #[cfg(feature = "nightly")]
    use test::Bencher;

    use root_system::RootSystem;
    use super::TypeB;

    #[test]
    fn root_system() {
        assert!(TypeB::new(0).is_err());
        assert!(TypeB::new(1).is_err());

        for rank in 2..30 {
            let g = TypeB::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
        }

        let g = TypeB::new(5).unwrap();
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -2],
                [0, 0, 0, -1, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        for rank in (2..10).chain(30..31) {
            let g = TypeB::new(rank).unwrap();
            assert_eq!(g.num_simple_roots(), g.simple_roots().len());
            assert_eq!(g.num_positive_roots(), g.positive_roots().len());
            assert_eq!(g.num_roots(), g.roots().len());
        }
    }

    #[test]
    fn fmt() {
        for rank in (2..10).chain(30..31) {
            let g = TypeB::new(rank).unwrap();
            assert_eq!(format!("{}", g), format!("B{}", rank));
        }
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_10(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeB::new(10).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_50(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeB::new(50).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}