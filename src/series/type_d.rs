use std::fmt;

use error::Error;
use root::Root;
use root_system::{self, CartanMatrix, RootSystem};

/// The \\(D_{n}\\) infinite series of Lie groups.
///
/// The \\(D_{n}\\) (\\(n \geq 4\\)) series is also denoted as
/// \\(\mathrm{SO}(2n)\\) and corresponds to the [special orthogonal
/// group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of even
/// dimension which is the group of \\(2n \times 2n\\) orthogonal matrices with
/// determinant 1.
///
/// The Cartan matrix for \\(D_{n}\\) is of the form
///
/// \\begin{equation}
///   \begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & -1 \\\\
///     0 & 0 & -1 & 2 & 0 \\\\
///     0 & 0 & -1 & 0 & 2
///   \end{pmatrix}
/// \\end{equation}
///
/// ## Exceptional Isomorphisms
///
/// - (\\(D_{1} \cong A_{1}\\));
/// - \\(D_{2} \cong A_{1} \times A_{1}\\); and,
/// - \\(D_{3} \cong A_{3}\\).
///
/// Since \\(D_{2}\\) is not isomorphic to a simple Lie group, it must be
/// handled separately.
#[derive(Debug)]
pub struct TypeD {
    rank: usize,
    cartan_matrix: CartanMatrix,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
}

impl TypeD {
    /// Create new Lie group \\(D_{n}\\).
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeD;
    ///
    /// let d4 = TypeD::new(4).unwrap();
    ///
    /// assert_eq!(d4.rank(), 4);
    /// assert_eq!(d4.num_simple_roots(), 4);
    /// assert_eq!(d4.num_positive_roots(), 12);
    /// assert_eq!(d4.num_roots(), 28);
    ///
    /// println!("The roots of {} are:", d4);
    /// for r in d4.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeD, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            1 => Err(Error::new(
                "D1 is not well defined though could be said to be isomoprhic to A1.  Please use the latter.",
            )),
            2 => Err(Error::new(
                "D2 is isomorphic to A1 x A1.  Please use the latter.",
            )),
            3 => Err(Error::new(
                "D3 is isomorphic to A3.  Please use the latter.",
            )),
            rank => {
                let cartan_matrix = Self::cartan_matrix(rank);
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = root_system::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeD {
                    rank,
                    cartan_matrix,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
        }
    }

    /// Generate the Cartan matrix for the \\(D_{n}\\) group.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 && i != rank - 1 => -1,
            (i, j) if i + 1 == j && j != rank - 1 => -1,
            (i, j) if i == j + 2 && i == rank - 1 => -1,
            (i, j) if i + 2 == j && j == rank - 1 => -1,
            _ => 0,
        })
    }
}

impl RootSystem for TypeD {
    fn rank(&self) -> usize {
        self.rank
    }

    fn cartan_matrix(&self) -> &CartanMatrix {
        &self.cartan_matrix
    }

    fn simple_roots(&self) -> &[Root] {
        &self.simple_roots
    }

    fn positive_roots(&self) -> &[Root] {
        &self.positive_roots
    }

    fn roots(&self) -> &[Root] {
        &self.roots
    }
}

////////////////////////////////////////////////////////////////////////////////
// Trait implementations
////////////////////////////////////////////////////////////////////////////////

impl fmt::Display for TypeD {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("D{}", self.rank))
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
    use super::TypeD;

    #[test]
    fn root_system() {
        assert!(TypeD::new(0).is_err());
        assert!(TypeD::new(1).is_err());
        assert!(TypeD::new(2).is_err());
        assert!(TypeD::new(3).is_err());

        for rank in 4..30 {
            let g = TypeD::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
        }

        let g = TypeD::new(5).unwrap();
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, -1],
                [0, 0, -1, 2, 0],
                [0, 0, -1, 0, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        for rank in (4..10).chain(30..31) {
            let g = TypeD::new(rank).unwrap();
            assert_eq!(g.num_simple_roots(), g.simple_roots().len());
            assert_eq!(g.num_positive_roots(), g.positive_roots().len());
            assert_eq!(g.num_roots(), g.roots().len());
        }
    }

    #[test]
    fn fmt() {
        for rank in (4..10).chain(30..31) {
            let g = TypeD::new(rank).unwrap();
            assert_eq!(format!("{}", g), format!("D{}", rank));
        }
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_10(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeD::new(10).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_50(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeD::new(50).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
