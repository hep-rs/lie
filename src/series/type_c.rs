use std::fmt;

use error::Error;
use root_system::{CartanMatrix, RootSystem};

/// The \\(C_{n}\\) infinite series of Lie groups.
///
/// The \\(C_{n}\\) (\\(n \geq 3\\)) series is also denoted as
/// \\(\mathrm{Sp}(2n)\\) and corresponds to the group of [unitary symplectic
/// matrices](https://en.wikipedia.org/wiki/Symplectic_matrix).
///
/// Note that there are two conventions in the literature whereby \\(C_{n} =
/// \mathrm{Sp}(2n)\\) or \\(C_{n} = \mathrm{Sp}(n)\\).
///
/// ## Exceptional Isomorphisms
///
/// - \\(C_{1} \cong A_{1}\\); and,
/// - \\(C_{2} \cong B_{1}\\).
#[derive(Debug)]
pub struct TypeC {
    rank: usize,
    cartan_matrix: CartanMatrix,
}

impl TypeC {
    /// Create new Lie group \\(C_{n}\\).
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeC;
    ///
    /// let c3 = TypeC::new(3).unwrap();
    ///
    /// assert_eq!(c3.rank(), 3);
    /// assert_eq!(c3.num_simple_roots(), 3);
    /// assert_eq!(c3.num_positive_roots(), 9);
    /// assert_eq!(c3.num_roots(), 21);
    ///
    /// println!("The roots of {} are:", c3);
    /// for r in &c3.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeC, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            1 => Err(Error::new(
                "C1 is isomorphic to A1.  Please use the latter.",
            )),
            2 => Err(Error::new(
                "C2 is isomorphic to B1.  Please use the latter.",
            )),
            rank => {
                Ok(TypeC {
                    rank,
                    cartan_matrix: Self::cartan_matrix(rank),
                })
            }
        }
    }

    /// Generate the Cartan matrix for the \\(C_{n}\\) group.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 && i != rank - 1 => -1,
            (i, j) if i == j + 1 && i == rank - 1 => -2,
            (i, j) if i + 1 == j => -1,
            _ => 0,
        })
    }
}

impl RootSystem for TypeC {
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

impl fmt::Display for TypeC {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("C{}", self.rank))
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
    use super::TypeC;

    #[test]
    fn root_system() {
        assert!(TypeC::new(0).is_err());
        assert!(TypeC::new(1).is_err());
        assert!(TypeC::new(2).is_err());

        for rank in 3..30 {
            let g = TypeC::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
        }

        let g = TypeC::new(5).unwrap();
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -2, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        for rank in 3..10 {
            let g = TypeC::new(rank).unwrap();
            assert_eq!(g.num_simple_roots(), g.simple_roots().len());
            assert_eq!(g.num_positive_roots(), g.positive_roots().len());
            assert_eq!(g.num_roots(), g.roots().len());
        }
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_10(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeC::new(10).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_50(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeC::new(50).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
