use ndarray::Array2;
use std::fmt;

use crate::error::Error;
use crate::root::Root;
use crate::root_system::{self, BasisLengths, CartanMatrix, InverseCartanMatrix, RootSystem};

/// The \\(B_{n}\\) infinite series of Lie groups.
///
/// The \\(B_{n}\\) (\\(n \geq 2\\)) series is also denoted as
/// \\(\mathrm{SO}(2n+1)\\) and corresponds to the [special orthogonal
/// group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of odd
/// dimension which is the group of \\((2n + 1) \times (2n + 1)\\) orthogonal
/// matrices with determinant 1.
///
/// The Cartan matrix for \\(B_{n}\\) is of the form
///
/// \\begin{equation}
///   \begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & 0 \\\\
///     0 & 0 & -1 & 2 & -2 \\\\
///     0 & 0 & 0 & -1 & 2
///   \end{pmatrix}
/// \\end{equation}
///
/// ## Exceptional Isomorphisms
///
/// - \\(B_{1} \cong A_{1}\\).
#[derive(Debug)]
pub struct TypeB {
    rank: usize,
    cartan_matrix: CartanMatrix,
    inverse_cartan_matrix: InverseCartanMatrix,
    basis_lengths: BasisLengths,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
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
    /// for r in b3.roots() {
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
                let cartan_matrix = Self::cartan_matrix(rank);
                let inverse_cartan_matrix = Self::inverse_cartan_matrix(rank);
                let basis_lengths = Self::basis_lengths(rank);
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = root_system::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeB {
                    rank,
                    cartan_matrix,
                    inverse_cartan_matrix,
                    basis_lengths,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
        }
    }

    /// Generate the Cartan matrix for the \\(B_{n}\\) group.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        let mut m = CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 => -1,
            (i, j) if i + 1 == j => -1,
            _ => 0,
        });
        m[[rank - 2, rank - 1]] = -2;
        m
    }

    /// Generate the inverse Cartan matrix for the \\(B_{n}\\) group.
    fn inverse_cartan_matrix(rank: usize) -> InverseCartanMatrix {
        let m = Array2::from_shape_fn((rank, rank), |indices| {
            let v = match indices {
                // The last row increments by one
                (i, j) if i == rank - 1 => j + 1,
                (i, j) if i <= j => 2 * (i + 1),
                (i, j) if i > j => 2 * (j + 1),
                _ => unreachable!(),
            };
            v as i64
        });
        (m, 2)
    }

    /// Generate the basis lengths in \\(B_{n}\\).
    ///
    /// For \\(B_{n}\\), the first \\(n - 1\\) simple roots have lengths
    /// \\(\sqrt{2}\\), the last one is unit length.
    fn basis_lengths(rank: usize) -> BasisLengths {
        BasisLengths::from_shape_fn(rank, |i| if i < rank - 1 { 2 } else { 1 })
    }
}

impl RootSystem for TypeB {
    fn rank(&self) -> usize {
        self.rank
    }

    fn cartan_matrix(&self) -> &CartanMatrix {
        &self.cartan_matrix
    }

    fn inverse_cartan_matrix(&self) -> &InverseCartanMatrix {
        &self.inverse_cartan_matrix
    }

    fn basis_lengths(&self) -> &BasisLengths {
        &self.basis_lengths
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
    use super::*;

    use crate::root_system::RootSystem;
    use ndarray::Array2;
    #[cfg(feature = "nightly")]
    use test::Bencher;

    #[test]
    fn root_system() {
        assert!(TypeB::new(0).is_err());
        assert!(TypeB::new(1).is_err());

        for rank in 2..30 {
            let g = TypeB::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
            assert_eq!(g.determinant(), 2);

            let cm = g.cartan_matrix();
            let &(ref icm, d) = g.inverse_cartan_matrix();
            assert_eq!(cm.dot(icm), icm.dot(cm));
            assert_eq!(cm.dot(icm) / d, Array2::eye(rank));
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
    fn basis_lengths() {
        let g = TypeB::new(5).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![2, 2, 2, 2, 1]);
    }

    #[test]
    fn inner_product() {
        let g = TypeB::new(5).unwrap();
        let sij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.inner_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });

        assert_eq!(
            sij,
            array![
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -1, 1],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());
    }

    #[test]
    fn scalar_product() {
        let g = TypeB::new(5).unwrap();
        let aij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.scalar_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(g.cartan_matrix(), &aij);
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
