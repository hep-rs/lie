use std::fmt;

use error::Error;
use root::Root;
use root_system::{self, CartanMatrix, RootSystem, BasisLengths};

/// The \\(A_{n}\\) infinite series of Lie groups.
///
/// The \\(A_{n}\\) (\\(n \geq 1\\)) series is also denoted as
/// \\(\mathrm{SU}(n+1)\\) and corresponds to [special unitary
/// group](https://en.wikipedia.org/wiki/Special_unitary_group) which is the
/// group of \\(n \times n\\) unitary matrices with determinant 1.
///
/// The Cartan matrix for \\(A_{n}\\) is of the form
///
/// \\begin{equation}
///   \begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & 0 \\\\
///     0 & 0 & -1 & 2 & -1 \\\\
///     0 & 0 & 0 & -1 & 2
///   \end{pmatrix}
/// \\end{equation}
#[derive(Debug)]
pub struct TypeA {
    rank: usize,
    cartan_matrix: CartanMatrix,
    basis_lengths: BasisLengths,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
}

impl TypeA {
    /// Create new Lie group \\(A_{n}\\).
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeA;
    ///
    /// let a3 = TypeA::new(3).unwrap();
    ///
    /// assert_eq!(a3.rank(), 3);
    /// assert_eq!(a3.num_simple_roots(), 3);
    /// assert_eq!(a3.num_positive_roots(), 6);
    /// assert_eq!(a3.num_roots(), 15);
    ///
    /// println!("The roots of {} are:", a3);
    /// for r in a3.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeA, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            rank => {
                let cartan_matrix = Self::cartan_matrix(rank);
                let basis_lengths = Self::basis_lengths(rank);
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = self::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeA {
                    rank,
                    cartan_matrix,
                    basis_lengths,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
        }
    }

    /// Generate the Cartan matrix for the \\(A_{n}\\) group.
    ///
    /// The matrix has \\(2\\) along the main diagonal, \\(-1\\) along the
    /// diagonals directly above and below the main diagonal.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 => -1,
            (i, j) if i + 1 == j => -1,
            _ => 0,
        })
    }

    /// Generate the basis lengths in \\(A_{n}\\).
    ///
    /// For \\(A_{n}\\), all simple roots are of length \\(\sqrt{2}\\).
    fn basis_lengths(rank: usize) -> BasisLengths {
        BasisLengths::from_shape_fn(rank, |_| 2)
    }
}

impl RootSystem for TypeA {
    fn rank(&self) -> usize {
        self.rank
    }

    fn cartan_matrix(&self) -> &CartanMatrix {
        &self.cartan_matrix
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

/// Return the positive roots of the Lie group's root system.
///
/// This implementation is faster than the naïve implementation to find
/// positive roots since, in the \\(\alpha\\) basis, all positive roots take
/// the form (grouped for each level):
///
/// \\begin{align}
///   \begin{pmatrix}
///     1 0 0 0 \\\\
///     0 1 0 0 \\\\
///     0 0 1 0 \\\\
///     0 0 0 1 \\\\
///   \end{pmatrix} &&
///   \begin{pmatrix}
///     1 1 0 0 \\\\
///     0 1 1 0 \\\\
///     0 0 1 1 \\\\
///   \end{pmatrix} &&
///   \begin{pmatrix}
///     1 1 1 0 \\\\
///     0 1 1 1 \\\\
///   \end{pmatrix} &&
///   \begin{pmatrix}
///     1 1 1 1
///   \end{pmatrix}
/// \\end{align}
fn find_positive_roots(simple_roots: &[Root]) -> Vec<Root> {
    let rank = simple_roots[0].rank();
    let roots: Vec<Vec<_>> = (1..rank + 1)
        .map(|level| {
            (0..rank - level + 1)
                .map(|offset| {
                    simple_roots[offset..offset + level].iter().fold(
                        Root::zero(rank),
                        |result, r| {
                            result + r
                        },
                    )
                })
                .collect()
        })
        .collect();

    roots.concat()
}

////////////////////////////////////////////////////////////////////////////////
// Trait implementations
////////////////////////////////////////////////////////////////////////////////

impl fmt::Display for TypeA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("A{}", self.rank))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    #[cfg(feature = "nightly")]
    use test::Bencher;

    use ndarray::Array2;
    use root_system::RootSystem;
    use super::TypeA;

    #[test]
    fn root_system() {
        assert!(TypeA::new(0).is_err());

        for rank in 1..30 {
            let g = TypeA::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
        }

        let g = TypeA::new(5).unwrap();
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -1, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        assert!(TypeA::new(0).is_err());

        for rank in (1..10).chain(30..31) {
            let g = TypeA::new(rank).unwrap();
            assert_eq!(g.num_simple_roots(), g.simple_roots().len());
            assert_eq!(g.num_positive_roots(), g.positive_roots().len());
            assert_eq!(g.num_roots(), g.roots().len());
        }
    }

    #[test]
    fn basis_lengths() {
        let g = TypeA::new(5).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![2, 2, 2, 2, 2]);
    }

    #[test]
    fn inner_product() {
        let g = TypeA::new(5).unwrap();
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
                [0, 0, 0, -1, 2],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());
    }

    #[test]
    fn scalar_product() {
        let g = TypeA::new(5).unwrap();
        let aij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.scalar_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(g.cartan_matrix(), &aij);
    }

    #[test]
    fn fmt() {
        for rank in (1..10).chain(30..31) {
            let g = TypeA::new(rank).unwrap();
            assert_eq!(format!("{}", g), format!("A{}", rank));
        }
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_10(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeA::new(10).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_50(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeA::new(50).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
