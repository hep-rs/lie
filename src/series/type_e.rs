use std::fmt;

use crate::error::Error;
use crate::root::Root;
use crate::root_system::{self, BasisLengths, CartanMatrix, InverseCartanMatrix, RootSystem};

/// The \\(E_{n}\\) exceptional Lie groups.
///
/// This is only defined for \\(n = 6, 7, 8\\). though through the exceptional
///
/// The three Cartan matrix for \\(n = 6, 7, 8\\) respectively are:
///
/// \\begin{align}
///   &\begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & 0 & -1 \\\\
///     0 & 0 & -1 & 2 & -1 & 0 \\\\
///     0 & 0 & 0 & -1 & 2 & 0 \\\\
///     0 & 0 & -1 & 0 & 0 & 2
///   \end{pmatrix} \\\\
///   &\begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & 0 & 0 & -1 \\\\
///     0 & 0 & -1 & 2 & -1 & 0 & 0 \\\\
///     0 & 0 & 0 & -1 & 2 & -1 & 0 \\\\
///     0 & 0 & 0 & 0 & -1 & 2 & 0 \\\\
///     0 & 0 & -1 & 0 & 0 & 0 & 2
///   \end{pmatrix} \\\\
///   &\begin{pmatrix}
///     2 & -1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
///     -1 & 2 & -1 & 0 & 0 & 0 & 0 & 0 \\\\
///     0 & -1 & 2 & -1 & 0 & 0 & 0 & -1 \\\\
///     0 & 0 & -1 & 2 & -1 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & -1 & 2 & -1 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & -1 & 2 & -1 & 0 \\\\
///     0 & 0 & 0 & 0 & 0 & -1 & 2 & 0 \\\\
///     0 & 0 & -1 & 0 & 0 & 0 & 0 & 2
///   \end{pmatrix} \\\\
/// \\end{align}
///
/// ## Exceptional Isomorphisms
///
/// - \\(E_{3} \cong A_{1} \times A_{2}\\);
/// - \\(E_{4} \cong A_{4}\\); and,
/// - \\(E_{5} \cong D_{5}\\).
///
/// Since \\(E_{3}\\) is not isomorphic to a simple Lie group, it must be
/// handled separately.
#[derive(Debug)]
pub struct TypeE {
    rank: usize,
    cartan_matrix: CartanMatrix,
    inverse_cartan_matrix: InverseCartanMatrix,
    basis_lengths: BasisLengths,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
}

impl TypeE {
    /// Create new Lie group \\(E_{n}\\).
    ///
    /// This function will automatic convert the exceptional isomorphisms to
    /// their corresponding 'standard' label.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeE;
    ///
    /// let e6 = TypeE::new(6).unwrap();
    ///
    /// assert_eq!(e6.rank(), 6);
    /// assert_eq!(e6.num_simple_roots(), 6);
    /// assert_eq!(e6.num_positive_roots(), 36);
    /// assert_eq!(e6.num_roots(), 78);
    ///
    /// println!("The roots of {} are:", e6);
    /// for r in e6.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeE, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            1 => Err(Error::new("E1 is not well defined.")),
            2 => Err(Error::new("E2 is not well defined.")),
            3 => Err(Error::new(
                "E3 is isomorphic to A1 x A2.  Please use the latter.",
            )),
            4 => Err(Error::new(
                "E4 is isomorphic to A4.  Please use the latter.",
            )),
            5 => Err(Error::new(
                "E5 is isomorphic to D5.  Please use the latter.",
            )),
            rank if rank == 6 || rank == 7 || rank == 8 => {
                let cartan_matrix = Self::cartan_matrix(rank);
                let inverse_cartan_matrix = Self::inverse_cartan_matrix(rank);
                let basis_lengths = Self::basis_lengths(rank);
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = root_system::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeE {
                    rank,
                    cartan_matrix,
                    inverse_cartan_matrix,
                    basis_lengths,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
            _ => Err(Error::new(
                "Rank of E series groups is only defined for 6, 7 and 8.",
            )),
        }
    }

    /// Generate the Cartan matrix for the \\(E_{n}\\) group.
    fn cartan_matrix(rank: usize) -> CartanMatrix {
        let mut m = CartanMatrix::from_shape_fn((rank, rank), |indices| match indices {
            (i, j) if i == j => 2,
            (i, j) if i == j + 1 && i != rank - 1 => -1,
            (i, j) if i + 1 == j && j != rank - 1 => -1,
            _ => 0,
        });
        m[[2, rank - 1]] = -1;
        m[[rank - 1, 2]] = -1;
        m
    }

    /// Generate the inverse Cartan matrix for the \\(E_{n}\\) group.
    fn inverse_cartan_matrix(rank: usize) -> InverseCartanMatrix {
        match rank {
            6 => (
                array![
                    [4, 5, 6, 4, 2, 3],
                    [5, 10, 12, 8, 4, 6],
                    [6, 12, 18, 12, 6, 9],
                    [4, 8, 12, 10, 5, 6],
                    [2, 4, 6, 5, 4, 3],
                    [3, 6, 9, 6, 3, 6]
                ],
                3,
            ),
            7 => (
                array![
                    [4, 6, 8, 6, 4, 2, 4],
                    [6, 12, 16, 12, 8, 4, 8],
                    [8, 16, 24, 18, 12, 6, 12],
                    [6, 12, 18, 15, 10, 5, 9],
                    [4, 8, 12, 10, 8, 4, 6],
                    [2, 4, 6, 5, 4, 3, 3],
                    [4, 8, 12, 9, 6, 3, 7]
                ],
                2,
            ),
            8 => (
                array![
                    [4, 7, 10, 8, 6, 4, 2, 5],
                    [7, 14, 20, 16, 12, 8, 4, 10],
                    [10, 20, 30, 24, 18, 12, 6, 15],
                    [8, 16, 24, 20, 15, 10, 5, 12],
                    [6, 12, 18, 15, 12, 8, 4, 9],
                    [4, 8, 12, 10, 8, 6, 3, 6],
                    [2, 4, 6, 5, 4, 3, 2, 3],
                    [5, 10, 15, 12, 9, 6, 3, 8]
                ],
                1,
            ),
            _ => unreachable!(),
        }
    }

    /// Generate the basis lengths in \\(E_{n}\\).
    ///
    /// For \\(E_{n}\\), all simple roots have length \\(\sqrt{2}\\).
    fn basis_lengths(rank: usize) -> BasisLengths {
        BasisLengths::from_shape_fn(rank, |_| 2)
    }
}

impl RootSystem for TypeE {
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

impl fmt::Display for TypeE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("E{}", self.rank))
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
        for rank in (0..6).chain(9..15) {
            assert!(TypeE::new(rank).is_err())
        }

        for rank in 6..9 {
            let g = TypeE::new(rank).unwrap();
            assert_eq!(g.rank(), rank);
            assert_eq!(g.cartan_matrix().dim(), (rank, rank));
            assert_eq!(g.determinant(), 9 - rank as i64);

            let cm = g.cartan_matrix();
            let &(ref icm, d) = g.inverse_cartan_matrix();
            assert_eq!(cm.dot(icm), icm.dot(cm));
            assert_eq!(cm.dot(icm) / d, Array2::eye(rank));
        }

        assert_eq!(
            TypeE::new(6).unwrap().cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0],
                [0, -1, 2, -1, 0, -1],
                [0, 0, -1, 2, -1, 0],
                [0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 2],
            ]
        );

        assert_eq!(
            TypeE::new(7).unwrap().cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, -1],
                [0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 0, 2],
            ]
        );

        assert_eq!(
            TypeE::new(8).unwrap().cartan_matrix(),
            &array![
                [2, -1, 0, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, 0, -1],
                [0, 0, -1, 2, -1, 0, 0, 0],
                [0, 0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 0, 0, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        for rank in 6..9 {
            let g = TypeE::new(rank).unwrap();
            assert_eq!(g.num_simple_roots(), g.simple_roots().len());
            assert_eq!(g.num_positive_roots(), g.positive_roots().len());
            assert_eq!(g.num_roots(), g.roots().len());
        }
    }

    #[test]
    fn basis_lengths() {
        let g = TypeE::new(6).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![2, 2, 2, 2, 2, 2]);

        let g = TypeE::new(7).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![2, 2, 2, 2, 2, 2, 2]);

        let g = TypeE::new(8).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![2, 2, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn inner_product() {
        let g = TypeE::new(6).unwrap();
        let sij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.inner_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(
            sij,
            array![
                [2, -1, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0],
                [0, -1, 2, -1, 0, -1],
                [0, 0, -1, 2, -1, 0],
                [0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 2],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());

        let g = TypeE::new(7).unwrap();
        let sij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.inner_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(
            sij,
            array![
                [2, -1, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, -1],
                [0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 0, 2],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());

        let g = TypeE::new(8).unwrap();
        let sij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.inner_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(
            sij,
            array![
                [2, -1, 0, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, 0, -1],
                [0, 0, -1, 2, -1, 0, 0, 0],
                [0, 0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0, -1, 2, 0],
                [0, 0, -1, 0, 0, 0, 0, 2],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());
    }

    #[test]
    fn scalar_product() {
        let g = TypeE::new(6).unwrap();
        let aij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.scalar_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(g.cartan_matrix(), &aij);

        let g = TypeE::new(7).unwrap();
        let aij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.scalar_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(g.cartan_matrix(), &aij);

        let g = TypeE::new(8).unwrap();
        let aij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.scalar_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(g.cartan_matrix(), &aij);
    }

    #[test]
    fn fmt() {
        for rank in 6..9 {
            let g = TypeE::new(rank).unwrap();
            assert_eq!(format!("{}", g), format!("E{}", rank));
        }
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_6(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeE::new(6).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_7(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeE::new(7).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_8(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeE::new(8).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
