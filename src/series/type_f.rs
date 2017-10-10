use std::fmt;

use error::Error;
use root::Root;
use root_system::{self, CartanMatrix, RootSystem, BasisLengths};

/// The \\(F_{n}\\) exceptional Lie groups.
///
/// The only allows value of \\(n\\) is 4.
///
/// The Cartan matrix for \\(F_{4}\\) is:
///
/// \\begin{equation}
///   \begin{pmatrix}
///     2 & -1 & 0 & 0 \\\\
///     -1 & 2 & -2 & 0 \\\\
///     0 & -1 & 2 & -1 \\\\
///     0 & 0 & -1 & 2
///   \end{pmatrix}
/// \\end{equation}
#[derive(Debug)]
pub struct TypeF {
    rank: usize,
    cartan_matrix: CartanMatrix,
    basis_lengths: BasisLengths,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
}

impl TypeF {
    /// Create new Lie group \\(F_{n}\\).
    ///
    /// This function will automatic convert the exceptional isomorphisms to
    /// their corresponding 'standard' label.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeF;
    ///
    /// let f4 = TypeF::new(4).unwrap();
    ///
    /// assert_eq!(f4.rank(), 4);
    /// assert_eq!(f4.num_simple_roots(), 4);
    /// assert_eq!(f4.num_positive_roots(), 24);
    /// assert_eq!(f4.num_roots(), 52);
    ///
    /// println!("The roots of {} are:", f4);
    /// for r in f4.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeF, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            rank if rank == 4 => {
                let cartan_matrix =
                    array![
                    [2, -1, 0, 0],
                    [-1, 2, -2, 0],
                    [0, -1, 2, -1],
                    [0, 0, -1, 2],
                ];
                let basis_lengths = array![4, 4, 2, 2];
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = root_system::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeF {
                    rank,
                    cartan_matrix,
                    basis_lengths,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
            _ => Err(Error::new("Rank of F series groups is only defined for 4.")),
        }
    }
}

impl RootSystem for TypeF {
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

////////////////////////////////////////////////////////////////////////////////
// Trait implementations
////////////////////////////////////////////////////////////////////////////////

impl fmt::Display for TypeF {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("F{}", self.rank))
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
    use super::TypeF;

    #[test]
    fn root_system() {
        for rank in (0..10).filter(|&i| i != 4) {
            assert!(TypeF::new(rank).is_err())
        }

        let g = TypeF::new(4).unwrap();
        assert_eq!(g.rank(), 4);
        assert_eq!(g.cartan_matrix().dim(), (4, 4));
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1, 0, 0],
                [-1, 2, -2, 0],
                [0, -1, 2, -1],
                [0, 0, -1, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        let g = TypeF::new(4).unwrap();
        assert_eq!(g.num_simple_roots(), g.simple_roots().len());
        assert_eq!(g.num_positive_roots(), g.positive_roots().len());
        assert_eq!(g.num_roots(), g.roots().len());
    }

    #[test]
    fn basis_lengths() {
        let g = TypeF::new(4).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![4, 4, 2, 2]);
    }

    #[test]
    fn inner_product() {
        let g = TypeF::new(4).unwrap();
        let sij = Array2::from_shape_fn((g.rank(), g.rank()), |(i, j)| {
            g.inner_product(&g.simple_roots()[i], &g.simple_roots()[j])
        });
        assert_eq!(
            sij,
            array![
                [4, -2, 0, 0],
                [-2, 4, -2, 0],
                [0, -2, 2, -1],
                [0, 0, -1, 2],
            ]
        );
        assert_eq!(&sij.diag(), g.basis_lengths());
    }

    #[test]
    fn fmt() {
        let g = TypeF::new(4).unwrap();
        assert_eq!(format!("{}", g), "F4");
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_4(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeF::new(4).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
