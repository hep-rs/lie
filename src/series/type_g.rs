use num;

use std::fmt;

use error::Error;
use root::Root;
use root_system::{self, CartanMatrix, RootSystem, BasisLengths};

/// The \\(G_{n}\\) exceptional Lie groups.
///
/// The only allows value of \\(n\\) is 2.
///
/// The Cartan matrix for \\(G_{2}\\) is:
///
/// \\begin{equation}
///   \begin{pmatrix}
///     2 & -1 \\\\
///     -3 & 2
///   \end{pmatrix}
/// \\end{equation}
#[derive(Debug)]
pub struct TypeG {
    rank: usize,
    cartan_matrix: CartanMatrix,
    basis_lengths: BasisLengths,
    simple_roots: Vec<Root>,
    positive_roots: Vec<Root>,
    roots: Vec<Root>,
}

impl TypeG {
    /// Create new Lie group \\(G_{n}\\).
    ///
    /// This function will automatic convert the exceptional isomorphisms to
    /// their corresponding 'standard' label.
    ///
    /// # Examples
    ///
    /// ```
    /// use lie::RootSystem;
    /// use lie::series::TypeG;
    ///
    /// let g2 = TypeG::new(2).unwrap();
    ///
    /// assert_eq!(g2.rank(), 2);
    /// assert_eq!(g2.num_roots(), 14);
    ///
    /// println!("The roots of {} are:", g2);
    /// for r in g2.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<Self, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            rank if rank == 2 => {
                let cartan_matrix = array![[2, -1], [-3, 2]];
                let basis_lengths = array![num::rational::Ratio::new(1, 3), num::One::one()];
                let simple_roots = root_system::find_simple_roots(&cartan_matrix);
                let positive_roots = root_system::find_positive_roots(&simple_roots);
                let roots = root_system::find_roots_from_positive(&positive_roots);
                Ok(TypeG {
                    rank,
                    cartan_matrix,
                    basis_lengths,
                    simple_roots,
                    positive_roots,
                    roots,
                })
            }
            _ => Err(Error::new("Rank of G series groups can only be 2.")),
        }
    }
}

impl RootSystem for TypeG {
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

impl fmt::Display for TypeG {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(&format!("G{}", self.rank))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    #[cfg(feature = "nightly")]
    use test::Bencher;

    use num::One;
    use num::rational::Ratio;
    use super::TypeG;
    use root_system::RootSystem;

    #[test]
    fn root_system() {
        for i in (0..10).filter(|&i| i != 2) {
            assert!(TypeG::new(i).is_err());
        }

        let g = TypeG::new(2).unwrap();
        assert_eq!(g.rank(), 2);
        assert_eq!(g.cartan_matrix().dim(), (2, 2));
        assert_eq!(
            g.cartan_matrix(),
            &array![
                [2, -1],
                [-3, 2],
            ]
        );
    }

    #[test]
    fn roots() {
        let g = TypeG::new(2).unwrap();
        assert_eq!(g.num_simple_roots(), g.simple_roots().len());
        assert_eq!(g.num_positive_roots(), g.positive_roots().len());
        assert_eq!(g.num_roots(), g.roots().len());
    }

    #[test]
    fn basis_lengths() {
        let g = TypeG::new(2).unwrap();
        assert_eq!(g.basis_lengths().len(), g.num_simple_roots());
        assert_eq!(g.basis_lengths(), &array![Ratio::new(1, 3), One::one()]);
    }

    #[test]
    fn fmt() {
        let g = TypeG::new(2).unwrap();
        assert_eq!(format!("{}", g), "G2");
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeG::new(2).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }

}
