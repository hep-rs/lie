use std::fmt;

use error::Error;
use root_system::{CartanMatrix, RootSystem};

/// The \\(G_{n}\\) exceptional Lie groups.
///
/// The only allows value of \\(n\\) is 2.
#[derive(Debug)]
pub struct TypeG {
    rank: usize,
    cartan_matrix: CartanMatrix,
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
    /// for r in &g2.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<Self, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            rank if rank == 2 => {
                Ok(TypeG {
                    rank,
                    cartan_matrix: array![[2, -1], [-3, 2]],
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

    fn num_roots(&self) -> usize {
        14
    }

    fn num_positive_roots(&self) -> usize {
        6
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
