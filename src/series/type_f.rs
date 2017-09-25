use std::fmt;

use error::Error;
use root_system::{CartanMatrix, RootSystem};

/// The \\(F_{n}\\) exceptional Lie groups.
///
/// The only allows value of \\(n\\) is 4.
#[derive(Debug)]
pub struct TypeF {
    rank: usize,
    cartan_matrix: CartanMatrix,
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
    /// for r in &f4.roots() {
    ///     println!("level {} | {}", r.level(), r);
    /// }
    /// ```
    pub fn new(rank: usize) -> Result<TypeF, Error> {
        match rank {
            0 => Err(Error::new("Rank of a Lie group must be at least 1.")),
            rank if rank == 4 => {
                Ok(TypeF {
                    rank,
                    cartan_matrix: array![
                        [2, -1, 0, 0],
                        [-1, 2, -2, 0],
                        [0, -1, 2, -1],
                        [0, 0, -1, 2],
                    ],
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

    fn num_roots(&self) -> usize {
        52
    }

    fn num_positive_roots(&self) -> usize {
        24
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

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_roots_4(b: &mut Bencher) {
        b.iter(|| {
            let g = TypeF::new(4).unwrap();
            assert_eq!(g.num_roots(), g.roots().len());
        });
    }
}
