use ndarray::{self, Axis};
use rayon::prelude::*;

use root::Root;

/// Type definition for a Cartan matrix.
///
/// # Definition
///
/// The Cartan matrix of a Lie group is the matrix
///
/// \\begin{equation}
///   A_{ij} = 2 \frac{\langle \alpha_{i}, \alpha_{j} \rangle}{\langle \alpha_{i}, \alpha_{i} \rangle}
/// \\end{equation}
///
/// Where \\(\alpha_{i}\\) are the simple roots of the root system of the Lie
/// group \\(G\\).  As a result, the matrix is of size \\(n \times n\\) where
/// \\(n = \mathop{\rm Rank}(G)\\).  An immediate consequence of the above
/// definition and properties of root systems is that \\(A_{ii} = 2\\) and
/// \\(A_{ij} \leq 0\\) and \\(A_{ij} = 0 \Leftrightarrow A_{ji} = 0\\).
///
/// Since the projection of one root onto another must result in a half-integral
/// multiple of the latter root, it follows that all entries inside the matrix
/// are integers.
pub type CartanMatrix = ndarray::Array2<i64>;

/// Trait for root systems.
pub trait RootSystem {
    /// Return the rank of the Lie group.
    ///
    /// # Definition
    ///
    /// The rank of a Lie group is the dimensionality of the group's Cartan
    /// subgroup.  In the case of groups of matrices, the Cartan subgroup
    /// corresponds to the subgroup of diagonal matrices.
    ///
    /// The rank of the Lie group also corresponds to the number of simple roots
    /// in the Lie group's root system.
    ///
    /// # Example
    ///
    /// The Lie group \\(\mathrm{SU}(3)\\) has rank 2 and the two generators of
    /// the Cartan subgroup are:
    ///
    /// \\begin{equation}
    ///   \left\\{
    ///   \begin{pmatrix} 1 & 0 & 0 \\\\ 0 & -1 & 0 \\\\ 0 & 0 & 0 \end{pmatrix}
    ///   ,
    ///   \frac{1}{\sqrt{3}} \begin{pmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & -2 \end{pmatrix}
    ///   \right\\}
    /// \\end{equation}
    ///
    /// In the case of \\(\mathrm{SU}(3)\\), the simple roots are \\(\\{(2, -1),
    /// (-1, 2)\\}\\) and correspond to the rows of the Cartan matrix of
    /// \\(\mathrm{SU}(3)\\).
    fn rank(&self) -> usize {
        self.cartan_matrix().shape()[0]
    }

    /// Return the Cartan matrix of the Lie group.
    ///
    /// # Definition
    ///
    /// The Cartan matrix of a Lie group is the matrix
    ///
    /// \\begin{equation}
    ///   A_{ij} = 2 \frac{\langle \alpha_{i}, \alpha_{j} \rangle}{\langle \alpha_{i}, \alpha_{i} \rangle}
    /// \\end{equation}
    ///
    /// Where \\(\alpha_{i}\\) are the simple roots of the root system of the Lie
    /// group \\(G\\).  As a result, the matrix is of size \\(n \times n\\) where
    /// \\(n = \mathop{\rm Rank}(G)\\).  An immediate consequence of the above
    /// definition and properties of root systems is that \\(A_{ii} = 2\\) and
    /// \\(A_{ij} \leq 0\\) and \\(A_{ij} = 0 \Leftrightarrow A_{ji} = 0\\).
    ///
    /// # Example
    ///
    /// The Cartan matrix for \\(\mathrm{SU}(3)\\) is:
    ///
    /// \\begin{equation}
    ///   \begin{pmatrix} 2 & -1 \\\\ -1 & 2 \end{pmatrix}
    /// \\end{equation}
    fn cartan_matrix(&self) -> &CartanMatrix;

    /// Return the roots in the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// # Definition
    ///
    /// The roots in the Lie group's root system corresponds to the number of
    /// generators for the Lie group.
    ///
    /// # Example
    ///
    /// The Lie group \\(\mathrm{SU}(3)\\) has 8 generators (one common choice
    /// of generators is the 8 [Gell-Mann
    /// matrices](https://en.wikipedia.org/wiki/Gell-Mann_matrices)), and thus
    /// has 8 roots:
    ///
    /// \\begin{align}
    ///   (1, &~ 1) \\\\
    ///   (-1, 2) &~ (2, -1) \\\\
    ///   (0, 0)  &~ (0, 0) \\\\
    ///   (-2, 1) &~ (1, -2) \\\\
    ///   (-1, &~ {-1})
    /// \\end{align}
    fn roots(&self) -> Vec<Root> {
        let mut roots: Vec<_> = self.positive_roots()
            .iter()
            .map(|r| vec![r * -1, r.clone()])
            .chain((0..self.num_simple_roots()).map(|_| {
                vec![Root::zero(self.rank())]
            }))
            .collect::<Vec<Vec<_>>>()
            .concat();
        roots.sort_unstable();
        roots
    }

    /// Return a number of roots in the Lie group's root system.
    ///
    /// # Warning
    ///
    /// The default implementation of this function will generate all the roots
    /// in the root system, and then return the length of the corresponding
    /// vector.  If there exists a simple closed form for the number of roots,
    /// it should be implemented.
    fn num_roots(&self) -> usize {
        warn!("`num_roots` should be rewritten by the implementation.");
        self.roots().len()
    }
    /// Return the positive roots of the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// # Definition
    ///
    /// The set of positive roots, \\(\Phi\^{+} \subseteq \Phi\\), is the subset
    /// chosen such that:
    ///
    /// - For each \\(\alpha \in \Phi\\), only one of \\(\alpha\\) and
    ///   \\(-\alpha\\) are in \\(\Phi\^{+}\\); and,
    /// - For any two distinct \\(\alpha, \beta \in \Phi\^{+}\\) then
    ///   \\(\alpha+\beta \in \Phi\^{+}\\) provided that \\(\alpha + \beta \in
    ///   \Phi\\).
    ///
    /// # Example
    ///
    /// The positive roots of \\(\mathrm{SU}(3)\\) are:
    ///
    /// \\begin{align}
    ///   (1, &~ 1) \\\\
    ///   (-1, 2) &~ (2, -1) \\\\
    /// \\end{align}
    fn positive_roots(&self) -> Vec<Root> {
        // The idea of the algorithm is that given on root β on level n, we want
        // to see if β + αᵢ is also a root.  Let kᵢ be the number of times the
        // root αᵢ appears in the decomposition of β, then β + αᵢ will be a new
        // root provided that:
        //
        //   m - kⱼ Aⱼᵢ > 0
        //
        // where A is the Cartan matrix.  Note that kⱼ Aⱼᵢ is exactly equal to
        // the ith coefficient of the root in the ω basis.  Does does not seem
        // to be any really efficient way of finding the value of m other than
        // by going through already found roots; however, there are a few
        // heuristics we can do.
        if self.rank() < 24 {
            find_positive_roots_single_thread(&self.simple_roots())
        } else {
            find_positive_roots_multithread(&self.simple_roots())
        }
    }

    /// Return the number of positive roots in the Lie group's root system.
    ///
    /// # Warning
    ///
    /// The default implementation of this function will generate all the
    /// positive roots in the root system, and then return the length of the
    /// corresponding vector.  If there exists a simple closed form for the
    /// number of positive roots, it should be implemented.
    fn num_positive_roots(&self) -> usize {
        warn!("`num_positive_roots` should be rewritten by the implementation.");
        self.positive_roots().len()
    }

    /// Return the simple roots of the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// # Definition
    ///
    /// The set of simple roots, \\(\Delta\\), consists of the subset of
    /// positive roots which cannot be expressed as the sum of other elements in
    /// \\(\Phi\^{+}\\).
    ///
    /// # Example
    ///
    /// The simple roots of \\(\mathrm{SU}(3)\\) are:
    ///
    /// \\begin{align}
    ///   (-1, 2) &~ (2, -1) \\\\
    /// \\end{align}
    fn simple_roots(&self) -> Vec<Root> {
        self.cartan_matrix()
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, w)| Root::simple(w.to_owned(), i))
            .rev()
            .collect()
    }

    /// Return the number of simple roots in the Lie group's root system.
    ///
    /// # Warning
    ///
    /// The default implementation of this function will generate all the simple
    /// roots in the root system, and then return the length of the
    /// corresponding vector.  If there exists a simple closed form for the
    /// number of simple roots, it should be implemented.
    fn num_simple_roots(&self) -> usize {
        self.rank()
    }
}

fn find_positive_roots_multithread(simple_roots: &[Root]) -> Vec<Root> {
    // We can immediately add all the simple roots to the positive roots.
    let mut roots = simple_roots.to_vec();
    // We want to know the index of each simple root.  We *assume* that we have
    // simple roots and don't verify this.
    let simple_roots: Vec<_> = simple_roots
        .iter()
        .map(|r| {
            let i = r.alpha().iter().position(|&k| k == 1).unwrap();
            (i, r)
        })
        .collect();

    // Keep track of the number of roots were in the previous level
    let mut num_prev_level = simple_roots.len();
    while num_prev_level > 0 {
        // Go through each root on the level below,
        let new_roots: Vec<Vec<_>> = roots[roots.len() - num_prev_level..]
            .par_iter()
            .map(|root| {
                // and check if we should add the ith simple root
                simple_roots
                    .par_iter()
                    .filter_map(|&(i, simple_root)| {
                        let m = (1..root.alpha()[i] + 1)
                            .take_while(|&m| roots.binary_search(&(root - simple_root * m)).is_ok())
                            .last();
                        let m = match m {
                            Some(m) => m,
                            None => 0,
                        };

                        if m > root.omega()[i] {
                            Some(root + simple_root)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let mut new_roots: Vec<Root> = new_roots.as_slice().concat();
        new_roots.sort_unstable();
        new_roots.dedup();
        num_prev_level = new_roots.len();
        roots.append(&mut new_roots)
    }

    roots
}

fn find_positive_roots_single_thread(simple_roots: &[Root]) -> Vec<Root> {
    // We can immediately add all the simple roots to the positive roots.
    let mut roots = simple_roots.to_vec();
    // We want to know the index of each simple root.  We *assume* that we have
    // simple roots and don't verify this.
    let simple_roots: Vec<_> = simple_roots
        .iter()
        .map(|r| {
            let i = r.alpha().iter().position(|&k| k == 1).unwrap();
            (i, r)
        })
        .collect();

    // Keep track of the number of roots were in the previous level
    let mut num_prev_level = simple_roots.len();
    while num_prev_level > 0 {
        // Go through each root on the level below,
        let new_roots: Vec<Vec<_>> = roots[roots.len() - num_prev_level..]
            .iter()
            .map(|root| {
                // and check if we should add the ith simple root
                simple_roots
                    .iter()
                    .filter_map(|&(i, simple_root)| {
                        let m = (1..root.alpha()[i] + 1)
                            .take_while(|&m| roots.binary_search(&(root - simple_root * m)).is_ok())
                            .last();
                        let m = match m {
                            Some(m) => m,
                            None => 0,
                        };

                        if m > root.omega()[i] {
                            Some(root + simple_root)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let mut new_roots: Vec<Root> = new_roots.as_slice().concat();
        new_roots.sort_unstable();
        new_roots.dedup();
        num_prev_level = new_roots.len();
        roots.append(&mut new_roots)
    }

    roots
}
