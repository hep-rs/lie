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
///
/// ## Definition
///
/// Let \\(V\\) be a finite dimensional Euclidean vector space with the standard
/// Euclidean inner product being denoted by \\(\langle \cdot, \cdot \rangle\\).
/// A *root system* in \\(V\\) is a finite set \\(\Phi\\) of non-zero vectors
/// (called *roots*) that satisfy the following conditions:
///
/// 1. The roots span \\(V\\);
///
/// 2. The only scalar multiple of a root \\(\alpha \in \Phi\\) that belong to
///    \\(\Phi\\) are \\(\alpha\\) and \\(-\alpha\\);
///
/// 3. For every root \\(\alpha \in \Phi\\), the set \\(\Phi\\) is closed under
///    reflections through the hyperplane perpendicular to \\(\alpha\\);
///
/// 4. If \\(\alpha, \beta \in \Phi\\), then the projection of
///    \\(\beta\\) onto \\(\alpha\\) is a half-integral multiple of
///    \\(\alpha\\).
///
/// In addition to the set of roots themselves, there are two important subsets
/// of \\(\Phi\\) which are commonly used:
///
/// - \\(\Phi\^{+} \subseteq \Phi\\): the set of positive roots.  This is the
///   subset chosen such that:
///   - For each \\(\alpha \in \Phi\\), only one of \\(\alpha\\) and
///     \\(-\alpha\\) are in \\(\Phi\^{+}\\); and,
///   - For any two distinct \\(\alpha, \beta \in \Phi\^{+}\\) then \\(\alpha +
///     \beta \in \Phi\^{+}\\) provided that \\(\alpha + \beta \in \Phi\\).
/// - \\(\Delta \subseteq \Phi\^{+}\\): the set of simple root.  An element of
///   \\(\Phi\^{+}\\) is a simple root if it cannot be expressed as the sum of
///   two other elements in \\(\Phi\^{+}\\).  The set of simple roots has the
///   property that every root in \\(\Phi\\) can be expressed as linear
///   combination of elements of \\(\Delta\\) with all coefficients
///   non-negative, or all coefficients non-positive.
///
/// Although the roots are, strictly speaking, elements of a Euclidean vector
/// space; we will not be handling these vectors directly.  Instead, all the
/// relevant information is contained in the [Cartan
/// matrix](type.CartanMatrix.html).
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
    /// # Example
    ///
    /// The Cartan matrix for \\(\mathrm{SU}(3)\\) is:
    ///
    /// \\begin{equation}
    ///   \begin{pmatrix} 2 & -1 \\\\ -1 & 2 \end{pmatrix}
    /// \\end{equation}
    fn cartan_matrix(&self) -> &CartanMatrix;

    /// Return the simple roots of the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// This can be algorithmically created using the `find_simple_roots`
    /// function.
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
    fn simple_roots(&self) -> &[Root];

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

    /// Return the positive roots of the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// This can be algorithmically created using the `find_positive_roots`
    /// function.
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
    fn positive_roots(&self) -> &[Root];

    /// Return the number of positive roots in the Lie group's root system.
    fn num_positive_roots(&self) -> usize {
        self.positive_roots().len()
    }

    /// Return the roots in the Lie group's root system.
    ///
    /// The roots are returned sorted as per the root's ordering scheme.
    ///
    /// This can be algorithmically created using the `find_roots_from_simple`
    /// or `find_roots_from_positive` functions.
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
    fn roots(&self) -> &[Root];

    /// Return a number of roots in the Lie group's root system.
    fn num_roots(&self) -> usize {
        self.roots().len()
    }
}

/// Find all the simple roots from the Cartan matrix.
///
/// # Example
///
/// ```
/// use lie::{Root, CartanMatrix};
/// use lie::root_system::find_simple_roots;
///
/// let cm = CartanMatrix::from(vec![[2, -1], [-3, 2]]);
/// let simple_roots = find_simple_roots(&cm);
/// assert_eq!(simple_roots.len(), 2);
/// ```
///
/// # Panics
///
/// Panics if the Cartan matrix is not square, or if it has size 0.
pub fn find_simple_roots(cm: &CartanMatrix) -> Vec<Root> {
    match cm.dim() {
        (d1, d2) if d1 != d2 => panic!("The Cartan matrix must be square."),
        (0, 0) => panic!("The Cartan matrix must be non-zero in size."),
        _ => (),
    }

    let mut roots: Vec<_> = cm.axis_iter(Axis(0))
        .enumerate()
        .map(|(i, w)| Root::simple(w.to_owned(), i))
        .rev()
        .collect();
    roots.sort_unstable();
    roots
}

/// Find all the roots given a set of simple roots.
///
/// This algorithmically finds and creates all of the roots in a root system
/// given a set of simple roots, returning a sorted vector.  This function
/// assumes the initial lost of simple roots to already be sorted.
///
/// This algorithm can be quite time consuming, especially for very large root
/// systems (more than 20 simple roots), but will automatically multithread in
/// such circumstances.  Ideally, this list should be generated once and then
/// stored for later use as needed.
///
/// # Example
///
/// ```
/// use lie::Root;
/// use lie::root_system::find_roots_from_simple;
///
/// let simple_roots = vec![
///     Root::simple(vec![2, -1], 0),
///     Root::simple(vec![-3, 2], 1),
/// ];
/// let roots = find_roots_from_simple(&simple_roots);
/// assert_eq!(roots.len(), 14);
/// ```
///
/// # Panics
///
/// Panics if the list of simple roots is empty.  Also panics if any of the
/// simple roots provided is actually not simple.
pub fn find_roots_from_simple(simple_roots: &[Root]) -> Vec<Root> {
    find_roots_from_positive(&find_positive_roots(simple_roots))
}

/// Find all the roots given a set of simple roots.
///
/// This algorithmically finds and creates all of the roots in a root system
/// given a set of positive roots, returning a sorted vector.  This function
/// assumes the initial lost of positive roots to already be sorted.
///
/// This algorithm can be quite time consuming, especially for very large root
/// systems (more than 20 simple roots), but will automatically multithread in
/// such circumstances.  Ideally, this list should be generated once and then
/// stored for later use as needed.
///
/// # Example
///
/// ```
/// use lie::Root;
/// use lie::root_system::{find_roots_from_positive, find_positive_roots};
///
/// let simple_roots = vec![
///     Root::simple(vec![2, -1], 0),
///     Root::simple(vec![-3, 2], 1),
/// ];
/// let positive_roots = find_positive_roots(&simple_roots);
/// let roots = find_roots_from_positive(&positive_roots);
/// for r in &roots {
///     println!("{}", r);
/// }
/// assert_eq!(roots.len(), 14);
/// ```
///
/// # Panics
///
/// Panics if the list of positive roots is empty.
pub fn find_roots_from_positive(positive_roots: &[Root]) -> Vec<Root> {
    assert!(
        !positive_roots.is_empty(),
        "The list of positive roots must be non-empty."
    );
    let rank = positive_roots[0].rank();
    positive_roots
        .iter()
        .rev()
        .map(|r| -1 * r)
        .chain((0..rank).map(|_| Root::zero(rank)))
        .chain(positive_roots.iter().map(|r| r.clone()))
        .collect()
}

/// Find all the roots given a set of simple roots.
///
/// This algorithmically finds and creates all of the roots in a root system
/// given a set of simple roots, returning a sorted vector.  This function
/// assumes the initial lost of simple roots to already be sorted.
///
/// This algorithm can be quite time consuming, especially for very large root
/// systems (more than 20 simple roots), but will automatically multithread in
/// such circumstances.  Ideally, this list should be generated once and then
/// stored for later use as needed.
///
/// # Example
///
/// ```
/// use lie::Root;
/// use lie::root_system::find_positive_roots;
///
/// let simple_roots = vec![
///     Root::simple(vec![2, -1], 0),
///     Root::simple(vec![-3, 2], 1),
/// ];
/// let roots = find_positive_roots(&simple_roots);
/// assert_eq!(roots.len(), 6);
/// ```
///
/// # Panics
///
/// Panics if the list of simple roots is empty.  Also panics if any of the
/// simple roots provided is actually not simple.
pub fn find_positive_roots(simple_roots: &[Root]) -> Vec<Root> {
    assert!(
        !simple_roots.is_empty(),
        "At least one simple root must be provided."
    );
    assert!(
        simple_roots.iter().all(|r| r.is_simple()),
        "All roots provided must be simple."
    );

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
    if simple_roots[0].rank() < 24 {
        find_positive_roots_single_thread(simple_roots)
    } else {
        find_positive_roots_multithread(simple_roots)
    }
}

/// Internal function to find all simple roots using Rayon's parallelization.
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

/// Internal function to find all simple roots using a single thread.
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

#[cfg(test)]
mod test {
    use super::CartanMatrix;
    use root::Root;

    #[test]
    fn find_simple_roots() {
        let cm = CartanMatrix::from(vec![[2, -1], [-3, 2]]);
        let simple_roots = super::find_simple_roots(&cm);
        assert_eq!(simple_roots.len(), 2);
        assert_eq!(simple_roots[0], Root::simple(vec![-3, 2], 1));
        assert_eq!(simple_roots[1], Root::simple(vec![2, -1], 0));
    }

    #[test]
    fn find_positive_roots() {
        let cm = CartanMatrix::from(vec![[2, -1], [-3, 2]]);
        let simple_roots = super::find_simple_roots(&cm);
        let positive_roots = super::find_positive_roots(&simple_roots);
        assert_eq!(positive_roots.len(), 6);
        assert_eq!(positive_roots[0], Root::new(vec![-3, 2], vec![0, 1]));
        assert_eq!(positive_roots[1], Root::new(vec![2, -1], vec![1, 0]));
        assert_eq!(positive_roots[2], Root::new(vec![-1, 1], vec![1, 1]));
        assert_eq!(positive_roots[3], Root::new(vec![1, 0], vec![2, 1]));
        assert_eq!(positive_roots[4], Root::new(vec![3, -1], vec![3, 1]));
        assert_eq!(positive_roots[5], Root::new(vec![0, 1], vec![3, 2]));
    }

    #[test]
    fn find_roots() {
        let cm = CartanMatrix::from(vec![[2, -1], [-3, 2]]);
        let simple_roots = super::find_simple_roots(&cm);
        let positive_roots = super::find_positive_roots(&simple_roots);
        let roots1 = super::find_roots_from_simple(&simple_roots);
        let roots2 = super::find_roots_from_positive(&positive_roots);

        assert_eq!(roots1, roots2);

        assert_eq!(roots1.len(), 14);
        assert_eq!(roots1[0], Root::new(vec![0, -1], vec![-3, -2]));
        assert_eq!(roots1[1], Root::new(vec![-3, 1], vec![-3, -1]));
        assert_eq!(roots1[2], Root::new(vec![-1, 0], vec![-2, -1]));
        assert_eq!(roots1[3], Root::new(vec![1, -1], vec![-1, -1]));
        assert_eq!(roots1[4], Root::new(vec![-2, 1], vec![-1, 0]));
        assert_eq!(roots1[5], Root::new(vec![3, -2], vec![0, -1]));
        assert_eq!(roots1[6], Root::zero(2));
        assert_eq!(roots1[7], Root::zero(2));
        assert_eq!(roots1[8], Root::new(vec![-3, 2], vec![0, 1]));
        assert_eq!(roots1[9], Root::new(vec![2, -1], vec![1, 0]));
        assert_eq!(roots1[10], Root::new(vec![-1, 1], vec![1, 1]));
        assert_eq!(roots1[11], Root::new(vec![1, 0], vec![2, 1]));
        assert_eq!(roots1[12], Root::new(vec![3, -1], vec![3, 1]));
        assert_eq!(roots1[13], Root::new(vec![0, 1], vec![3, 2]));
    }
}
