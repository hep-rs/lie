use ndarray::Array1;
use std::ops;
use std::cmp;
use std::fmt;

/// Instance of a root.
///
/// A root is a vector in the underlying (Euclidean) vector space.  Note that we
/// do not store these vectors from the underlying vector space, but instead use
/// a different basis which is dependent on the root system and the selection of
/// simple roots.
///
/// The roots are stored in the following two bases:
///
/// - **\\(\alpha\\)-basis**: Represents the root \\(\beta\\) as the array
///   \\((k_{1}, \dots, k_{n})\\) such that \\(\beta = \sum_{i} k_{i}
///   \alpha_{i}\\), where \\(\alpha_{i}\\) are the simple roots.  Note that
///   \\(n\\) corresponds to the number of simple roots and thus the rank of the
///   corresponding Lie group.
///
/// - **\\(\omega\\)-basis**: In the \\(\omega\\) basis, each \\(\alpha_{i}\\)
///   corresponds to the \\(i\\)th row of Cartan matrix.  Thus a root
///   \\(\beta\\) in the \\(\omega\\) basis is expressed \\(\beta_{j} = \sum_{i}
///   k_{i} A_{ij}\\).
#[derive(Clone)]
pub struct Root {
    omega: Array1<i64>,
    alpha: Array1<i64>,
}

impl Root {
    /// Create a zero-root.
    ///
    /// A zero root has all weights set to zero and is at level 0.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let r = Root::zero(5);
    /// assert_eq!(r.level(), 0);
    /// assert_eq!(r.rank(), 5);
    /// assert!(!r.is_simple());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the rank is 0.
    pub fn zero(rank: usize) -> Self {
        if rank == 0 {
            panic!("Rank of a zero root must be at least 1.")
        } else {
            Root {
                omega: Array1::zeros(rank),
                alpha: Array1::zeros(rank),
            }
        }
    }

    /// Create a simple root.
    ///
    /// A simple root is a root which cannot be written as the linear
    /// combination of any other root in the system.  Note that creating a
    /// simple root through this function *defines* the root to be simple, and
    /// it will depend on the root system itself whether this is true or not.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let r = Root::simple(vec![2, -1, 0], 0);
    /// assert!(r.is_simple());
    /// assert_eq!(r.level(), 1);
    /// assert_eq!(r.alpha().as_slice().unwrap(), &[1, 0, 0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is greater than the length of the omega basis (or
    /// equivalently, if `alpha` is greater than the rank of the corresponding
    /// root system).
    pub fn simple<A>(omega: A, alpha: usize) -> Self
    where
        A: Into<Array1<i64>>,
    {
        let omega = omega.into();
        if alpha >= omega.dim() {
            panic!("Simple root must have an index that is strictly less than rank.")
        } else {
            let alpha = Array1::from_shape_fn(omega.dim(), |i| if i == alpha { 1 } else { 0 });
            Root { omega, alpha }
        }
    }

    /// Return the rank of the root's corresponding Lie group.
    ///
    /// Although the root on its own has no way of this, the number of
    /// coefficient in the root's \\(\alpha\\)-basis is the number of simple
    /// roots of the root system which corresponds to the Lie group's rank.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let r = Root::simple(vec![2, -1, 0], 0);
    /// assert_eq!(r.rank(), 3);
    /// ```
    pub fn rank(&self) -> usize {
        self.alpha.dim()
    }

    /// Return the level of the root.
    ///
    /// A root \\(\beta\\) can be written uniquely as a linear combination of
    /// simple roots \\(\alpha_{i}\\), \\(\beta = \sum_{i} k_{i} \alpha_{i}\\).
    /// The level of \\(\beta\\) is then defined as \\(\sum_{i} k_{i}\\).
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let a1 = Root::simple(vec![2, -1, 0], 0);
    /// let a2 = Root::simple(vec![-1, 2, -1], 1);
    ///
    /// assert_eq!(a1.level(), 1);
    /// assert_eq!((&a1 + &a2).level(), 2);
    /// assert_eq!((&a1 + &a2 + &a2).level(), 3);
    pub fn level(&self) -> i64 {
        self.alpha.scalar_sum()
    }

    /// Return the root's decomposition in terms of the \\(\omega\\) basis.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let a1 = Root::simple(vec![2, -1, 0], 0);
    /// let a2 = Root::simple(vec![-1, 2, -1], 1);
    ///
    /// let r = &a1 + &a2 + &a2;
    /// assert_eq!(a1.omega().as_slice().unwrap(), &[2, -1, 0]);
    /// assert_eq!(a2.omega().as_slice().unwrap(), &[-1, 2, -1]);
    /// assert_eq!(r.omega().as_slice().unwrap(), &[0, 3, -2]);
    /// ```
    pub fn omega(&self) -> &Array1<i64> {
        &self.omega
    }

    /// Return the root's decomposition in terms of the \\(\alpha\\) basis.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let a1 = Root::simple(vec![2, -1, 0], 0);
    /// let a2 = Root::simple(vec![-1, 2, -1], 1);
    ///
    /// let r = &a1 + &a2 + &a2;
    /// assert_eq!(r.level(), 3);
    /// assert_eq!(r.alpha().as_slice().unwrap(), &[1, 2, 0]);
    /// ```
    pub fn alpha(&self) -> &Array1<i64> {
        &self.alpha
    }

    /// Check whether the root is a simple root or not.
    ///
    /// Note that this ultimately requires information about the root system,
    /// which the root does not have access to.  As a result, this function's
    /// accuracy relies on the root having been correctly generated in the first
    /// place.
    ///
    /// # Example
    ///
    /// ```
    /// use lie::Root;
    ///
    /// let r = Root::zero(5);
    /// assert!(!r.is_simple());
    ///
    /// let r = Root::simple(vec![2, -1, 0], 0);
    /// assert!(r.is_simple());
    /// ```
    pub fn is_simple(&self) -> bool {
        self.level() == 1
    }
}

////////////////////////////////////////////////////////////////////////////////
// Trait Implementations
////////////////////////////////////////////////////////////////////////////////

impl fmt::Debug for Root {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.omega))
    }
}
impl fmt::Display for Root {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.omega))
    }
}

/// The roots are first ordered by level, and then by the coefficients in the
/// \\(\alpha\\)-basis if they are the same level.
///
/// # Example
///
/// ```
/// use lie::Root;
///
/// let a1 = Root::simple(vec![2, -1], 0);
/// let a2 = Root::simple(vec![-3, 2], 1);
/// let r1 = &a1 + &a2;
///
/// assert!(a1 > a2);
/// assert!(r1 > a1);
/// assert!(r1 > a2);
/// ```
impl cmp::PartialOrd for Root {
    fn partial_cmp(&self, other: &Root) -> Option<cmp::Ordering> {
        match self.level().cmp(&other.level()) {
            cmp::Ordering::Less => Some(cmp::Ordering::Less),
            cmp::Ordering::Greater => Some(cmp::Ordering::Greater),
            cmp::Ordering::Equal => self.alpha.iter().partial_cmp(other.alpha.iter()),
        }
    }
}

impl cmp::Ord for Root {
    fn cmp(&self, other: &Root) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl cmp::PartialEq for Root {
    fn eq(&self, other: &Root) -> bool {
        self.cmp(other) == cmp::Ordering::Equal
    }
}

impl Eq for Root {}

////////////////////////////////////////////////////////////////////////////////
// Arithmetic Operations
////////////////////////////////////////////////////////////////////////////////

impl ops::Add<Root> for Root {
    type Output = Self;

    fn add(self, other: Root) -> Self::Output {
        Root {
            omega: self.omega + other.omega,
            alpha: self.alpha + other.alpha,
        }
    }
}

impl<'a> ops::Add<&'a Root> for Root {
    type Output = Self;

    fn add(self, other: &Root) -> Self::Output {
        Root {
            omega: self.omega + &other.omega,
            alpha: self.alpha + &other.alpha,
        }
    }
}

impl<'a> ops::Add<&'a Root> for &'a Root {
    type Output = Root;

    fn add(self, other: &Root) -> Self::Output {

        Root {
            omega: &self.omega + &other.omega,
            alpha: &self.alpha + &other.alpha,
        }
    }
}

impl<'a> ops::Add<Root> for &'a Root {
    type Output = Root;

    fn add(self, other: Root) -> Self::Output {

        Root {
            omega: &self.omega + &other.omega,
            alpha: &self.alpha + &other.alpha,
        }
    }
}

impl ops::Sub<Root> for Root {
    type Output = Self;

    fn sub(self, other: Root) -> Self::Output {
        Root {
            omega: self.omega - other.omega,
            alpha: self.alpha - other.alpha,
        }
    }
}

impl<'a> ops::Sub<&'a Root> for Root {
    type Output = Self;

    fn sub(self, other: &Root) -> Self::Output {
        Root {
            omega: self.omega - &other.omega,
            alpha: self.alpha - &other.alpha,
        }
    }
}

impl<'a> ops::Sub<&'a Root> for &'a Root {
    type Output = Root;

    fn sub(self, other: &Root) -> Self::Output {

        Root {
            omega: &self.omega - &other.omega,
            alpha: &self.alpha - &other.alpha,
        }
    }
}

impl<'a> ops::Sub<Root> for &'a Root {
    type Output = Root;

    fn sub(self, other: Root) -> Self::Output {

        Root {
            omega: &self.omega - &other.omega,
            alpha: &self.alpha - &other.alpha,
        }
    }
}

impl ops::Mul<i64> for Root {
    type Output = Self;

    fn mul(self, other: i64) -> Self::Output {
        Root {
            omega: self.omega * other,
            alpha: self.alpha * other,
        }
    }
}

impl<'a> ops::Mul<i64> for &'a Root {
    type Output = Root;

    fn mul(self, other: i64) -> Self::Output {
        Root {
            omega: &self.omega * other,
            alpha: &self.alpha * other,
        }
    }
}

impl ops::Mul<Root> for i64 {
    type Output = Root;

    fn mul(self, other: Root) -> Self::Output {
        Root {
            omega: self * other.omega,
            alpha: self * other.alpha,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::Root;

    macro_rules! assert_gt {
        ( $left : expr, $right : expr) => {
            assert!($left > $right);
            assert!($right < $left);
            assert_ne!($left, $right)
        }
    }

    #[test]
    fn zero() {
        let r = Root::zero(5);
        assert_eq!(r.level(), 0);
        assert_eq!(r.rank(), 5);
        assert!(!r.is_simple());
    }

    #[test]
    fn simple() {
        let r = Root::simple(vec![2, -1, 0], 0);
        assert_eq!(r.level(), 1);
        assert_eq!(r.rank(), 3);
        assert!(r.is_simple());
    }

    #[test]
    fn ordering() {
        // Test the roots of G2 for ordering
        let a1 = Root::simple(vec![2, -1], 0);
        let a2 = Root::simple(vec![-3, 2], 1);
        let r1 = &a1 + &a2;
        let r2 = &r1 + &a1;
        let r3 = &r2 + &a1;
        let r4 = &r3 + &a2;

        assert_gt!(a1, a2);

        assert_gt!(r1, a1);
        assert_gt!(r1, a2);

        assert_gt!(r2, a1);
        assert_gt!(r2, a2);
        assert_gt!(r2, r1);

        assert_gt!(r3, a1);
        assert_gt!(r3, a2);
        assert_gt!(r3, r1);
        assert_gt!(r3, r2);

        assert_gt!(r4, a1);
        assert_gt!(r4, a2);
        assert_gt!(r4, r1);
        assert_gt!(r4, r2);
        assert_gt!(r4, r3);

        let mut v_unsorted = vec![
            r2.clone(),
            r4.clone(),
            r3.clone(),
            a1.clone(),
            r1.clone(),
            a2.clone(),
        ];
        let v_sorted = vec![a2, a1, r1, r2, r3, r4];
        v_unsorted.sort_unstable();
        assert_eq!(v_unsorted, v_sorted);

        // Test the roots of A3 for ordering
        let a1 = Root::simple(vec![2, -1, 0], 0);
        let a2 = Root::simple(vec![-1, 2, -1], 1);
        let a3 = Root::simple(vec![0, -1, 2], 2);
        let r1 = &a1 + &a2;
        let r2 = &a2 + &a3;
        let r3 = &a1 + &a2 + &a3;

        assert_gt!(a2, a3);
        assert_gt!(a1, a2);
        assert_gt!(a1, a3);

        assert_gt!(r1, a1);
        assert_gt!(r1, a2);
        assert_gt!(r1, a3);

        assert_gt!(r2, a1);
        assert_gt!(r2, a2);
        assert_gt!(r2, a3);
        assert_gt!(r1, r2);

        assert_gt!(r3, a1);
        assert_gt!(r3, a2);
        assert_gt!(r3, a3);
        assert_gt!(r3, r1);
        assert_gt!(r3, r2);

        let mut v_unsorted = vec![
            r2.clone(),
            a3.clone(),
            r3.clone(),
            a1.clone(),
            r1.clone(),
            a2.clone(),
        ];
        let v_sorted = vec![a3, a2, a1, r2, r1, r3];
        v_unsorted.sort_unstable();
        assert_eq!(v_unsorted, v_sorted);
    }
}
