//! Definitions for the 7 families of Lie groups.
//!
//! Four of the families form infinite series:
//!
//! - \\(A_{n} = \mathrm{SU}(n+1)\\), the [special unitary
//!   group](https://en.wikipedia.org/wiki/Special_unitary_group);
//!
//! - \\(B_{n} = \mathrm{SO}(2n + 1)\\), the [special orthogonal
//!   group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of odd
//!   dimension;
//!
//! - \\(C_{n} = \mathrm{Sp}(2n)\\), the group of [unitary symplectic
//!   matrices](https://en.wikipedia.org/wiki/Symplectic_matrix); and,
//!
//! - \\(D_{n} = \mathrm{SO}(2n)\\), the [special orthogonal
//!   group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of even
//!   dimension.
//!
//! The last three families contain the 5 exceptional Lie groups:
//!
//! - [\\(G_{2}\\)](https://en.wikipedia.org/wiki/G2_(mathematics));
//! - [\\(F_{4}\\)](https://en.wikipedia.org/wiki/F4_(mathematics));
//! - [\\(E_{6}\\)](https://en.wikipedia.org/wiki/E6_(mathematics));
//! - [\\(E_{7}\\)](https://en.wikipedia.org/wiki/E7_(mathematics)); and,
//! - [\\(E_{8}\\)](https://en.wikipedia.org/wiki/E8_(mathematics)).
//!
//! The corresponding Lie algebras are conventionally denoted using fraktur
//! symbols: \\(\mathfrak{su}\_{n}\\), \\(\mathfrak{so}\_{n}\\),
//! \\(\mathfrak{sp}\_{n}\\), \\(\mathfrak{g}\_{2}\\), \\(\mathfrak{f}\_{4}\\),
//! \\(\mathfrak{e}\_{6}\\), \\(\mathfrak{e}\_{7}\\) and
//! \\(\mathfrak{e}\_{8}\\).

//! ## Exceptional Isomorphisms
//!
//! For low values of \\(n\\), there are accidental isomorphisms between the
//! above categories.  In particular, they are:
//!
//! - \\(A_{1} \cong B_{1} \cong C_{1}\\),
//! - \\(B_{2} \cong C_{2}\\),
//! - \\(D_{2} \cong A_{1} \times A_{1}\\),
//! - \\(D_{3} \cong A_{3}\\),
//! - \\(E_{3} \cong A_{1} \times A_{2}\\)
//! - \\(E_{4} \cong A_{4}\\),
//! - \\(E_{5} \cong D_{5}\\).
//!
//! As a result, the indices for \\(A_{n}\\), \\(B_{n}\\), \\(C_{n}\\) and
//! \\(D_{n}\\) start at 1, 2, 3 and 4 respectively.
//!
//! This library does *not* handle exceptional isomorphisms, though it might
//! return an error indicating what the standard label for the specified group
//! is.  For example, trying to create an instance of the \\(B_{1} =
//! \mathrm{SO}(3)\\) group will produce an error and instead one must use the
//! isomorphic group \\(A_{1} = \mathrm{SU}(2)\\).
mod type_a;
mod type_b;
mod type_c;
mod type_d;
mod type_e;
mod type_f;
mod type_g;

pub use self::type_a::TypeA;
pub use self::type_b::TypeB;
pub use self::type_c::TypeC;
pub use self::type_d::TypeD;
pub use self::type_e::TypeE;
pub use self::type_f::TypeF;
pub use self::type_g::TypeG;
