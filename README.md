# lie

`lie-rs` provides functionalities to manipulate Lie groups and their
representations, including the ability to:

- Quickly verify whether a product of irreducible representations can form a
  singlet;
- Decompose the product of irreducible representation into sum of
  irreducible representations;
- List how indices of a product of irreducible representations can be
  contracted to form a singlet; and
- Other tools to manipulate root systems and Cartan matrices.

**This library is still undergoing development.**

[![Crates.io](https://img.shields.io/crates/v/hep-lie.svg)](https://crates.io/crates/hep-lie)
[![Travis](https://img.shields.io/travis/hep-rs/lie/master.svg)](https://travis-ci.org/hep-rs/lie)
[![Codecov](https://img.shields.io/codecov/c/github/hep-rs/lie/master.svg)](https://codecov.io/gh/hep-rs/lie)

Licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).

The library is primarily developed within the context of particle physics;
however, this should in no way be a restriction for contributions to this
library and extensions beyond the initial scope of this library are very
welcome.

It is hoped that this library will be accessible to people who are not fully
familiar with all aspects of the study of Lie groups and their
representations.  The documentation will attempt to provide explanations of
various definitions.

## Lie Groups and Algebra

A *Lie group*, \\(G\\), is a group that is also a differential manifold
which ultimately means that all elements of the group \\(G\\) can be
describe by a set real parameters.  For example, the group of real \\(2
\times 2\\) orthognal matrices with determinant 1, \\(\mathrm{SO}(2)\\), can
be described in terms of one parameter:

\\begin{equation}
  \mathrm{SO}(2) = \left\\{
    \begin{pmatrix} \cos \theta & \sin \theta \\ -\sin \theta && \cos \theta \end{pmatrix}
    \middle\|
    \theta \in [0, 2\pi)
  \\}
\\end{equation}

This library will look at handling only compact and simply connected groups.

Very closely related to Lie groups are *Lie algebras* which are denoted
using lowercase Gothic characters \\(\mathfrak{g}\\).  A Lie algebra is a
(real or complex) vector space \\(\mathfrak{g}\\) together with a map
\\([\cdot, \cdot] : \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}\\)
which satisfies:

- \\([\cdot, \cdot]\\) is bilinear;
- \\([\cdot, \cdot]\\) is skew symmetry: \\([x, y] = -[y, x]\\) for all \\(x, y
  \in \mathfrak{g}\\); and,
- The Jacobi identity holds:
  \\begin{equation}
    [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
  \\end{equation}
  for all \\(x, y, z \in \mathfrak{g}\\).

Two elements \\(x, y \in \mathfrak{g}\\) commute if \\([x, y] = 0\\), and if
all elements in \\(\mathfrak{g}\\) commute with each other, the Lie algebra
is commutative

Although mathematically very different objects, Lie groups and Lie algebras
are very closely related and the distinction will occasionally be blurred.
In particular, given that a Lie group \\(G\\) forms smooth manifold, one can
consider the tangent space at the identity.  This tangent space forms
exactly the group's Lie algebra.  Through the exponential map and its
inverse, there is a correspondence between the Lie group and its Lie
algebra.

- \\(A_{n} = \mathrm{SU}(n+1)\\), the [special unitary
  group](https://en.wikipedia.org/wiki/Special_unitary_group);

- \\(B_{n} = \mathrm{SO}(2n + 1)\\), the [special orthogonal
  group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of odd
  dimension;

- \\(C_{n} = \mathrm{Sp}(2n)\\), the group of [unitary symplectic
  matrices](https://en.wikipedia.org/wiki/Symplectic_matrix);

- \\(D_{n} = \mathrm{SO}(2n)\\), the [special orthogonal
  group](https://en.wikipedia.org/wiki/Special_orthogonal_group) of even
  dimension; and,

- The exception Lie group
  [\\(G_{2}\\)](https://en.wikipedia.org/wiki/G2_(mathematics)),
  [\\(F_{4}\\)](https://en.wikipedia.org/wiki/F4_(mathematics)),
  [\\(E_{6}\\)](https://en.wikipedia.org/wiki/E6_(mathematics)),
  [\\(E_{7}\\)](https://en.wikipedia.org/wiki/E7_(mathematics)) and
  [\\(E_{8}\\)](https://en.wikipedia.org/wiki/E8_(mathematics)).

The corresponding Lie algebras are conventionally denoted using fraktur
symbols: \\(\mathfrak{su}_{n}\\), \\(\mathfrak{so}_{n}\\),
\\(\mathfrak{sp}_{n}\\), \\(\mathfrak{g}_{2}\\), \\(\mathfrak{f}_{4}\\),
\\(\mathfrak{e}_{6}\\), \\(\mathfrak{e}_{7}\\) and \\(\mathfrak{e}_{8}\\).

### Exceptional Isomorphisms

For low values of \\(n\\), there are accidental isomorphisms between the
above categories.  In particular, they are:

- \\(A_{1} \cong B_{1} \cong C_{1}\\),
- \\(B_{2} \cong C_{2}\\),
- \\(D_{2} \cong A_{1} \times A_{1}\\),
- \\(D_{3} \cong A_{3}\\),
- \\(E_{3} \cong A_{1} \times A_{2}\\)
- \\(E_{4} \cong A_{4}\\),
- \\(E_{5} \cong D_{5}\\).

As a result, the indices for \\(A_{n}\\), \\(B_{n}\\), \\(C_{n}\\) and
\\(D_{n}\\) start at 1, 2, 3 and 4 respectively.

This library does *not* handle exceptional isomorphisms, though it might
return an error indicating what the standard label for the specified group
is.  For example, trying to create an instance of the \\(\mathrm{SO}(3)\\)
group will produce an error and instead one must use the isomorphic group
\\(\mathrm{SU}(2)\\).

## Root Systems

A [root system](https://en.wikipedia.org/wiki/Root_system) is a
configuration of vectors in Euclidean space which satisfy certain
geometrical properties.  Root systems are extremely useful in the study of
simple Lie groups and their representations.

### Definition

Let \\(V\\) be a finite dimensional Euclidean vector space with the standard
Euclidean inner product being denoted by \\(\langle \cdot, \cdot \rangle\\).
A *root system* in \\(V\\) is a finite set \\(\Phi\\) of non-zero vectors
(called *roots*) that satisfy the following conditions:

1. The roots span \\(V\\);

2. The only scalar multiple of a root \\(\alpha \in \Phi\\) that belong to
   \\(\Phi\\) are \\(\alpha\\) and \\(-\alpha\\);

3. For every root \\(\alpha \in \Phi\\), the set \\(\Phi\\) is closed under
   reflections through the hyperplane perpendicular to \\(\alpha\\);

4. If \\(\alpha, \beta \in \Phi\\), then the projection of
   \\(\beta\\) onto \\(\alpha\\) is a half-integral multiple of
   \\(\alpha\\).

In addition to the set of roots themselves, there are two important subsets
of \\(\Phi\\) which are commonly used:

- \\(\Phi\^{+} \subseteq \Phi\\): the set of positive roots.  This is the
  subset chosen such that:
  - For each \\(\alpha \in \Phi\\), only one of \\(\alpha\\) and
    \\(-\alpha\\) are in \\(\Phi\^{+}\\); and,
  - For any two distinct \\(\alpha, \beta \in \Phi\^{+}\\) then \\(\alpha +
    \beta \in \Phi\^{+}\\) provided that \\(\alpha + \beta \in \Phi\\).
- \\(\Delta \subseteq \Phi\^{+}\\): the set of simple root.  An element of
  \\(\Phi\^{+}\\) is a simple root if it cannot be expressed as the sum of
  two other elements in \\(\Phi\^{+}\\).  The set of simple roots has the
  property that every root in \\(\Phi\\) can be expressed as linear
  combination of elements of \\(\Delta\\) with all coefficients
  non-negative, or all coefficients non-positive.

### Connection to Group Representations
