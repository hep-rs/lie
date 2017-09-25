# Exit on any error
set -eux

# Run clippy and see if it has anything to say
clippy() {
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" && $CLIPPY ]]; then
        cargo clippy
    fi
}

# Run the standard build and test suite.
build_and_test() {
    cargo build
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" ]]; then
        cargo test --all-features
    else
        cargo test
    fi
}

main() {
    build_and_test
}

main
