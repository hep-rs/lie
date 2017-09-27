# Exit on any error
set -eux

# Run clippy and see if it has anything to say
clippy() {
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" && $CLIPPY ]]; then
        cargo clippy $FEATURES
    fi
}

# Run the standard build and test suite.
build_and_test() {
    cargo build
    cargo test $FEATURES
}

main() {
    build_and_test
}

main
