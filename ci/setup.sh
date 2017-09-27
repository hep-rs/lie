# Exit on any error
set -eux

export PATH=$PATH:~/.cargo/bin

setup_features() {
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" ]]; then
        export FEATURES="--all-features"
    else
        export FEATURES=""
    fi
}

# Check if we are running nightly and install clippy if so.
clippy() {
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" ]]; then
        ( cargo install clippy && export CLIPPY=true ) || export CLIPPY=false;
    fi
}

main() {
    clippy
    setup_features
}

main
