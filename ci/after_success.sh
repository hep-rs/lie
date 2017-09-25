# Exit on any error
set -ux

install_kcov() {
    set -e
    # Download and install kcov
    wget https://github.com/SimonKagstrom/kcov/archive/master.tar.gz -O - | tar -xz
    cd kcov-master
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    make install DESTDIR=../../kcov-build
    cd ../..
    rm -rf kcov-master
    set +e
}

run_kcov() {
    # Run kcov on all the test suites
    for file in target/debug/lie-*[^\.d]; do
        mkdir -p "target/cov/$(basename $file)";
        echo "Testing $(basename $file)"
        ./kcov-build/usr/local/bin/kcov \
            --exclude-pattern=/.cargo,/usr/lib\
            --verify "target/cov/$(basename $file)" \
            "$file";
    done

    bash <(curl -s https://codecov.io/bash)
    echo "Uploaded code coverage"
}

kcov_suite() {
    if [[ "$TRAVIS_RUST_VERSION" == "stable" ]]; then
        install_kcov
        run_kcov
    fi
}

make_doc() {
    if [[ "$TRAVIS_RUST_VERSION" == "stable" ]]; then
        SED_SUB='s|</body>|<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML"></script></body>|'

        cargo doc
        find $(dirname $(realpath $0))/target/doc \
             -type f \
             -name "*.html" \
             -exec sed -i "$SED_SUB" '{}' +
    fi
}

main() {
    kcov_suite
    make_doc
}

main
