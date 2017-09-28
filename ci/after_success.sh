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

setup_git() {
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "Travis CI"
}

make_doc() {
    # Only want to update the docs when pushing to master.  Ultimately, this
    # should be changed to using a tag once things have stabilized.
    if [[ "$TRAVIS_RUST_VERSION" == "stable"
       && "$TRAVIS_EVENT_TYPE" == "push"
       && "$TRAVIS_BRANCH" == "master" ]]; then

        cargo doc $FEATURES

        SED_SUB='s|</body>|<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML"></script></body>|'
        find ./target/doc \
             -type f \
             -name "*.html" \
             -exec sed -i "$SED_SUB" '{}' +

        setup_git
        git clone --depth=1 --branch=gh-pages https://${GH_TOKEN}@github.com/$TRAVIS_REPO_SLUG.git ./target/doc-git

        cd ./target/doc-git
        for f in index.html; do
            cp $f ../doc/
        done
        rm -rf *
        cp -R ../doc/* .
        git add -A .
        git commit --message "Travis build: $TRAVIS_BUILD_NUMBER"
        git push

    fi
}

main() {
    kcov_suite
    make_doc
}

main
