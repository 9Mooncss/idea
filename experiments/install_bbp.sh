#!/bin/sh
# Builds the official bbpPairings engine into experiments/bbp/ (needs git, g++, make).
set -e
D=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
git clone -q https://github.com/BieremaBoyzProgramming/bbpPairings.git "$T/src"
(cd "$T/src" && git checkout -q 8f9e3c5ffdc4d7a08a33d31c6fb49b4e7acef4d5 && make -j4 >/dev/null)
mkdir -p "$D/bbp" && cp "$T/src/bbpPairings.exe" "$D/bbp/"
echo "built $D/bbp/bbpPairings.exe"
