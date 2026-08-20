#!/bin/sh
# Builds the feasibility-stage guests. Requires the wasm32-unknown-unknown
# target (`rustup target add wasm32-unknown-unknown`). The host tests that
# consume these artifacts skip when they are absent, so this is never on the
# critical path of a normal build or of CI's default job.
set -e
cd "$(dirname "$0")"
for guest in abi-guest min-guest text-guest; do
  (cd "$guest" && cargo build --release --target wasm32-unknown-unknown)
  echo "guest: $(pwd)/$guest/target/wasm32-unknown-unknown/release/$(echo "plank_$guest" | tr - _).wasm"
done
