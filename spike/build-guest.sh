#!/bin/sh
# Builds the feasibility-stage guest. Requires the wasm32-unknown-unknown
# target (`rustup target add wasm32-unknown-unknown`). The host tests that
# consume the artifact skip when it is absent, so this is never on the
# critical path of a normal build or of CI's default job.
set -e
cd "$(dirname "$0")/abi-guest"
cargo build --release --target wasm32-unknown-unknown
echo "guest: $(cd .. && pwd)/abi-guest/target/wasm32-unknown-unknown/release/plank_abi_guest.wasm"
