#!/bin/sh
# Builds the feasibility-stage guests. Requires the wasm32-unknown-unknown
# target (`rustup target add wasm32-unknown-unknown`). The host tests that
# consume these artifacts skip when they are absent, so this is never on the
# critical path of a normal build or of CI's default job.
set -e
cd "$(dirname "$0")"
# Builds are compared by hash (`verify.sh`, the release `SHA256SUMS`), and
# panic locations embed absolute source paths, so map the two that differ per
# machine to fixed names. `cygpath -m` gives rustc a native path under Git Bash.
# The third remap maps a local `rust-src` copy back to std's `/rustc/<commit>`
# paths; see guests/build.sh.
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
RUST_SRC="$(native "$(rustc --print sysroot)")/lib/rustlib/src/rust"
RUST_COMMIT=$(rustc -vV | sed -n 's/^commit-hash: //p')
REMAP="--remap-path-prefix=$(native "${CARGO_HOME:-$HOME/.cargo}")=/cargo --remap-path-prefix=$(native "$(pwd)")=/src --remap-path-prefix=$RUST_SRC=/rustc/$RUST_COMMIT"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$REMAP"
for guest in abi-guest min-guest text-guest; do
  (cd "$guest" && cargo build --release --target wasm32-unknown-unknown)
  echo "guest: ${CARGO_TARGET_DIR:-$(pwd)/$guest/target}/wasm32-unknown-unknown/release/$(echo "plank_$guest" | tr - _).wasm"
done
