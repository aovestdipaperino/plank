#!/bin/sh
# Builds the WASM guests. Requires the wasm32-unknown-unknown and wasm32-wasip1
# targets (`rustup target add wasm32-unknown-unknown wasm32-wasip1`). The host
# tests that consume these artifacts skip when they are absent, so this is never
# on the critical path of a normal build or of CI's default job.
#
# `screensavers` and `arcades` are separate plugins, not one artifact with a
# flag: they are separate things, and a user should be able to install ambient
# faces without installing games.
set -e
cd "$(dirname "$0")"
# Builds are compared by hash (`verify.sh`, the release `SHA256SUMS`), and
# panic locations embed absolute source paths, so map the two that differ per
# machine to fixed names. `cygpath -m` gives rustc a native path under Git Bash.
#
# The third remap undoes a substitution rustc makes when the `rust-src`
# component is installed (rust-analyzer installs it): std's locations, shipped
# as `/rustc/<commit>/library/...`, are rewritten to the local
# `<sysroot>/lib/rustlib/src/rust/library/...`, so the module would embed the
# toolchain's absolute path, user name included, and hash differently from a
# machine without `rust-src`. Mapping it back gives every machine the
# `/rustc/<commit>` form. Without `rust-src` it matches nothing.
#
# What no flag fixes: on Windows rustc joins the rest of each path with `\`
# (`/cargo\registry\...`), so a Windows build never matches a macOS or Linux
# one byte for byte. `package.sh` records the OS family and `verify.sh` says so.
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
RUST_SRC="$(native "$(rustc --print sysroot)")/lib/rustlib/src/rust"
RUST_COMMIT=$(rustc -vV | sed -n 's/^commit-hash: //p')
REMAP="--remap-path-prefix=$(native "${CARGO_HOME:-$HOME/.cargo}")=/cargo --remap-path-prefix=$(native "$(pwd)")=/src --remap-path-prefix=$RUST_SRC=/rustc/$RUST_COMMIT"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$REMAP"
# Where cargo puts a guest's build: its own `target/`, or `CARGO_TARGET_DIR`
# (absolute) when set, which every guest then shares; the crate names differ.
target_of() { printf '%s' "${CARGO_TARGET_DIR:-$(pwd)/$1/target}"; }
for guest in screensavers arcades; do
  (cd "$guest" && cargo build --release --target wasm32-unknown-unknown)
  echo "guest: $(target_of "$guest")/wasm32-unknown-unknown/release/plank_$guest.wasm"
done
# csvedit is the plank-tv submodule (`git submodule update --init
# guests/csvedit`), and it needs WASI: Turbo Vision reads the clock.
(cd csvedit && cargo build --release --target wasm32-wasip1)
echo "guest: $(target_of csvedit)/wasm32-wasip1/release/plank_tv.wasm"
