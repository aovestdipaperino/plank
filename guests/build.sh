#!/bin/sh
# Builds the WASM guests. Requires the wasm32-unknown-unknown target
# (`rustup target add wasm32-unknown-unknown`). The host tests that consume
# these artifacts skip when they are absent, so this is never on the critical
# path of a normal build or of CI's default job.
#
# `screensavers` and `arcades` are separate plugins, not one artifact with a
# flag: they are separate things, and a user should be able to install ambient
# faces without installing games.
set -e
cd "$(dirname "$0")"
for guest in screensavers; do
  (cd "$guest" && cargo build --release --target wasm32-unknown-unknown)
  echo "guest: $(pwd)/$guest/target/wasm32-unknown-unknown/release/plank_$guest.wasm"
done
