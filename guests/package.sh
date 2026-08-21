#!/bin/sh
# Packages the WASM guests as installable plugin directories and tarballs.
#
# `build.sh` produces bare `.wasm` files, which nothing can install: a plugin is
# a *directory* with `.plank-plugin/plugin.json` naming its modules, and the
# modules live under `wasm/`. This assembles that layout so a release has
# something `/plugins install` can point at.
#
# The manifest is source (`guests/<guest>/plugin.json`), not generated here: it
# declares the component's id, surfaces and capability grants, which are facts
# about the guest rather than about how it was packaged.
#
# Output: dist/<guest>/ (the installable directory) and dist/plank-<guest>.tar.gz.
set -e
cd "$(dirname "$0")"
ROOT=$(pwd)
DIST="$ROOT/dist"
rm -rf "$DIST"
mkdir -p "$DIST"

sh ./build.sh >/dev/null

for guest in screensavers arcades; do
  built="$ROOT/$guest/target/wasm32-unknown-unknown/release/plank_$guest.wasm"
  if [ ! -f "$built" ]; then
    echo "missing build output: $built" >&2
    exit 1
  fi
  # The module name inside the plugin matches the manifest's `module` field, not
  # cargo's `plank_<guest>` crate name: the manifest is what the host reads.
  out="$DIST/$guest"
  mkdir -p "$out/.plank-plugin" "$out/wasm"
  cp "$ROOT/$guest/plugin.json" "$out/.plank-plugin/plugin.json"
  cp "$built" "$out/wasm/$guest.wasm"
  # `-C` so the archive holds `<guest>/...` rather than the absolute path, and
  # so extracting it yields a directory that is already a plugin.
  tar -czf "$DIST/plank-$guest.tar.gz" -C "$DIST" "$guest"
  echo "plugin: $out"
done

# The trust store keys on a *module* hash, so that is what gets recorded — the
# tarball's hash says nothing about what plank will actually load. The rustc
# version is recorded beside them because the bytes are only reproducible for a
# given toolchain: a verifier needs to know what to reproduce *with*.
#
# Verified with guests/verify.sh, which rebuilds from a clean tree and compares.
{
  echo "# plank guest modules"
  echo "# $(rustc --version)"
  for guest in screensavers arcades; do
    (cd "$DIST/$guest/wasm" && shasum -a 256 "$guest.wasm")
  done
} > "$DIST/SHA256SUMS"
cat "$DIST/SHA256SUMS"
