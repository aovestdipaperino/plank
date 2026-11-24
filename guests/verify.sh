#!/bin/sh
# Rebuilds the guests from a clean tree and checks they still hash to what
# `package.sh` recorded.
#
# Why this exists: the trust store keys on a module's SHA-256, so a published
# artifact nobody can independently reproduce makes that hash an assertion
# rather than a check. The modules *are* byte-reproducible — same toolchain, any
# build directory — and this is what keeps that true rather than assumed.
#
# What it does not prove: reproducibility across rustc versions. `SHA256SUMS`
# records the version it was built with for exactly that reason. A mismatch
# after a toolchain bump is expected and is not evidence of tampering; a
# mismatch on the same toolchain is.
set -e
# SHA-256 of a file as `<hash>  <name>`: `shasum -a 256` where it exists
# (macOS, Git for Windows), else `sha256sum` (most Linux). Git Bash's
# `sha256sum` marks binary mode as `<hash> *<name>`; the `*` is dropped so
# every OS writes the same line.
sha256() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi |
    sed 's/ \*/  /'
}
cd "$(dirname "$0")"
SUMS="${1:-dist/SHA256SUMS}"
if [ ! -f "$SUMS" ]; then
  echo "no $SUMS; run guests/package.sh first" >&2
  exit 1
fi
# package.sh below empties dist/, the default home of $SUMS: read the recorded
# sums from a copy, or the rebuild would be checked against its own output and
# always pass.
RECORDED=$(mktemp)
trap 'rm -f "$RECORDED"' EXIT
cp "$SUMS" "$RECORDED"
SHOWN="$SUMS"
SUMS="$RECORDED"

recorded_rustc=$(grep '^# rustc' "$SUMS" | sed 's/^# //')
current_rustc=$(rustc --version)
if [ "$recorded_rustc" != "$current_rustc" ]; then
  echo "toolchain differs from the one that produced $SHOWN:" >&2
  echo "  recorded: $recorded_rustc" >&2
  echo "  current:  $current_rustc" >&2
  echo "a hash mismatch below is expected, not tampering" >&2
fi

# Sums from before this line was recorded were all built on macOS.
recorded_family=$(sed -n 's/^# built on: //p' "$SUMS")
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) current_family=windows ;;
  *) current_family=unix ;;
esac
if [ "${recorded_family:-unix}" != "$current_family" ]; then
  echo "$SHOWN was built on ${recorded_family:-unix}, this is $current_family:" >&2
  echo "  Windows embeds \\-joined source paths, so a hash mismatch below is" >&2
  echo "  expected, not tampering; verify on macOS or Linux (WSL2 works)" >&2
fi

# A clean target for each guest: an incremental rebuild would compare an
# artifact against itself and pass without proving anything.
for guest in screensavers arcades; do
  (cd "$guest" && cargo clean -q --release --target wasm32-unknown-unknown 2>/dev/null || true)
done
(cd csvedit && cargo clean -q --release --target wasm32-wasip1 2>/dev/null || true)
sh ./package.sh >/dev/null

fail=0
while read -r expected name; do
  case "$expected" in \#*) continue ;; esac
  [ -n "$name" ] || continue
  guest=$(echo "$name" | sed 's/\.wasm$//')
  actual=$(cd "dist/$guest/wasm" && sha256 "$name" | cut -d' ' -f1)
  if [ "$actual" = "$expected" ]; then
    echo "ok       $name"
  else
    echo "MISMATCH $name" >&2
    echo "  expected $expected" >&2
    echo "  actual   $actual" >&2
    fail=1
  fi
done < "$SUMS"
exit "$fail"
