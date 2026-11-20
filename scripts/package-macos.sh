#!/bin/bash
set -euo pipefail
umask 022
export PATH="$HOME/.cargo/bin:$PATH"

cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Build on native Apple Silicon macOS.' >&2; exit 1; }
export MACOSX_DEPLOYMENT_TARGET=14.0
export CARGO_TARGET_DIR="$PWD/target"
version=$(awk -F '"' '/^version = / {print $2; exit}' Cargo.toml)
[[ "$version" =~ ^[0-9]+[.][0-9]+[.][0-9]+$ ]] || { echo 'Invalid release version.' >&2; exit 1; }
out="$PWD/target/packages"
name="LiteInfer-$version-macOS-arm64.pkg"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cargo_root="$HOME/.cargo"
if [[ -n "${CARGO_HOME:-}" ]]; then cargo_root="$CARGO_HOME"; fi

CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_PANIC=unwind CARGO_PROFILE_RELEASE_LTO=false \
    CFLAGS="-ffile-prefix-map=$PWD=/liteinfer -ffile-prefix-map=$cargo_root=/cargo" \
    RUSTFLAGS="--remap-path-prefix=$PWD=/liteinfer --remap-path-prefix=$cargo_root=/cargo" \
    cargo build --locked --release --target aarch64-apple-darwin

bin="$stage/root/usr/local/bin"
share="$stage/root/usr/local/share/liteinfer"
mkdir -p "$bin" "$share" "$out"
install -m 755 "$CARGO_TARGET_DIR/aarch64-apple-darwin/release/liteinfer" "$bin/liteinfer"
/usr/bin/strip -x "$bin/liteinfer"
install -m 644 LICENSE NOTICE THIRD_PARTY_NOTICES.md assets/LUCIDE-LICENSE.txt docs/install-macos.md "$share/"
install -m 644 licenses/objc2-LICENSE.md "$share/"
python3 scripts/generate-third-party-licenses.py "$PWD" "$share/THIRD_PARTY_RUST_LICENSES.md"
cp "$(rustc --print sysroot)/share/doc/rust/COPYRIGHT-library.html" "$share/RUST-STANDARD-LIBRARY-COPYRIGHT.html"
install -m 644 examples/config.toml "$share/config.example.toml"
xattr -cr "$stage/root"

/usr/bin/pkgbuild --root "$stage/root" --install-location / --ownership recommended \
    --filter '(^|/)[.]_.*$' --identifier io.github.zxcasd01-d.liteinfer \
    --version "$version" "$stage/gateway.pkg"
cat > "$stage/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>LiteInfer $version</title>
  <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
  <volume-check><allowed-os-versions><os-version min="14.0"/></allowed-os-versions></volume-check>
  <choices-outline><line choice="default"><line choice="io.github.zxcasd01-d.liteinfer"/></line></choices-outline>
  <choice id="default"/>
  <choice id="io.github.zxcasd01-d.liteinfer" visible="false"><pkg-ref id="io.github.zxcasd01-d.liteinfer"/></choice>
  <pkg-ref id="io.github.zxcasd01-d.liteinfer" version="$version" onConclusion="none">#gateway.pkg</pkg-ref>
</installer-gui-script>
XML
/usr/bin/productbuild --distribution "$stage/distribution.xml" --package-path "$stage" "$stage/$name"
mv "$stage/$name" "$out/$name"
cd "$out"
/usr/bin/shasum -a 256 "$name" > "$name.sha256"
printf '%s\n' "$out/$name" "$out/$name.sha256"
