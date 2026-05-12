#!/bin/bash
# Nailbite Version Bump Script
#
# Updates version across all project files to maintain consistency.
#
# Usage:
#   bash scripts/bump-version.sh 0.2.0
#
# Files updated:
#   - src-tauri/Cargo.toml
#   - src-tauri/tauri.conf.json
#   - package.json
#
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/bump-version.sh <version>"
    echo "Example: bash scripts/bump-version.sh 0.2.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (semver)
if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "Error: Invalid version format '${NEW_VERSION}'"
    echo "Expected format: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-prerelease"
    echo "Examples: 0.2.0, 1.0.0, 0.2.0-beta.1"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep -m1 '^version = ' src-tauri/Cargo.toml | sed 's/version = "\(.*\)"/\1/')

echo "=========================================="
echo "Nailbite Version Bump"
echo "Current version: ${CURRENT_VERSION}"
echo "New version: ${NEW_VERSION}"
echo "=========================================="

# Update src-tauri/Cargo.toml
echo "Updating src-tauri/Cargo.toml..."
sed -i "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" src-tauri/Cargo.toml

# Update src-tauri/tauri.conf.json
echo "Updating src-tauri/tauri.conf.json..."
sed -i "s/\"version\": \"${CURRENT_VERSION}\"/\"version\": \"${NEW_VERSION}\"/" src-tauri/tauri.conf.json

# Update package.json
echo "Updating package.json..."
sed -i "s/\"version\": \"${CURRENT_VERSION}\"/\"version\": \"${NEW_VERSION}\"/" package.json

# Verify updates
echo ""
echo "Verification:"
echo "-------------"

CARGO_VER=$(grep -m1 '^version = ' src-tauri/Cargo.toml | sed 's/version = "\(.*\)"/\1/')
TAURI_VER=$(grep -m1 '"version"' src-tauri/tauri.conf.json | sed 's/.*"version": "\([^"]*\)".*/\1/')
PKG_VER=$(grep -m1 '"version"' package.json | sed 's/.*"version": "\([^"]*\)".*/\1/')

echo "src-tauri/Cargo.toml:     ${CARGO_VER}"
echo "src-tauri/tauri.conf.json: ${TAURI_VER}"
echo "package.json:              ${PKG_VER}"

# Check all versions match
if [ "${CARGO_VER}" = "${NEW_VERSION}" ] && [ "${TAURI_VER}" = "${NEW_VERSION}" ] && [ "${PKG_VER}" = "${NEW_VERSION}" ]; then
    echo ""
    echo "Version bump successful!"
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git diff"
    echo "  2. Commit: git commit -am \"chore(release): v${NEW_VERSION}\""
    echo "  3. Tag: git tag v${NEW_VERSION}"
    echo "  4. Push: git push origin main v${NEW_VERSION}"
else
    echo ""
    echo "Error: Version mismatch detected!"
    exit 1
fi
