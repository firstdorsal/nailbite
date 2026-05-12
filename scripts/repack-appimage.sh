#!/bin/bash
# Inject additional shared libraries into an AppImage.
#
# Usage:
#   repack-appimage <appimage> <library> [symlink...]
#       Inject a single library, optionally with symlinks pointing at it.
#   repack-appimage <appimage> --dir <source-dir> [glob...]
#       Inject every file matched by the given globs (default: *.so*) from
#       <source-dir>. Symlinks within <source-dir> are preserved as symlinks
#       in the AppImage so libonnxruntime.so → libonnxruntime.so.1 → real
#       file works exactly as upstream intended.
#
# The AppImage is extracted, the libraries are copied into usr/lib/, optional
# symlinks are created pointing to the library basename, and the AppImage is
# repacked in place.
set -euo pipefail

APPIMAGE="$1"; shift

MODE="single"
SOURCE_DIR=""
declare -a GLOBS=()
declare -a SYMLINKS=()
LIBRARY=""

if [ "${1:-}" = "--dir" ]; then
    MODE="dir"
    shift
    SOURCE_DIR="$1"; shift
    GLOBS=("$@")
    if [ ${#GLOBS[@]} -eq 0 ]; then
        GLOBS=("*.so*")
    fi
else
    LIBRARY="$1"; shift
    SYMLINKS=("$@")
fi

echo "Repacking AppImage: $APPIMAGE"

cd "$(dirname "$APPIMAGE")"
APPIMAGE="$(basename "$APPIMAGE")"

# Find the real squashfs offset (skip false hsqs matches in code sections)
for candidate in $(LC_ALL=C grep -obaP 'hsqs' "$APPIMAGE" | cut -d: -f1); do
    tail -c +$((candidate + 1)) "$APPIMAGE" > _payload.squashfs
    if unsquashfs -s _payload.squashfs >/dev/null 2>&1; then
        OFFSET=$candidate
        break
    fi
    rm -f _payload.squashfs
done

if [ -z "${OFFSET:-}" ]; then
    echo "ERROR: Could not find valid squashfs in $APPIMAGE" >&2
    exit 1
fi

echo "  Squashfs offset: $OFFSET"

# Extract runtime header and squashfs
head -c "$OFFSET" "$APPIMAGE" > _runtime
unsquashfs -d _appdir _payload.squashfs
rm _payload.squashfs

mkdir -p _appdir/usr/lib

if [ "$MODE" = "single" ]; then
    LIB_BASENAME=$(basename "$LIBRARY")
    echo "  Injecting: $LIB_BASENAME"
    cp "$LIBRARY" "_appdir/usr/lib/$LIB_BASENAME"
    for link in "${SYMLINKS[@]}"; do
        ln -sf "$LIB_BASENAME" "_appdir/usr/lib/$link"
    done
else
    echo "  Injecting libraries from: $SOURCE_DIR"
    cd "$SOURCE_DIR"
    declare -a copied=()
    for g in "${GLOBS[@]}"; do
        for f in $g; do
            [ -e "$f" ] || continue
            copied+=("$f")
        done
    done
    if [ ${#copied[@]} -eq 0 ]; then
        echo "ERROR: No files matched globs ${GLOBS[*]} in $SOURCE_DIR" >&2
        exit 1
    fi
    # Use cp -a to preserve symlinks. Sort ensures determinism.
    IFS=$'\n' SORTED=($(printf '%s\n' "${copied[@]}" | sort -u))
    for f in "${SORTED[@]}"; do
        echo "    + $f"
    done
    cp -a "${SORTED[@]}" "$OLDPWD/_appdir/usr/lib/"
    cd "$OLDPWD"
fi

# Repack: use same compression as original where possible
mksquashfs _appdir _payload.squashfs -root-owned -noappend -comp gzip -b 256K
cat _runtime _payload.squashfs > "$APPIMAGE"
chmod +x "$APPIMAGE"
rm -rf _appdir _runtime _payload.squashfs

echo "  Done. New size: $(du -h "$APPIMAGE" | cut -f1)"
