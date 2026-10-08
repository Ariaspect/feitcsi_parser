#!/usr/bin/env bash
# Build the LG-board bundle of the presence calculators: capture in, 1/0 out.
#
#   scripts/build_board_bundle.sh             # -> dist/board-calc/ and dist/board-calc.tar.gz
#
#   af8  CAPTURE.bin [--json]   the A+F 8-feature classifier (backend.af8) -- the verdict
#   h2   CAPTURE.bin [--json]   hybrid 2's one-minute range rule (backend.hybrid2_calc)
#
# Code only, ~260 KB. The board's Python is a 32-bit ARM build
# (cpython-312-arm-linux-gnueabi) on a 64-bit kernel, and it already carries
# NumPy 1.26.4 for that ABI -- the one NumPy that can load into it, and the one
# LG's detector needs. The calculators run on it; no NumPy is shipped. (An
# aarch64 wheel cannot load into a 32-bit interpreter, and PyPI has no 32-bit
# ARM NumPy 2.)
#
#   backend/   the eleven modules the calculators import, copied rather than the
#              whole package -- the board has no scipy and no CSIKit, and a
#              module that reaches for either would only fail there. Copying
#              the list (and testing the copy) is what proves the list.
#   af8, h2    the entry points: the bundle on PYTHONPATH, the system NumPy.
#   MANIFEST   the commit it was built from.
#   SHA256SUMS every file, for `sha256sum -c SHA256SUMS` on the board.
#
# Nothing here touches the board; see docs/board_calculator.md for the install.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$REPO/dist/board-calc}

MODULES=(__init__ mtk index presence farsense framediff hybrid hybrid2 doppler hybrid2_calc af8)

rm -rf "$OUT"
mkdir -p "$OUT/backend"
for m in "${MODULES[@]}"; do
    cp "$REPO/backend/$m.py" "$OUT/backend/"
done

entry() {   # entry NAME MODULE WHAT
    cat >"$OUT/$1" <<EOF
#!/bin/sh
# $3:  $1 CAPTURE.bin [--json]  ->  1 | 0
# -s skips the user site, -P keeps the working directory off sys.path, so the
# only backend this sees is the bundle's. OPENBLAS_NUM_THREADS=1: one core, the
# way it was timed, leaving the rest to the board.
HERE=\$(cd "\$(dirname "\$0")" && pwd)
OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-1} PYTHONPATH="\$HERE" exec python3 -s -P -m backend.$2 "\$@"
EOF
    chmod +x "$OUT/$1"
}
entry af8 af8 "AF8 presence verdict (A+F 8 features, logistic)"
entry h2 hybrid2_calc "Hybrid 2 one-minute range rule"

{
    echo "commit   $(git -C "$REPO" rev-parse HEAD)$(git -C "$REPO" diff --quiet -- backend scripts || echo ' (dirty)')"
    echo "built    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "numpy    the board's own (1.26.4, cpython-312-arm-linux-gnueabi)"
} >"$OUT/MANIFEST"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum >SHA256SUMS)

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "$OUT  ($(du -sh "$OUT" | cut -f1) unpacked, $(du -h "$OUT.tar.gz" | cut -f1) packed)"
