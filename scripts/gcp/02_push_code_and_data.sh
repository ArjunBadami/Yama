#!/usr/bin/env bash
# Upload the code tarball and the processed dataset to GCS.
# Re-run whenever code or data changes; the VM pulls both at boot.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

[ -f "$DATA_DIR/train.jsonl" ] || die "no $DATA_DIR/train.jsonl; build data first (see README)"

info "packaging code"
TMP="$(mktemp -d)"
tar -czf "$TMP/viveka.tar.gz" \
  --exclude='.venv' --exclude='runs' --exclude='data' --exclude='.git' \
  --exclude='__pycache__' --exclude='*.egg-info' --exclude='.cache' \
  pyproject.toml README.md configs src scripts
gcloud storage cp "$TMP/viveka.tar.gz" "$BUCKET/code/viveka.tar.gz"
rm -rf "$TMP"

info "syncing $DATA_DIR -> $BUCKET/data/processed"
gcloud storage rsync -r "$DATA_DIR" "$BUCKET/data/processed"

info "done"
gcloud storage ls -l "$BUCKET/code/" "$BUCKET/data/processed/"
