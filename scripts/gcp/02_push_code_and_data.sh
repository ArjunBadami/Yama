#!/usr/bin/env bash
# Upload the code tarball (and, if present, a locally built dataset) to GCS.
# Run from Cloud Shell inside the repo checkout. Re-run whenever code changes.
#
# The dataset is optional here: if $BUCKET/data/processed/train.jsonl is absent when
# the VM boots, startup.sh builds it on the VM from $DATA_SOURCES and uploads it.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

info "packaging code"
TMP="$(mktemp -d)"
tar -czf "$TMP/viveka.tar.gz" \
  --exclude='.venv' --exclude='runs' --exclude='data' --exclude='.git' \
  --exclude='__pycache__' --exclude='*.egg-info' --exclude='.cache' \
  pyproject.toml README.md configs src scripts
gcloud storage cp "$TMP/viveka.tar.gz" "$BUCKET/code/viveka.tar.gz"
rm -rf "$TMP"

if [ -f "$DATA_DIR/train.jsonl" ]; then
  info "syncing $DATA_DIR -> $BUCKET/data/processed"
  gcloud storage rsync -r "$DATA_DIR" "$BUCKET/data/processed"
elif gcloud storage ls "$BUCKET/data/processed/train.jsonl" >/dev/null 2>&1; then
  info "no local dataset; using the one already in $BUCKET/data/processed"
else
  warn "no dataset locally or in GCS; the VM will build one from: $DATA_SOURCES (limit=$DATA_LIMIT)"
fi

info "done"
gcloud storage ls -l "$BUCKET/code/"
gcloud storage ls "$BUCKET/data/processed/" 2>/dev/null || true
