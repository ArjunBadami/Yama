#!/usr/bin/env bash
# Delete the training VM (checkpoints live in GCS, so this loses nothing).
# Pass --bucket to also delete the bucket and everything in it.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

if resolve_vm_zone; then
  info "deleting VM $VM_NAME in $ZONE"
  gcloud compute instances delete "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --quiet
else
  info "no VM named $VM_NAME"
fi

if [ "${1:-}" = "--bucket" ]; then
  warn "deleting bucket $BUCKET and all contents"
  gcloud storage rm -r "$BUCKET"
fi
