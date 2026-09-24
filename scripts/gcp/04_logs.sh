#!/usr/bin/env bash
# Tail the training log on the VM (falls back to serial console if SSH isn't up yet).
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh
resolve_vm_zone || die "no VM named $VM_NAME. Create one with scripts/gcp/03_create_vm.sh"
info "VM is in $ZONE"

if gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --quiet \
     --command "sudo tail -n 50 -f /var/log/viveka-train.log" 2>/dev/null; then
  exit 0
fi
warn "ssh not available yet; showing serial console output"
gcloud compute instances get-serial-port-output "$VM_NAME" --zone "$ZONE" --project "$PROJECT" | tail -n 100
