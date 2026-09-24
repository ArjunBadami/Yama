#!/usr/bin/env bash
# Create (or restart) the training VM. Training starts automatically via startup.sh.
#
#   scripts/gcp/03_create_vm.sh                 # spot L4, config from env.sh
#   RUN_NAME=full-v1 TRAIN_CONFIG=configs/full.yaml scripts/gcp/03_create_vm.sh
#
# If the VM already exists it is (re)started with updated metadata, which is
# how you launch a second run on the same machine.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

META="viveka-bucket=$BUCKET,viveka-run-name=$RUN_NAME,viveka-train-config=$TRAIN_CONFIG,viveka-shutdown-when-done=$SHUTDOWN_WHEN_DONE,install-nvidia-driver=True"

if gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  info "VM $VM_NAME exists; updating metadata and starting"
  gcloud compute instances add-metadata "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
    --metadata "$META" --metadata-from-file startup-script=scripts/gcp/startup.sh
  STATUS="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --format='value(status)')"
  if [ "$STATUS" = "RUNNING" ]; then
    warn "VM is already running; re-running the startup script over SSH"
    gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "sudo google_metadata_script_runner startup" &
  else
    gcloud compute instances start "$VM_NAME" --zone "$ZONE" --project "$PROJECT"
  fi
else
  info "creating $VM_NAME ($MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE, spot=$SPOT) in $ZONE"
  ARGS=(
    --project "$PROJECT" --zone "$ZONE"
    --machine-type "$MACHINE_TYPE"
    --accelerator "type=$GPU_TYPE,count=$GPU_COUNT"
    --image-project "$IMAGE_PROJECT" --image-family "$IMAGE_FAMILY"
    --boot-disk-size "${BOOT_DISK_GB}GB" --boot-disk-type pd-balanced
    --maintenance-policy TERMINATE
    --service-account "$SA_EMAIL" --scopes cloud-platform
    --metadata "$META"
    --metadata-from-file startup-script=scripts/gcp/startup.sh
  )
  if [ "$SPOT" = "true" ]; then
    # STOP on preemption keeps the disk; restarting the VM re-runs startup.sh and resumes training.
    ARGS+=(--provisioning-model SPOT --instance-termination-action STOP)
  fi
  gcloud compute instances create "$VM_NAME" "${ARGS[@]}"
fi

cat <<EOF

Training is starting on the VM. Follow along with:
  scripts/gcp/04_logs.sh                       # tail the training log
  gcloud storage ls $BUCKET/runs/$RUN_NAME/    # checkpoints as they land

The VM stops itself when done (SHUTDOWN_WHEN_DONE=$SHUTDOWN_WHEN_DONE). Fetch results with:
  scripts/gcp/05_fetch_results.sh
EOF
