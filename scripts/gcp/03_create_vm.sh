#!/usr/bin/env bash
# Create (or restart) the training VM. The run queue starts automatically via startup.sh.
#
#   scripts/gcp/03_create_vm.sh                                   # default queue: A -> B -> C
#   RUN_QUEUE="full-v2:configs/full.yaml" scripts/gcp/03_create_vm.sh   # a single run
#   RUN_QUEUE="a:configs/base.yaml;b:configs/lora.yaml" scripts/gcp/03_create_vm.sh
#
# If the VM already exists it is (re)started with updated metadata. Runs that already
# have a DONE marker in GCS are skipped, so re-launching is always safe.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

# Validate the queue: every entry must be name:config and the config must exist.
IFS=';' read -ra QUEUE <<< "$RUN_QUEUE"
[ "${#QUEUE[@]}" -gt 0 ] || die "RUN_QUEUE is empty"
for entry in "${QUEUE[@]}"; do
  name="${entry%%:*}"; cfg="${entry#*:}"
  [ -n "$name" ] && [ "$name" != "$entry" ] || die "bad RUN_QUEUE entry '$entry' (want name:config)"
  [ -f "$cfg" ] || die "config '$cfg' for run '$name' not found"
done

# gcloud splits --metadata on commas, so the queue uses ';' between entries.
META="viveka-bucket=$BUCKET,viveka-run-queue=$RUN_QUEUE,viveka-shutdown-when-done=$SHUTDOWN_WHEN_DONE,viveka-max-hours=$MAX_VM_HOURS"
META="$META,viveka-data-sources=${DATA_SOURCES// /+},viveka-data-limit=$DATA_LIMIT,viveka-eval-limit=$EVAL_LIMIT,viveka-data-pos-rate=$DATA_POS_RATE"
META="$META,install-nvidia-driver=True"

if resolve_vm_zone; then
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
  # Always walk $ZONES. A ZONE left over in the shell (for example us-central1-a from an
  # earlier run) must not pin us to the zone that just stocked out.
  read -ra TRY_ZONES <<< "$ZONES"
  CREATED=""
  for z in "${TRY_ZONES[@]}"; do
    info "creating $VM_NAME ($MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE, spot=$SPOT) in $z"
    ARGS=(
      --project "$PROJECT" --zone "$z"
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
    if gcloud compute instances create "$VM_NAME" "${ARGS[@]}"; then
      ZONE="$z"; CREATED=1; break
    fi
    warn "no capacity in $z; trying the next zone"
  done
  if [ -z "$CREATED" ]; then
    die "no zone had capacity. Retry later, or run: SPOT=false $0"
  fi
fi

cat <<EOF

The VM will run, in order:
EOF
for entry in "${QUEUE[@]}"; do printf '  %-14s %s\n' "${entry%%:*}" "${entry#*:}"; done
cat <<EOF

Follow along with:
  scripts/gcp/04_logs.sh                  # tail the training log
  gcloud storage ls $BUCKET/runs/         # one folder per run; DONE marker when finished

The VM stops itself after the queue (SHUTDOWN_WHEN_DONE=$SHUTDOWN_WHEN_DONE), and unconditionally
after MAX_VM_HOURS=$MAX_VM_HOURS hours (~\$$(awk "BEGIN{printf \"%.2f\", $MAX_VM_HOURS*0.35}") worst case on spot). Fetch results with:
  scripts/gcp/05_fetch_results.sh
If the VM is preempted, just run this script again; finished runs are skipped and the
in-progress run resumes from its last checkpoint.
EOF
