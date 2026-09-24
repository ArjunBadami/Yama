#!/usr/bin/env bash
# Shared settings for the GCP scripts. Source this: `source scripts/gcp/env.sh`
# Override anything by exporting it before sourcing.

export PROJECT="${PROJECT:-propel-dev-486222}"
export REGION="${REGION:-us-central1}"
export ZONE="${ZONE:-us-central1-a}"

# Artifacts bucket (data, code tarballs, runs). Bucket names are global; keep the project suffix.
export BUCKET="${BUCKET:-gs://viveka-${PROJECT}}"

# Training VM
export VM_NAME="${VM_NAME:-viveka-train}"
export MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"    # 8 vCPU / 32GB, comes with 1x L4
export GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
export GPU_COUNT="${GPU_COUNT:-1}"
export BOOT_DISK_GB="${BOOT_DISK_GB:-200}"
export SPOT="${SPOT:-true}"                             # spot VMs are ~60-70% cheaper; training auto-resumes

# Deep Learning VM image. `pytorch-latest-gpu` was retired; pin a real family.
# List current ones with:
#   gcloud compute images list --project deeplearning-platform-release --no-standard-images \
#     --filter="family~pytorch" --format="value(family)" | sort -u
export IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
export IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"

# Service account the VM runs as (created by 00_setup_project.sh)
export SA_NAME="${SA_NAME:-viveka-train}"
export SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

# What to train on the VM: a queue of "<run_name>:<config>" entries separated by ';'.
# Runs execute in order in a single VM boot. Finished runs (DONE marker in GCS) are skipped,
# so re-launching after a preemption continues with the unfinished ones.
# Default is the plan's order: A (frozen) -> B (LoRA) -> C (full).
export RUN_QUEUE="${RUN_QUEUE:-frozen-v1:configs/base.yaml;lora-v1:configs/lora.yaml;full-v1:configs/full.yaml}"
export SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-true}" # stop the VM after the queue to stop billing

# Hard cost caps.
# The VM stops itself MAX_VM_HOURS after boot regardless of what is running (watchdog in startup.sh).
# At ~$0.30/hr spot for g2-standard-8, 8h is ~$2.50 worst case. Raise for longer queues.
export MAX_VM_HOURS="${MAX_VM_HOURS:-8}"
# Billing budget with email alerts at 50/90/100% (created by 00_setup_project.sh; needs billing perms).
export BUDGET_USD="${BUDGET_USD:-50}"

# Dataset. If $BUCKET/data/processed/train.jsonl does not exist when the VM boots,
# the VM builds it from these sources and uploads it (so nothing has to be built locally).
export DATA_DIR="${DATA_DIR:-data/processed}"           # optional local copy to push with 02_push_code_and_data.sh
export DATA_SOURCES="${DATA_SOURCES:-hotpotqa squad_v2}"
export DATA_LIMIT="${DATA_LIMIT:-30000}"                 # source rows per dataset (train split)
export EVAL_LIMIT="${EVAL_LIMIT:-2000}"                  # source rows per dataset (validation split -> eval.jsonl)
export DATA_POS_RATE="${DATA_POS_RATE:-0.5}"             # downsample train to this positive rate

# Colour helpers
info()  { printf '\033[1;34m[viveka]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[viveka]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[viveka]\033[0m %s\n' "$*" >&2; exit 1; }
