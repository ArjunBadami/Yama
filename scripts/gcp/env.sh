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

# Deep Learning VM image. List current families with:
#   gcloud compute images list --project deeplearning-platform-release --no-standard-images \
#     --filter="family~pytorch" --format="value(family)" | sort -u
export IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
export IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"

# Service account the VM runs as (created by 00_setup_project.sh)
export SA_NAME="${SA_NAME:-viveka-train}"
export SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

# What to train on the VM
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lora.yaml}"
export RUN_NAME="${RUN_NAME:-lora-modernbert-base}"
export DATA_DIR="${DATA_DIR:-data/processed}"           # local dir synced to $BUCKET/data/processed
export SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-true}" # stop the VM after training to stop billing

# Colour helpers
info()  { printf '\033[1;34m[viveka]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[viveka]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[viveka]\033[0m %s\n' "$*" >&2; exit 1; }
