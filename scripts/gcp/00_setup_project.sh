#!/usr/bin/env bash
# One-time project setup: APIs, bucket, service account + roles.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

gcloud config set project "$PROJECT" >/dev/null

info "enabling APIs"
gcloud services enable compute.googleapis.com storage.googleapis.com aiplatform.googleapis.com \
  logging.googleapis.com iam.googleapis.com --project "$PROJECT"

info "bucket $BUCKET"
if ! gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access
else
  info "bucket exists"
fi

info "service account $SA_EMAIL"
if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --display-name "Viveka training VM" --project "$PROJECT"
fi
for role in roles/storage.objectAdmin roles/aiplatform.user roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "$role" \
    --condition=None --quiet >/dev/null
done
# Allow the VM to stop itself when training finishes.
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" \
  --role roles/compute.instanceAdmin.v1 --condition=None --quiet >/dev/null

info "done. next: scripts/gcp/01_check_quota.sh"
