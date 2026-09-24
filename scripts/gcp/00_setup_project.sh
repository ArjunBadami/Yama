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

info "billing budget (\$$BUDGET_USD/month, alerts at 50/90/100%)"
BILLING_ACCOUNT="$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' 2>/dev/null || true)"
if [ -z "$BILLING_ACCOUNT" ]; then
  warn "could not read the billing account (need roles/billing.viewer); create a budget manually:"
  warn "  https://console.cloud.google.com/billing/budgets?project=$PROJECT"
elif gcloud billing budgets list --billing-account="${BILLING_ACCOUNT#billingAccounts/}" \
       --filter="displayName=viveka-$PROJECT" --format='value(name)' 2>/dev/null | grep -q .; then
  info "budget viveka-$PROJECT exists"
else
  gcloud services enable billingbudgets.googleapis.com --project "$PROJECT" >/dev/null 2>&1 || true
  if gcloud billing budgets create --billing-account="${BILLING_ACCOUNT#billingAccounts/}" \
       --display-name="viveka-$PROJECT" --budget-amount="${BUDGET_USD}USD" \
       --filter-projects="projects/$PROJECT" \
       --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 >/dev/null 2>&1; then
    info "budget created; alerts go to billing admins/users by email"
  else
    warn "budget creation failed (needs roles/billing.costsManager on the billing account). Create it manually:"
    warn "  https://console.cloud.google.com/billing/budgets?project=$PROJECT"
  fi
fi

info "done. next: scripts/gcp/01_check_quota.sh"
