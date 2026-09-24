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
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --display-name "Viveka training VM" --project "$PROJECT"
fi
# IAM is eventually consistent: a freshly created SA can be "not found" by policy
# bindings for several seconds. Wait until it is visible, then bind with retries.
for i in $(seq 1 20); do
  gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1 && break
  sleep 3
done
bind_role() {
  local role="$1"
  for attempt in $(seq 1 6); do
    if gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "$role" \
         --condition=None --quiet >/dev/null 2>&1; then
      info "  bound $role"; return 0
    fi
    sleep $((attempt * 3))
  done
  die "could not bind $role to $SA_EMAIL after retries"
}
# storage: data/checkpoints; aiplatform: teacher calls from the VM; logging/monitoring: agent;
# compute.instanceAdmin.v1: lets the VM stop itself when the queue finishes.
for role in roles/storage.objectAdmin roles/aiplatform.user roles/logging.logWriter \
            roles/monitoring.metricWriter roles/compute.instanceAdmin.v1; do
  bind_role "$role"
done

info "billing budget (\$$BUDGET_USD/month, alerts at 50/90/100%)"
# Never let this step block: no prompts, hard timeout, and any failure just prints the console link.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
BUDGET_URL="https://console.cloud.google.com/billing/budgets?project=$PROJECT"
BILLING_ACCOUNT="$(timeout 30 gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' 2>/dev/null || true)"
if [ -z "$BILLING_ACCOUNT" ]; then
  warn "could not read the billing account (need roles/billing.viewer). Create the budget manually: $BUDGET_URL"
elif timeout 60 gcloud billing budgets list --billing-account="${BILLING_ACCOUNT#billingAccounts/}" \
       --filter="displayName=viveka-$PROJECT" --format='value(name)' 2>/dev/null | grep -q .; then
  info "budget viveka-$PROJECT exists"
else
  timeout 60 gcloud services enable billingbudgets.googleapis.com --project "$PROJECT" >/dev/null 2>&1 || true
  if timeout 60 gcloud billing budgets create --billing-account="${BILLING_ACCOUNT#billingAccounts/}" \
       --display-name="viveka-$PROJECT" --budget-amount="${BUDGET_USD}USD" \
       --filter-projects="projects/$PROJECT" \
       --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 >/dev/null 2>&1; then
    info "budget created; alerts go to billing admins/users by email"
  else
    warn "budget not created (needs roles/billing.costsManager on the billing account, or the command timed out)."
    warn "Create it manually (2 minutes): $BUDGET_URL"
  fi
fi
unset CLOUDSDK_CORE_DISABLE_PROMPTS

info "done. next: scripts/gcp/01_check_quota.sh"
