#!/usr/bin/env bash
# Show GPU quota in the chosen region and where to request more.
# GPU quota is the most common blocker for a new project: the default is often 0.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

info "GPU quotas in $REGION (limit / usage)"
gcloud compute regions describe "$REGION" --project "$PROJECT" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -Ei "metric|NVIDIA_L4|PREEMPTIBLE_NVIDIA_L4|NVIDIA_A100|GPUS_ALL_REGIONS" || true

echo
info "project-wide: GPUS_ALL_REGIONS"
gcloud compute project-info describe --project "$PROJECT" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -Ei "metric|GPUS_ALL_REGIONS" || true

cat <<EOF

You need, for a 1x L4 ${SPOT:+spot }VM in $REGION:
  - GPUS_ALL_REGIONS                >= $GPU_COUNT   (project-wide)
  - NVIDIA_L4_GPUS                  >= $GPU_COUNT   (on-demand)   or
  - PREEMPTIBLE_NVIDIA_L4_GPUS      >= $GPU_COUNT   (spot)

If any is 0, request an increase (usually approved within minutes to a day for L4):
  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT
  Filter: "NVIDIA L4" and "GPUs (all regions)", pick region $REGION, edit -> new limit $GPU_COUNT.

Check which zones in the region actually have L4s:
  gcloud compute accelerator-types list --filter="zone~$REGION AND name=$GPU_TYPE" --format="value(zone)"
EOF
