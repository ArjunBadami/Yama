#!/usr/bin/env bash
# Show GPU quota in the chosen region and where to request more.
# GPU quota is the most common blocker for a new project: the default is often 0.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

show_quotas() {  # $1 = json with a top-level "quotas" list, $2 = substring pattern(s) comma-separated
  python3 -c '
import json, sys
keys = sys.argv[1].split(",")
qs = json.load(sys.stdin).get("quotas", [])
rows = [q for q in qs if any(k in q["metric"] for k in keys)]
for q in rows:
    print("  {:<38} limit={:<7g} usage={:g}".format(q["metric"], q["limit"], q["usage"]))
sys.exit(0 if rows else 1)
' "$2"
}

info "GPU quotas in $REGION (limit / usage)"
gcloud compute regions describe "$REGION" --project "$PROJECT" --format=json | show_quotas - "NVIDIA_L4,NVIDIA_T4,NVIDIA_A100" || true

echo
info "project-wide: GPUS_ALL_REGIONS"
if ! gcloud compute project-info describe --project "$PROJECT" --format=json | show_quotas - "GPUS_ALL_REGIONS"; then
  warn "GPUS_ALL_REGIONS is not listed for this project. That usually means the billing account is"
  warn "still on the Free Trial, which cannot use GPUs at all. Upgrade to a paid account first:"
  warn "  https://console.cloud.google.com/billing?project=$PROJECT  (Upgrade button)"
  warn "then request GPUS_ALL_REGIONS >= 1 at the quotas page."
fi

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
