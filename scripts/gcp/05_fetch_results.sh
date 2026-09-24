#!/usr/bin/env bash
# Pull a finished run (best checkpoint, reports, metrics) from GCS to ./runs/<RUN_NAME>.
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

SRC="$BUCKET/runs/$RUN_NAME"
DST="runs/$RUN_NAME"
mkdir -p "$DST"
info "$SRC -> $DST"
# Skip the resumable trainer state (optimizer etc.); only needed on the VM.
gcloud storage rsync -r "$SRC" "$DST" --exclude='.*trainer_state\.pt$'

echo
if [ -f "$DST/eval_report.json" ]; then
  info "eval summary"
  python - "$DST/eval_report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"n={r['n']} logloss={r['log_loss']:.4f} brier={r['brier']:.4f} auroc={r['auroc']:.4f} ece={r['ece']:.4f} T={r.get('temperature',1):.3f}")
s = r["selective"]
print(f"selective hi={s['hi']} lo={s['lo']}: coverage={s['coverage']:.3f} false_suff_among_calls={s['false_sufficient_among_calls']:.4f}")
for c, m in r.get("per_category", {}).items():
    print(f"  {c:<24} n={m['n']:<6} acc={m['accuracy@0.5']:.3f} logloss={m['log_loss']:.3f}")
EOF
fi
info "predict locally with: viveka-predict --ckpt $DST/best --question '...' --evidence '...' '...'"
