#!/usr/bin/env bash
# Pull finished runs (best checkpoint, reports, metrics) from GCS to ./runs/<name> and
# print a side-by-side comparison.
#
#   scripts/gcp/05_fetch_results.sh                 # every run in $RUN_QUEUE
#   scripts/gcp/05_fetch_results.sh lora-v1 full-v1 # specific runs
set -euo pipefail
cd "$(dirname "$0")/../.." && source scripts/gcp/env.sh

if [ "$#" -gt 0 ]; then
  RUNS=("$@")
else
  IFS=';' read -ra QUEUE <<< "$RUN_QUEUE"
  RUNS=(); for e in "${QUEUE[@]}"; do RUNS+=("${e%%:*}"); done
fi

FETCHED=()
for name in "${RUNS[@]}"; do
  SRC="$BUCKET/runs/$name"; DST="runs/$name"
  if ! gcloud storage ls "$SRC/" >/dev/null 2>&1; then warn "$name: nothing in $SRC yet"; continue; fi
  if ! gcloud storage ls "$SRC/DONE" >/dev/null 2>&1; then warn "$name: not finished yet (no DONE marker); fetching anyway"; fi
  mkdir -p "$DST"
  info "$SRC -> $DST"
  # Skip the resumable trainer state (optimizer etc.); only needed on the VM.
  gcloud storage rsync -r "$SRC" "$DST" --exclude='.*trainer_state\.pt$'
  FETCHED+=("$DST")
done
[ "${#FETCHED[@]}" -gt 0 ] || die "no runs fetched"

echo
python3 - "${FETCHED[@]}" <<'EOF'
import json, sys, os
rows = []
for d in sys.argv[1:]:
    p = os.path.join(d, "eval_report.json")
    if not os.path.exists(p):
        print(f"{os.path.basename(d)}: no eval_report.json yet"); continue
    r = json.load(open(p)); s = r["selective"]
    rows.append((os.path.basename(d), r["log_loss"], r["brier"], r["auroc"], r["ece"],
                 r["at_threshold"]["0.5"]["accuracy"], s["coverage"], s["false_sufficient_among_calls"], r.get("temperature", 1.0)))
if rows:
    print(f"{'run':<16}{'logloss':>9}{'brier':>8}{'auroc':>8}{'ece':>8}{'acc@.5':>8}{'cover':>8}{'fsuff@hi':>10}{'T':>7}")
    for n, ll, br, au, ece, acc, cov, fs, T in rows:
        print(f"{n:<16}{ll:9.4f}{br:8.4f}{au:8.4f}{ece:8.4f}{acc:8.3f}{cov:8.3f}{fs:10.4f}{T:7.3f}")
    print("\nper-category accuracy@0.5:")
    cats = sorted({c for d in sys.argv[1:] if os.path.exists(os.path.join(d, "eval_report.json"))
                   for c in json.load(open(os.path.join(d, "eval_report.json"))).get("per_category", {})})
    print(f"{'category':<24}" + "".join(f"{os.path.basename(d):>14}" for d in sys.argv[1:]))
    for c in cats:
        line = f"{c:<24}"
        for d in sys.argv[1:]:
            p = os.path.join(d, "eval_report.json")
            m = json.load(open(p)).get("per_category", {}).get(c) if os.path.exists(p) else None
            line += f"{m['accuracy@0.5']:>14.3f}" if m else f"{'-':>14}"
        print(line)
EOF
echo
info "predict with: viveka-predict --ckpt runs/<name>/best --question '...' --evidence '...' '...'"
