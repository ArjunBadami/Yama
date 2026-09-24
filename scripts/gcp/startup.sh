#!/usr/bin/env bash
# VM startup script. Runs at EVERY boot, including after a spot preemption restart.
# Idempotent: if the run already finished (DONE marker in GCS) it does nothing;
# otherwise it pulls code + data, and training resumes from <run>/last/ if present.
#
# Configuration comes from instance metadata (set by 02_create_vm.sh):
#   viveka-bucket, viveka-run-name, viveka-train-config, viveka-shutdown-when-done
set -uo pipefail
LOG=/var/log/viveka-train.log
exec > >(tee -a "$LOG") 2>&1
echo "===== viveka startup $(date -u +%FT%TZ) ====="

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET="$(md viveka-bucket)"
RUN_NAME="$(md viveka-run-name)"
TRAIN_CONFIG="$(md viveka-train-config)"
SHUTDOWN="$(md viveka-shutdown-when-done || echo true)"
ZONE="$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"
NAME="$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)"
RUN_GCS="$BUCKET/runs/$RUN_NAME"

finish() {
  if [ "$SHUTDOWN" = "true" ]; then
    echo "stopping VM $NAME ($ZONE)"
    gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet || true
  fi
}

if gcloud storage ls "$RUN_GCS/DONE" >/dev/null 2>&1; then
  echo "run $RUN_NAME already finished; nothing to do"
  finish; exit 0
fi

# --- wait for the GPU driver (DLVM installs it on first boot) -------------
for i in $(seq 1 60); do
  if nvidia-smi >/dev/null 2>&1; then break; fi
  echo "waiting for nvidia driver... ($i)"; sleep 10
done
nvidia-smi || { echo "no GPU driver after 10 minutes"; exit 1; }

# --- code ------------------------------------------------------------------
mkdir -p /opt/viveka && cd /opt/viveka
gcloud storage cp "$BUCKET/code/viveka.tar.gz" /tmp/viveka.tar.gz
tar -xzf /tmp/viveka.tar.gz -C /opt/viveka
PY="$(command -v python3)"
if [ -x /opt/conda/bin/python ]; then PY=/opt/conda/bin/python; fi
echo "python: $PY ($($PY --version))"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -e . || { echo "pip install failed"; exit 1; }
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- data ------------------------------------------------------------------
mkdir -p data/processed
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/hf-cache
if gcloud storage ls "$BUCKET/data/processed/train.jsonl" >/dev/null 2>&1; then
  gcloud storage rsync -r "$BUCKET/data/processed" data/processed
else
  SOURCES="$(md viveka-data-sources || echo hotpotqa+squad_v2)"; SOURCES="${SOURCES//+/ }"
  LIMIT="$(md viveka-data-limit || echo 30000)"
  EVAL_LIMIT="$(md viveka-eval-limit || echo 2000)"
  POS_RATE="$(md viveka-data-pos-rate || echo 0.5)"
  echo "no dataset in $BUCKET/data/processed; building on the VM from: $SOURCES (limit=$LIMIT eval_limit=$EVAL_LIMIT)"
  # shellcheck disable=SC2086
  $PY -m viveka.datasets.build --sources $SOURCES --limit "$LIMIT" --eval-limit "$EVAL_LIMIT" \
    --target-pos-rate "$POS_RATE" --out data/processed || { echo "dataset build failed"; exit 1; }
  gcloud storage rsync -r data/processed "$BUCKET/data/processed"
  echo "dataset uploaded to $BUCKET/data/processed"
fi
[ -f data/processed/train.jsonl ] || { echo "no train.jsonl after data step"; exit 1; }

# --- train (resumes automatically if runs/<run>/last exists in GCS) --------
$PY -m viveka.train --config "$TRAIN_CONFIG" \
  "run_name=$RUN_NAME" "output.dir=runs/$RUN_NAME" "output.gcs_dir=$RUN_GCS"
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "training exited with $STATUS; leaving VM up for debugging (set viveka-shutdown-when-done=false to keep)"
  gcloud storage cp "$LOG" "$RUN_GCS/startup.log" || true
  exit $STATUS
fi

# --- evaluate on the held-out automatic eval set ---------------------------
if [ -f data/processed/eval.jsonl ]; then
  $PY -m viveka.evaluate --ckpt "runs/$RUN_NAME/best" --data data/processed/eval.jsonl \
    --out "runs/$RUN_NAME/eval_report.json" --dump-predictions "runs/$RUN_NAME/eval_preds.jsonl" \
    --batch-size 64
fi
gcloud storage rsync -r "runs/$RUN_NAME" "$RUN_GCS"
gcloud storage cp "$LOG" "$RUN_GCS/startup.log" || true
date -u +%FT%TZ | gcloud storage cp - "$RUN_GCS/DONE"
echo "===== done $(date -u +%FT%TZ) ====="
finish
