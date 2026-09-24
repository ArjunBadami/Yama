#!/usr/bin/env bash
# VM startup script. Runs at EVERY boot, including after a spot preemption restart.
#
# Executes a queue of runs ("<name>:<config>;<name>:<config>;...") in order. A run with a
# DONE marker in GCS is skipped; an unfinished run resumes from <run>/last/ if present.
# So re-starting the VM after a preemption always continues where it left off.
#
# Configuration comes from instance metadata (set by 03_create_vm.sh):
#   viveka-bucket, viveka-run-queue, viveka-shutdown-when-done,
#   viveka-data-sources, viveka-data-limit, viveka-eval-limit, viveka-data-pos-rate
set -uo pipefail
LOG=/var/log/viveka-train.log
exec > >(tee -a "$LOG") 2>&1
echo "===== viveka startup $(date -u +%FT%TZ) ====="

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET="$(md viveka-bucket)"
RUN_QUEUE="$(md viveka-run-queue)"
SHUTDOWN="$(md viveka-shutdown-when-done || echo true)"
ZONE="$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')"
NAME="$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)"
[ -n "$BUCKET" ] && [ -n "$RUN_QUEUE" ] || { echo "missing viveka-bucket / viveka-run-queue metadata"; exit 1; }

IFS=';' read -ra QUEUE <<< "$RUN_QUEUE"
echo "queue (${#QUEUE[@]}):"; for e in "${QUEUE[@]}"; do echo "  ${e%%:*}  <-  ${e#*:}"; done

finish() {
  [ -n "${WATCHDOG_PID:-}" ] && kill "$WATCHDOG_PID" 2>/dev/null
  if [ "$SHUTDOWN" = "true" ]; then
    echo "stopping VM $NAME ($ZONE)"
    gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet || true
  fi
}

# --- hard cost cap: stop the VM after MAX_HOURS no matter what ------------
# Independent of training success/failure/hangs. Preemption restarts reset the clock,
# but each boot is capped, so spend is bounded per boot.
MAX_HOURS="$(md viveka-max-hours || echo 8)"
(
  sleep "$(awk "BEGIN{print int($MAX_HOURS*3600)}")"
  echo "!!! watchdog: $MAX_HOURS hours elapsed since boot; forcing VM stop $(date -u +%FT%TZ)"
  gcloud storage cp "$LOG" "$BUCKET/watchdog-$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null || true
  gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet
) &
WATCHDOG_PID=$!
echo "watchdog armed: VM will stop after $MAX_HOURS h (pid $WATCHDOG_PID)"

is_done() { gcloud storage ls "$BUCKET/runs/$1/DONE" >/dev/null 2>&1; }

# Anything left to do? If not, don't even wait for the GPU.
PENDING=0
for e in "${QUEUE[@]}"; do is_done "${e%%:*}" || PENDING=$((PENDING + 1)); done
if [ "$PENDING" -eq 0 ]; then
  echo "all runs in the queue already finished; nothing to do"
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
$PY -m pip install -q -e . || { echo "pip install failed"; finish; exit 1; }
# pip upgrades torch, which breaks the DLVM's torchvision/torchaudio. Transformers
# imports both while loading ModernBERT. Reinstall the three as one matched set.
$PY -m pip install -q --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu129 \
  || { echo "torch stack reinstall failed"; finish; exit 1; }
$PY -c "import torch; from transformers.models.modernbert.modeling_modernbert import ModernBertModel; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('modernbert import ok', ModernBertModel.__name__)" \
  || { echo "torch/modernbert import failed"; finish; exit 1; }

# --- data ------------------------------------------------------------------
mkdir -p data/processed
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

# --- run the queue ---------------------------------------------------------
FAILED=0
for entry in "${QUEUE[@]}"; do
  RUN_NAME="${entry%%:*}"; TRAIN_CONFIG="${entry#*:}"; RUN_GCS="$BUCKET/runs/$RUN_NAME"
  if is_done "$RUN_NAME"; then echo "--- $RUN_NAME already done; skipping"; continue; fi
  echo "===== run $RUN_NAME ($TRAIN_CONFIG) start $(date -u +%FT%TZ) ====="

  # resumes automatically if runs/<run>/last exists in GCS
  $PY -m viveka.train --config "$TRAIN_CONFIG" \
    "run_name=$RUN_NAME" "output.dir=runs/$RUN_NAME" "output.gcs_dir=$RUN_GCS"
  STATUS=$?
  if [ $STATUS -ne 0 ]; then
    echo "!!! $RUN_NAME: training exited with $STATUS; continuing with the next run"
    gcloud storage cp "$LOG" "$RUN_GCS/startup.log" || true
    FAILED=$((FAILED + 1)); continue
  fi

  if [ -f data/processed/eval.jsonl ]; then
    $PY -m viveka.evaluate --ckpt "runs/$RUN_NAME/best" --data data/processed/eval.jsonl \
      --out "runs/$RUN_NAME/eval_report.json" --dump-predictions "runs/$RUN_NAME/eval_preds.jsonl" \
      --batch-size 64
  fi
  gcloud storage rsync -r "runs/$RUN_NAME" "$RUN_GCS"
  gcloud storage cp "$LOG" "$RUN_GCS/startup.log" || true
  date -u +%FT%TZ | gcloud storage cp - "$RUN_GCS/DONE"
  echo "===== run $RUN_NAME done $(date -u +%FT%TZ) ====="
done

if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED run(s) failed. Logs are in $BUCKET/runs/<name>/startup.log. Stopping the VM."
  finish
  exit 1
fi
echo "===== queue complete $(date -u +%FT%TZ) ====="
finish
