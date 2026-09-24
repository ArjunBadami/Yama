# Viveka

A System 1 evidence-sufficiency model:

```
(question, evidence)  ->  P(sufficient)
```

Viveka answers one narrow question for an agent: *does the evidence I have so far contain
enough information for a competent reasoner to answer this question?* It does not answer
the question. It is a small bidirectional encoder (ModernBERT-base) with a single-logit
head, trained with binary cross-entropy and temperature-calibrated so that thresholds like
`P > 0.98` mean something. Background and rationale: `system1_models_jev_laya_gcp_build_plan.md`.

## Architecture

```
question, evidence[]
        │
        ▼
 formatting.py      "question: {q}"  ‖  "evidence:\n- e1\n- e2 ..."      (text pair)
        │
        ▼
 tokenizer          [CLS] question tokens [SEP] evidence tokens [SEP]     (truncates evidence only)
        │
        ▼
 ModernBERT-base    22 layers, d=768, bidirectional attention, 8192-token context
        │           every token attends to every other: the question sees all the evidence
        ▼           and each evidence sentence sees the question and the other sentences
 pooling            h = hidden state at [CLS]   (or mean over non-pad tokens)         ∈ R^768
        │
        ▼
 head               z = w·h + b                  one linear layer, 769 parameters       ∈ R
        │
        ▼
 calibration        z / T                        T fitted post hoc on validation, stored in checkpoint
        │
        ▼
 sigmoid            P(sufficient) = 1 / (1 + e^{-z/T})                                  ∈ (0,1)
```

No decoder, no token generation, no fixed label vocabulary, no variable-size softmax. The
entire task-specific part of the network is the 769-parameter head; everything else is the
pretrained encoder, adapted as little as the data requires.

**Why an encoder.** The output is one number, so there is nothing to generate. A bidirectional
encoder reads the whole input at once and is ~20–100× smaller and cheaper than the decoders
usually asked to produce this judgment as JSON. ModernBERT specifically because it is a modern
(2024) encoder with long context, so distractor-heavy evidence fits without truncation.

**Training modes** (`model.mode` in config). Same architecture, same loss; the difference is
which weights may move.

| mode | trains | trainable params | when |
|---|---|---|---|
| `frozen` | head only | 769 | how much the pretrained representation already knows |
| `lora` | head + rank-16 LoRA adapters on `Wqkv`, `Wo`, `Wi` in every layer | ~3.4M | default; safe on small data, adapter is a few MB |
| `full` | head + all encoder weights | ~150M | max capacity; lower encoder LR (3e-5) than head (1e-3) |

LoRA keeps each pretrained weight matrix \(W\) frozen and learns a low-rank update
\(W x + B A x\) with \(A \in \mathbb{R}^{r \times d}\), \(B \in \mathbb{R}^{d \times r}\), \(r=16\).
At 150M params LoRA vs full is not a hardware decision (both fit on one L4); running both
tells us whether the task needs the encoder to change or only needs a new output interface.

**Loss.** Binary cross-entropy on the raw logit \(z\), optional `pos_weight` for class
imbalance. Model selection by validation log loss. Temperature \(T\) is fitted afterwards by
minimising BCE of \(\sigma(z/T)\) on validation; it rescales confidence without changing
rankings, so accuracy/AUROC are untouched while ECE and thresholds improve.

**Output contract.** `P(sufficient)` is used three ways by an agent: `P ≥ hi` (default 0.98)
→ hand off to the reasoning model; `P < lo` (0.20) → retrieve more; otherwise → uncertain,
escalate. False-sufficient is the expensive error, hence the asymmetric thresholds.

## Layout

```
configs/            base.yaml (frozen), lora.yaml (LoRA, default), full.yaml, smoke.yaml (CPU test)
src/viveka/
  schema.py         Example record + JSONL IO + group-aware splitting
  formatting.py     (question, evidence) -> text pair for the tokenizer
  data.py           torch Dataset / collator
  model/            SufficiencyModel (frozen|lora|full, cls|mean pooling), temperature scaling
  metrics.py        log loss, Brier, AUROC, ECE, reliability bins, per-category, selective thresholds
  datasets/         degrade.py (the curriculum), toy.py, hotpotqa.py, squad_v2.py, build.py (CLI)
  teacher/          Gemini-on-Vertex: label.py (Definition-B labels), synthetic.py (data generation)
  train.py          resumable trainer, GCS mirroring, post-hoc calibration
  evaluate.py       full report on a JSONL file
  inference.py      SufficiencyPredictor + CLI
scripts/gcp/        numbered runbook: project setup -> quota -> push -> VM -> logs -> fetch -> teardown
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
# optional: teacher (Gemini on Vertex)
.\.venv\Scripts\python.exe -m pip install -e ".[teacher]"
```

Smoke test the whole pipeline on CPU (~1 minute, tiny random encoder, offline toy data):

```powershell
python -m viveka.datasets.build --sources toy --toy-questions 400 --out data/processed/toy
python -m viveka.train --config configs/smoke.yaml
python -m viveka.evaluate --ckpt runs/smoke/best --data data/processed/toy/eval.jsonl
python -m viveka.inference --ckpt runs/smoke/best --question "Does Ayesha Khan have API access?" `
  --evidence "Ayesha Khan is on the Team plan." "Team plans permit API access."
```

Re-running `viveka.train` with the same config resumes from `runs/<name>/last/`.

## Data

Every example is:

```json
{"question": "...", "evidence": ["...", "..."], "sufficient": false,
 "source": "hotpotqa", "category": "missing_one_fact", "id": "hotpotqa-<id>::2", "meta": {...}}
```

`datasets/degrade.py` turns one annotated question (gold facts + on-topic distractors) into a
curriculum of labelled variants, so labels are mechanical, not judged:

| variant | label | category | defeats shortcut |
|---|---|---|---|
| all gold, little noise | 1 | obvious_sufficient / short_sufficient / multi_hop | length |
| all gold + heavy relevant noise | 1 | distractor_heavy | length, relevance |
| gold minus one fact | 0 | missing_one_fact | lexical overlap |
| relevant noise only | 0 | relevant_insufficient | relevance |
| lots of relevant noise only | 0 | long_insufficient | length, relevance |
| unrelated noise / empty | 0 | obvious_insufficient | — |

Sources (all public, no auth):

- **HotpotQA** (distractor): supporting sentences are annotated; most questions are two-paragraph
  bridge questions, so dropping one sentence is a natural multi-hop hard negative.
- **SQuAD 2.0**: single-sentence sufficiency, plus ~50k adversarial unanswerable questions that are
  high-overlap `relevant_insufficient` examples.
- **toy**: offline templates for smoke tests.

Build the v1 training set (downloads ~1GB from the Hugging Face hub; a few minutes). You normally
don't run this by hand: the GCP VM builds it on first boot if the bucket has no dataset (see below).
To build locally anyway, e.g. to inspect examples:

```powershell
python -m viveka.datasets.build --sources hotpotqa squad_v2 --limit 30000 --eval-limit 2000 `
  --target-pos-rate 0.5 --out data/processed
```

Splits are group-aware: all variants of one source question stay in the same split. `eval.jsonl`
comes from the datasets' own validation splits. The hand-reviewed Phase-0 eval set from the plan
should be curated separately under `data/eval/`.

### Teacher (Gemini on Vertex AI) — optional, costs API calls

Run from Cloud Shell (already authenticated) or from the VM (service account has `aiplatform.user`):

```bash
export GOOGLE_CLOUD_PROJECT=propel-dev-486222

# Definition-B labels: sufficient iff Gemini, restricted to the evidence, answers correctly k/k times.
python -m viveka.teacher.label --in data/processed/val.jsonl --out data/processed/val_teacher.jsonl --k 3 --limit 500

# Synthetic questions in new domains; the curriculum labels them.
python -m viveka.teacher.synthetic --n 300 --out data/processed/synthetic.jsonl
```

Responses are cached in `data/teacher_cache/` so reruns are free. Use the labeller first as an
*audit* (its stats print heuristic-vs-teacher agreement) before relabelling the training set.

## Training on GCP

Project `propel-dev-486222`, account `ab@propelup.ai`, region `us-central1`. Everything runs from
**Cloud Shell**; nothing uses a local gcloud. Compute Engine, one spot L4, code and data in GCS,
checkpoints mirrored to GCS after every save. The VM builds the dataset itself on first boot if the
bucket has none. Spot preemption stops the VM; starting it again re-runs the startup script and
training resumes from `last/`. The VM stops itself when done. Defaults live in `scripts/gcp/env.sh`.

### 1. Get the code into Cloud Shell

Open [Cloud Shell](https://shell.cloud.google.com/?project=propel-dev-486222) signed in as
`ab@propelup.ai`. Then either:

**(a) via a git remote** (recommended; lets you edit locally and `git pull` in Cloud Shell):

```bash
# locally, once: create an empty private repo on GitHub, then
git remote add origin git@github.com:<org>/viveka.git && git push -u origin main
# in Cloud Shell:
git clone https://github.com/<org>/viveka.git && cd viveka
```

**(b) via direct upload**: zip the repo locally (exclude `.venv`, `runs`, `data`), use the Cloud Shell
terminal menu **⋮ → Upload**, then `unzip viveka.zip && cd viveka`.

Make the scripts executable once: `chmod +x scripts/gcp/*.sh`.

### 2. Run the runbook

```bash
gcloud config set project propel-dev-486222
source scripts/gcp/env.sh
scripts/gcp/00_setup_project.sh        # APIs, bucket, service account (once)
scripts/gcp/01_check_quota.sh          # L4 + GPUS_ALL_REGIONS quota; request if 0 (do this early)
scripts/gcp/02_push_code_and_data.sh   # tarball code -> GCS (dataset optional; VM builds it if absent)
scripts/gcp/03_create_vm.sh            # create spot L4 VM; startup.sh builds data if needed, trains configs/lora.yaml
scripts/gcp/04_logs.sh                 # tail /var/log/viveka-train.log
scripts/gcp/05_fetch_results.sh        # pull runs/<RUN_NAME> (best/, eval_report.json, ...) into Cloud Shell
scripts/gcp/99_teardown.sh             # delete the VM (add --bucket to delete the bucket too)
```

First boot does three things in order: dataset build (~10–20 min for the default 30k rows per
source, uploaded to `$BUCKET/data/processed`), training, evaluation. Later runs skip the build.

Launch a different experiment on the same VM (after the first finishes and the VM has stopped):

```bash
RUN_NAME=full-v1 TRAIN_CONFIG=configs/full.yaml scripts/gcp/03_create_vm.sh
```

Rebuild the dataset with different settings: delete `$BUCKET/data/processed` and set
`DATA_SOURCES` / `DATA_LIMIT` / `EVAL_LIMIT` before `03_create_vm.sh`.

After iterating on code locally: `git push`, then in Cloud Shell `git pull && scripts/gcp/02_push_code_and_data.sh`
and start the VM again for a new `RUN_NAME`.

### Requirements

- `ab@propelup.ai` needs `roles/owner` on the project, or `roles/editor` plus
  `roles/iam.serviceAccountAdmin` and `roles/resourcemanager.projectIamAdmin` (for `00_setup_project.sh`).
- GPU quota: `GPUS_ALL_REGIONS ≥ 1` and `PREEMPTIBLE_NVIDIA_L4_GPUS ≥ 1` (spot) or `NVIDIA_L4_GPUS ≥ 1`
  in `us-central1`. New projects default to 0; `01_check_quota.sh` prints the request link.
  Approval is usually minutes to a day for L4.

Rough cost: g2-standard-8 spot is ~$0.25–0.35/hr in us-central1; a 3-epoch LoRA run over ~150k
examples is 1.5–2 hours. Expect about $1 per experiment on spot, under $10 for A/B/C with retries.
The 200GB boot disk costs ~$20/month while the VM exists (stopped or not); tear it down between sessions.

## Experiments

| config | what trains | when |
|---|---|---|
| `base.yaml` | head only, mean pooling | how much the frozen representation already knows |
| `lora.yaml` | LoRA (r=16 on Wqkv/Wo/Wi) + head | default first serious model |
| `full.yaml` | everything | cheap at 150M params; run it once LoRA works |

Model selection is by validation log loss. After training the best checkpoint is
temperature-scaled on the validation set (`best/calibration.json`), and evaluated on `eval.jsonl`
with the per-category breakdown. Watch `distractor_heavy` vs `short_sufficient` (length shortcut)
and `relevant_insufficient` / `missing_one_fact` (relevance shortcut); a good aggregate number with
a bad slice means the model is cheating.

For the agent loop, use `SufficiencyPredictor.decide()`:

```python
from viveka.inference import SufficiencyPredictor
v = SufficiencyPredictor("runs/lora-modernbert-base/best")
v.decide(question, evidence, hi=0.98, lo=0.20)
# {'p_sufficient': 0.993, 'action': 'answer'}      | 'retrieve_more' | 'uncertain'
```

## Next steps (from the plan, in order)

1. Curate the hand-reviewed Phase-0 eval set (1–3k examples, heavy on hard negatives).
2. Build v1 data from HotpotQA + SQuAD 2.0; audit a sample with the teacher labeller.
3. Train A → B → C on the L4; compare per-category results.
4. Add MuSiQue / 2WikiMultiHopQA for deeper multi-hop, and synthetic domains via the teacher.
5. Benchmark against a frontier-LLM judge on quality, latency, cost per 1k decisions.
6. Only then: RLCD-style calibration objectives, missing-evidence prediction, ModernBERT-large.
