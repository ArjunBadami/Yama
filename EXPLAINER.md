# What we are actually doing

This is a plain-language account of Viveka: the task, every layer of the model, how the data is made, how training / validation / evaluation differ, and why the Google Cloud machine is specified the way it is. Jargon is defined the first time it appears.

The design notes that motivated the project live in `system1_models_jev_laya_gcp_build_plan.md`. This file is about the system that is actually running.

---

## 1. The task

An agent that answers questions by searching documents keeps hitting the same decision:

> Do I have enough evidence to answer this question yet?

Today that decision is usually made by a large generative model (GPT, Gemini, Claude). You ask it to write JSON like `{"sufficient": true, "confidence": 0.91}`. Two problems follow. The model is doing a lot of expensive text generation for a yes/no. And the number `0.91` is just characters it chose to write. It is not the model's real probability, so you cannot safely say "only proceed when this is above 0.98."

Viveka is a small model whose only output is that probability, computed directly:

```
(question, evidence)  →  P(sufficient)
```

It does not answer the question. A big model still does that, and only after Viveka says the evidence is enough.

In an agent loop it sits here:

```
user question
    → retrieve some passages
    → Viveka: P(sufficient)
         low   → retrieve more
         high  → hand the evidence to the big model and answer
         middle → uncertain, retrieve once more or escalate
```

**System 1**, in the sense of this project, means a fast intuitive judgment rather than a deliberative chain of thought. The name is from Kahneman. Jev (from TypeSafe) is a product that does a related thing: given a state, a question, and a list of options, it returns a probability for each option. We are not building Jev. We picked one narrower judgment, sufficiency, because the output is a single number and the training labels can be constructed mechanically.

---

## 2. Words you will see constantly

**Model weights.** The millions of numbers inside a neural network. Training means nudging those numbers so the model's mistakes get smaller.

**Pretrained.** Someone else already trained the model on a huge amount of text, at enormous cost. We start from those weights. We do not teach the model English.

**Fine-tuning.** Continuing training from a pretrained model on our specific task, with our data. Much cheaper than pretraining.

**Encoder.** A model that reads a whole passage at once and turns it into numbers. It does not write text. BERT-style models are encoders. They look at every word with every other word (both directions). That is called **bidirectional** attention.

**Decoder.** A model that writes text one token at a time, left to right. GPT, Gemini, and Gemma are decoders. We are not using one. The output we want is a probability, not a sentence, so a decoder is extra machinery.

**Token.** A chunk of text the model actually sees, often a word or part of a word. "biocompatibility" might be several tokens. Roughly 1,500 words ≈ 2,048 tokens for English.

**Tokenizer.** The program that splits text into tokens and into the special markers the model expects (`[CLS]`, `[SEP]`).

**Embedding.** The list of numbers that stands for one token at the start of the network.

**Hidden state / representation.** The list of numbers the network has computed for a token after the transformer layers. For ModernBERT-base that list is 768 numbers long. 768 is the **hidden size**, also called `d`.

**Transformer layer.** One block of the network: attention (every token looks at the others) plus a small feed-forward network. ModernBERT-base stacks 22 of these.

**Attention.** A way for each token to pull information from the other tokens. "Sufficient" depends on the question and the evidence seeing each other, so attention is the right mechanism.

**[CLS].** A special token placed at the start of the input. After the transformer, its hidden state is often used as a summary of the whole input. ModernBERT was not pretrained to make that summary good for classification, which matters for the frozen experiment below.

**Pooling.** How we collapse a sequence of hidden states into one vector. **CLS pooling** takes the `[CLS]` vector. **Mean pooling** averages the vectors of the real tokens (padding ignored).

**Head.** The tiny new piece we bolt on the end. Ours is one linear layer: 768 inputs, 1 output. 769 trainable numbers including the bias. This is the only part that is "ours" in the frozen experiment.

**Logit.** The raw number coming out of the head, before it is turned into a probability. It can be any real number, negative or positive. Call it `z`.

**Sigmoid.** Squashes `z` into a probability between 0 and 1: `1 / (1 + e^(-z))`. A large positive logit becomes ~1, a large negative logit becomes ~0, zero becomes 0.5.

**Loss.** A number that says how wrong the model was on a batch. Training minimizes it.

**Binary cross-entropy (BCE).** The loss for "predict a probability of a yes/no label." It punishes confident wrong answers much harder than unsure ones. If the label is 1 and the model says 0.99, the loss is tiny. If the label is 1 and the model says 0.01, the loss is huge.

**Epoch.** One pass through the entire training set.

**Step.** One update of the weights. With our LoRA settings, one step is 16 training examples (4 at a time, four times, then one update). See gradient accumulation below.

**Batch.** The group of examples processed together on the GPU.

**Learning rate.** How big each weight update is. Too big and training diverges. Too small and it barely moves.

**GPU.** A chip that does the matrix math in training. Ours is an NVIDIA L4, 24 GB of memory.

**Checkpoint.** A saved copy of the model (and, for resume, the optimizer) so a crash does not throw the run away.

---

## 3. Why this model, not a bigger one

We use **ModernBERT-base** (`answerdotai/ModernBERT-base`):

- About 150 million parameters (the weights). A frontier model is hundreds of billions. Gemma, which is the decoder people often reach for on Google Cloud, starts around 2 billion and goes up. We do not need a decoder.
- It is an encoder from 2024, so the pretrained representations are much stronger than old BERT, and the context window is 8,192 tokens. We currently cap inputs at 2,048 (about 1,500 words) to fit the GPU. The question is never truncated; if something has to be cut, it is the end of the evidence.
- We do not train language from scratch. ModernBERT already knows words like "refund," "device," "sufficient," negation, and ordinary relations. Fine-tuning only teaches it to express that knowledge as one probability.

The first serious run is **LoRA** on this base model, not a bigger model. If the task is real, a 150M encoder should learn it. If it cannot, scaling the model is not the first fix. The data is.

---

## 4. The network, layer by layer

```
question + list of evidence sentences
        │
        ▼
formatting          "question: …"   and   "evidence:\n- …\n- …"
        │
        ▼
tokenizer           [CLS] question tokens [SEP] evidence tokens [SEP]
                    cut evidence only, never the question
                    pad shorter examples so a batch is rectangular
        │
        ▼
ModernBERT-base     22 transformer layers, bidirectional
                    output: one 768-vector per token
        │
        ▼
pooling             CLS vector, or the average of the real tokens
        │
        ▼
dropout (0.1)       during training, randomly zero 10% of the vector
                    so the head cannot rely on a few fragile coordinates
        │
        ▼
linear head         z = w · h + b          (768 weights + 1 bias)
        │
        ▼
temperature         z / T                  T is 1 until we calibrate
        │
        ▼
sigmoid             P(sufficient) = 1 / (1 + e^(-z/T))
```

**Dropout** is only on during training. At inference every coordinate is used. It is a regularizer: a way to stop the model memorizing the training set.

**Padding.** Examples in one batch have different lengths. The tokenizer pads the short ones with a dummy token and an **attention mask** tells the model "ignore the padding."

**Why a text pair.** We pass the question and the evidence as two segments. The tokenizer then knows it may only truncate the second segment. Otherwise a long evidence list could chop off the question, and the model would be judging sufficiency of text that no longer contains what was asked.

**The "UNEXPECTED" lines in the log.** ModernBERT's file on Hugging Face includes a head for its original pretraining task (predicting masked words). We load the encoder and ignore that head. The log says those keys are unexpected. That is correct. We replace that head with our single-number head.

### Three ways of training the same starting model

These are not three architectures. Each starts from the same untouched ModernBERT and learns the same task. They differ in which weights are allowed to move.

| run | config | what changes | trainable weights | pooling |
|---|---|---|---|---|
| `frozen-v1` | `configs/base.yaml` | only the head | 769 | mean |
| `lora-v1` | `configs/lora.yaml` | head + small adapters | about 3.4 million | CLS |
| `full-v1` | `configs/full.yaml` | head + every encoder weight | about 150 million | CLS |

**Frozen.** The encoder is locked. Only `w` and `b` learn. This asks: is sufficiency already linearly readable from the pretrained vectors? Mean pooling is used here on purpose. ModernBERT never trained `[CLS]` to be a sentence summary, so the CLS vector is a weak feature when the encoder cannot adapt. Mean pooling is a safer frozen summary.

**LoRA** (Low-Rank Adaptation). The original weight matrices stay frozen. Next to selected matrices we add a small update `B·A`, where `A` and `B` are thin. The rank `r` is 16: each update is forced through a 16-dimensional bottleneck. We attach those adapters to the attention and feed-forward matrices named `Wqkv`, `Wo`, and `Wi` in every layer. Alpha 32 scales the update. LoRA dropout is 0.05. The checkpoint we ship is small: the base model is re-downloaded, and we store only the adapters plus the head. This is the run we expect to keep.

**Full fine-tune.** Every weight moves, with a lower learning rate on the encoder (0.00003) than on the head (0.001), so we do not immediately overwrite what pretraining learned. At 150M parameters this is cheap compared with a big decoder. It is the "did LoRA have enough capacity?" check, not a different product.

They do not continue from each other. Frozen finishing badly does not initialize LoRA. Each run is an independent experiment from the original pretrained weights.

### What we saw

The finished numbers are in section 15. Short version: frozen failed as a gate (confident yeses were wrong 29% of the time). LoRA finished and is a real model of this task (confident yeses wrong about 6% of the time on the held-out set). Full fine-tuning was started and then dropped; it is not needed for the next step.

---

## 5. How a training step works

For each example the label `y` is 1 if we marked the evidence sufficient, otherwise 0. The model emits a probability `p`. The loss on that example is

```
-( y·log(p) + (1-y)·log(1-p) )
```

That is binary cross-entropy. We average it over the batch and adjust the weights to make it smaller. The adjustment rule is **AdamW**: it keeps a moving average of recent gradients (so noisy batches do not yank the weights) and applies **weight decay** (a small pull of the weights toward zero, so they do not grow without bound). LoRA uses weight decay 0.01. Frozen uses 0, because the only trained piece is a 769-parameter head and decaying it is unnecessary.

**Learning rate schedule.** The step size starts at 0 and rises linearly for the first 6% of steps (**warmup**), then declines linearly to 0. Warmup stops the first updates, when the head is random, from taking huge steps. LoRA's peak learning rate is 0.0002. Frozen's head uses 0.001 because a tiny head can take larger steps.

**Gradient clipping.** If the update would be enormous, we shrink it so its length is at most 1.0. This prevents one bizarre batch from wrecking the run.

**bf16.** On the GPU, most of the math is done in **bfloat16**, a 16-bit floating-point format. It is faster and uses less memory than 32-bit, and unlike older 16-bit floats it does not overflow easily. The L4 supports it. The log line `bf16=True` means this is on.

**Gradient accumulation.** The GPU cannot hold 16 long sequences at once during LoRA (it ran out of memory). So we run 4 sequences, remember the gradient, run another 4, and so on, four times, and only then update the weights. That is `batch_size: 4` and `grad_accum: 4`. The effective batch is still 16, and the number of weight updates is unchanged: 37,809 steps. Frozen never backpropagates through the encoder, so it can use a real batch of 32 and only needs 18,906 steps.

**Gradient checkpointing.** During the backward pass the network normally stores every intermediate activation. At 2,048 tokens that storage fills the 24 GB GPU. Checkpointing throws those activations away and recomputes them when needed. Training is slower, memory is much lower. It is on for LoRA and full, off for frozen (frozen does not need the backward pass through the encoder).

**Epochs: 3.** Three passes over 201,648 examples. LoRA's loss was still falling at the end of epoch 1, so the extra passes are earning their time. More than three, on this constructed data, risks memorizing the particular way we built the examples.

**Seeds.** `seed: 42` fixes the shuffle order and the initial head weights so a rerun is comparable.

---

## 6. Train, validation, and evaluation are three different sets

This is the distinction that makes the numbers meaningful.

**Train** (`train.jsonl`, 201,648 examples). The model learns from these. About half sufficient, half not, on purpose (`DATA_POS_RATE=0.5`). If we left the raw mix, there were more negatives than positives and the model could look accurate by always saying "not enough."

**Validation** (`val.jsonl`, 11,219 examples). Carved out of the same pool as train, before training, and never used for weight updates. Every 500 steps we score the model on it and save a checkpoint if validation log loss improved. Log loss is the selection metric because it cares about the probability, not just the yes/no. The best checkpoint is the one with the lowest validation log loss, not the one from the final step.

The split is by **group**, not by row. One HotpotQA question produces several variants (complete evidence, missing one fact, buried in noise, …). Those variants share a `group_id`. All of them go to train or all of them go to validation. If we split by row, the model could see the "sufficient" version of a question while training and the "missing one fact" version while being validated, and the score would be a leak, not a measurement. Five percent of groups are validation (`val_frac=0.05`).

**Evaluation** (`eval.jsonl`, 13,967 examples). Built from the **validation splits of HotpotQA and SQuAD**, which the training builder never reads. So these questions were not in the 30,000 source rows we trained on, even before the group split. This is the number we quote. Frozen's eval report is already in the bucket. LoRA's will be written when the run finishes, then temperature-scaled.

Validation and evaluation answering "did we memorize our own construction?" at two strengths. Validation can still share style with train. Evaluation at least uses questions the builder held out at the source.

We have **not** yet built the hand-reviewed set from the original plan (1,000–3,000 examples a person checked). Until that exists, or until we score a frontier model on the same eval file, "good eval numbers" means "good at this constructed task," not "safe to gate a production agent."

---

## 7. How the data is made

No language model wrote this training set. A **teacher** (Gemini) path exists in `src/viveka/teacher/` for later: it can label examples by seeing whether a strong model answers correctly from the evidence alone, and it can invent new questions. We did not run it. The current labels are mechanical.

### Sources

**HotpotQA**, distractor setting, 30,000 training questions plus 2,000 from its own validation split for eval. Each question comes with 10 Wikipedia paragraphs (2 gold, 8 retrieved distractors) and a list of the exact supporting sentences. Most questions are "bridge" questions: the answer needs facts from two paragraphs. That is real multi-hop structure.

**SQuAD 2.0**, 30,000 training questions plus 2,000 from its validation split. Answerable questions have the answer inside one sentence of a paragraph; that sentence is gold and the rest of the paragraph is on-topic noise. Unanswerable questions were written to look answerable and are labeled insufficient. Those are some of the better hard negatives, because they share vocabulary with the paragraph on purpose.

### The curriculum

For one source question we know three piles:

- **Gold:** sentences that support the answer.
- **Relevant distractors:** other sentences from the same passages, on topic, not the supporting ones.
- **Unrelated:** sentences from a different question.

`src/viveka/datasets/degrade.py` turns that into several labeled examples. Evidence order is shuffled so position is not a clue.

| what we build | label | category | shortcut it breaks |
|---|---|---|---|
| all gold, little or no extra text | sufficient | `obvious_sufficient`, or `short_sufficient` if it is short, or `multi_hop` if the question needs two passages | "long means yes" |
| all gold buried in many on-topic sentences | sufficient | `distractor_heavy` | "lots of relevant text means yes" and "long means yes" |
| gold with one supporting sentence removed | insufficient | `missing_one_fact` | "same words as the question means yes" |
| only on-topic non-gold sentences | insufficient | `relevant_insufficient` | "relevant means sufficient" |
| a long pile of on-topic non-gold sentences | insufficient | `long_insufficient` | "long means yes" |
| unrelated sentences or nothing | insufficient | `obvious_insufficient` | the trivial case |

HotpotQA emits the long-insufficient variant. SQuAD usually has only one gold sentence, so "drop one fact" falls back to a relevant-insufficient example. SQuAD unanswerable questions are their own relevant-insufficient examples and are not run through the gold-sentence logic.

**Title prefixes.** Hotpot sentences are stored as `Article title: sentence`. We shuffle sentences from different articles into one list. Without the title, "It first aired in 2006" loses its subject. Distractor sentences get titles too, so the mere presence of a title does not reveal the label. SQuAD sentences are not prefixed; they come from one paragraph.

**Known flaw, confirmed by reading samples.** On Hotpot comparison questions the dataset marks both entities as supporting facts even when one sentence already answers the question. Deleting the other sentence then produces a `missing_one_fact` example that is actually still sufficient (the Vanished Planet / Hex case). That makes some negatives too hard in the wrong direction: the model is told "not enough" when a reader could still answer. It is a reason to filter the next data build. It is not a reason to believe the whole set is mislabeled. The other categories we read (short sufficient, buried year, McVeigh, Bardo Pond vs the murderer, Afroman) match the intended task.

After balancing, the training mix is about:

| category | train count (approx.) |
|---|---|
| distractor_heavy | 50,400 |
| relevant_insufficient | 49,700 |
| multi_hop | 28,500 |
| long_insufficient | 25,600 |
| missing_one_fact | 25,500 |
| obvious_sufficient | 18,100 |
| short_sufficient | 3,800 |

---

## 8. Metrics, in words

**Accuracy.** Fraction of examples where `p ≥ 0.5` matches the label. Easy to game if one class dominates. Ours is roughly balanced, so accuracy is readable, but it ignores how confident the model was.

**Precision / recall / F1**, here for the "sufficient" class. Precision: of the times we said sufficient, how often was it. Recall: of the truly sufficient cases, how many we caught. F1 is their harmonic mean.

**False-sufficient rate.** Of the truly insufficient cases, how often we said sufficient. This is the expensive error. The agent would answer too early.

**False-insufficient rate.** Of the truly sufficient cases, how often we said not enough. This wastes a retrieval hop. Annoying, usually recoverable.

**Log loss.** The BCE number on a whole set. Lower is better. Sensitive to confident mistakes. We select checkpoints by validation log loss.

**Brier score.** Average of `(p - y)²`. Also punishes confident mistakes. 0 is perfect, 0.25 is a constant prediction of 0.5.

**AUROC.** If you pick a random sufficient example and a random insufficient one, how often is the sufficient one given the higher probability. 0.5 is chance, 1.0 is perfect ranking. It ignores calibration: a model can rank perfectly and still be numerically overconfident.

**ECE (expected calibration error).** Bucket predictions by confidence and compare each bucket's average confidence to the fraction that were actually sufficient. Lower is better. A model can have a fine ECE overall and still be badly wrong in the extreme tail (the 0.98 bucket). We look at both.

**Reliability diagram.** The same buckets, plotted. Not drawn automatically; the bin counts are in the JSON report.

**Selective prediction.** We do not force a yes/no on every example.

```
p ≥ 0.98   →  treat as sufficient (let the big model answer)
p < 0.20   →  treat as insufficient (retrieve more)
otherwise  →  abstain (uncertain)
```

**Coverage** is the fraction of examples that are not abstentions. **False-sufficient among calls** is how often the confident yeses are wrong. That pair is the product metric: decide as often as you can, without waving bad evidence through. Frozen: coverage 0.20, and 29% of the yeses were wrong. LoRA at epoch 1, on validation: coverage 0.72, and about 1% of the yeses were wrong.

**Temperature scaling.** After training, we fit a single number `T` on the validation logits so that `sigmoid(z / T)` matches the observed frequencies better. It does not change the ranking, so accuracy and AUROC stay put. It only stretches or squashes confidence. The fitted `T` is stored in the checkpoint. We do this because the plan's later idea, a reinforcement-learning calibration objective (RLCD), is unnecessary until a plain model plus one scalar has been measured and found wanting.

---

## 9. Inference

One checkpoint, not all three. After the runs finish we compare eval reports and keep one directory, most likely `runs/lora-v1/best`.

```
viveka-predict --ckpt runs/lora-v1/best --question "..." --evidence "..." "..."
```

That loads ModernBERT, the LoRA adapters, the head, and `T`, runs the same formatting as training, and returns:

```
{"p_sufficient": 0.99, "action": "answer"}
```

`action` is `answer`, `retrieve_more`, or `uncertain`, using the 0.98 / 0.20 cutoffs. Those cutoffs are configuration, not learned. They should be retuned on the eval report if the probability distribution shifts.

---

## 10. The computer it is training on

Nothing trains on your laptop. The laptop (and Cloud Shell) only send instructions.

**Cloud Shell** is a small free terminal in the browser, logged in as `ab@propelup.ai`, project `propel-dev-486222`. It is not the GPU. Closing it, or pressing Ctrl-C on the log, does not stop training.

**A VM** (virtual machine) is a rented computer. Ours is named `viveka-train`.

**Machine type `g2-standard-8`.** 8 virtual CPUs, 32 GB of ordinary RAM, and one **NVIDIA L4** GPU with 24 GB of GPU memory. The L4 is an inference-and-small-training card. It is enough for a 150M encoder. It is not enough for a 16-sequence batch at 2,048 tokens once gradients are stored, which is why LoRA uses batch 4 plus accumulation plus checkpointing. We did not buy an A100 or H100. The project thesis is to use no more model, and no more chip, than the computation needs. Quota on this project allows L4s and even A100s; capacity in a given zone is a separate problem and is why we hop zones.

**Zone.** A data-center hall inside a region. We are in `us-central1` (Iowa). A VM lives in one zone and cannot be started in another. When `us-central1-b` had no free L4, we deleted the stopped machine and created it again in a zone that had one. Deleting was safe because checkpoints, data, and code are in the bucket, not on the VM disk.

**On-demand, not spot.** Spot is leftover capacity at roughly a third of the price, which Google can take back at any moment. On-demand is the full price, about $0.85 per hour for this machine, and it is not preempted. Spot was sold out in the region when we needed it. The running machine is on-demand. A 16-hour cap is about $14 worst case. The GPU stops billing when the VM stops. The 200 GB disk bills a small amount until the VM is deleted.

**Boot disk, 200 GB.** Holds the OS, the PyTorch install, the dataset, and local checkpoints. Larger than the OS image (100 GB) on purpose; the dataset and model cache need the room. Ubuntu resized the partition itself.

**Image `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`.** The disk image the VM boots from: Ubuntu 22.04 with NVIDIA driver 580 and a PyTorch-oriented stack. The family name `pytorch-latest-gpu` no longer exists, which is why the first create failed. We still `pip install` our own PyTorch, torchvision, and torchaudio from the same CUDA 12.9 wheel index, because the image's copies and a newer `torch` do not match, and the training library imports the audio and vision packages even though we never use them. That mismatch crashed the first boots.

**Bucket `gs://viveka-propel-dev-486222`.** Durable storage. Holds the code tarball, `data/processed/*.jsonl`, and `runs/<name>/`. The VM is disposable. The bucket is not.

**Service account `viveka-train@...`.** The identity the VM runs as. It can read and write the bucket, call Vertex AI later if we use Gemini as a teacher, write logs, and stop itself. Your user account is not on the VM.

**Startup script.** When `03_create_vm.sh` creates or starts the VM, it hands Google `scripts/gcp/startup.sh` and says "run this every boot." That script installs the code, fixes the PyTorch stack, copies the dataset down (it does not rebuild it, because `train.jsonl` is already in the bucket), then trains whatever is in the queue. A finished run has a `DONE` file and is skipped. An unfinished run resumes from `last/`.

**Watchdog.** A timer inside that script. `MAX_VM_HOURS=16` on the current LoRA run. When it fires, the VM is stopped no matter what. This is the cap on a hung job. Checkpoints already uploaded survive.

**Why not Vertex AI custom jobs yet.** Vertex is Google's managed training service: you submit a job, it provisions machines, you do not SSH. It is the right tool once the recipe is stable. For the first experiments a single VM we can read the log of is simpler, which is what the plan says.

### What the numbered scripts are

| script | what you are doing |
|---|---|
| `00_setup_project.sh` | one-time: APIs, bucket, service account |
| `01_check_quota.sh` | can this project use an L4 |
| `02_push_code_and_data.sh` | upload the code tarball; data upload only if you built it locally |
| `03_create_vm.sh` | create or start the GPU machine and attach the startup script |
| `04_logs.sh` | print the log. Does not train, start, or stop anything |
| `05_fetch_results.sh` | download finished runs and print a comparison |
| `99_teardown.sh` | delete the VM so the disk stops billing |

`03` does not train. It turns the computer on. Training starts a few minutes later, inside the startup script, after packages install.

---

## 11. What is saved, and what "resume" means

Under `gs://viveka-propel-dev-486222/runs/lora-v1/`:

- `best/` — weights from the step with the best validation log loss. This is what we would ship.
- `last/` — latest weights plus optimizer and step number. This is what resume loads.
- `metrics.jsonl` — one JSON line every 20 training steps and every validation.
- `eval_report.json` — written only when the run finishes and is scored on `eval.jsonl`.
- `DONE` — written only after that. Its absence means the run is incomplete.
- `startup.log` — the boot log, including tracebacks.

Saves happen every 500 steps. A crash loses at most 500 steps. The LoRA out-of-memory crash lost the steps between 2,500 and about 2,920. Resume continued from 2,500.

---

## 12. Decisions we made, and why

| decision | why |
|---|---|
| Encoder, not Gemma or another decoder | the output is one probability, not text |
| ModernBERT-base, not large | the thesis is to use the smallest model that works; large is a later experiment |
| LoRA as the main run | adapts the encoder without a 600 MB full checkpoint; frozen already showed the encoder must move |
| Mean pooling only when frozen | CLS was not pretrained as a summary |
| CLS pooling for LoRA and full | once the encoder can train, it can make CLS into a useful summary |
| BCE, not a reinforcement objective | prove the task is learnable before inventing a training algorithm |
| Temperature scaling after training | one number often fixes calibration; measure that before anything fancier |
| Select checkpoints by log loss | it scores the probability, which is the product |
| Operate at 0.98, not 0.5 | false "sufficient" is the costly error |
| Labels by deleting gold sentences | the label is auditable; we do not ask a model "is this enough?" and inherit its biases |
| HotpotQA + SQuAD 2.0 | supporting-sentence annotations and adversarial unanswerables match the curriculum |
| Group split | variants of one question must not leak across train and validation |
| Eval from the datasets' own validation splits | questions the training slice never contained |
| Balance train to 50/50 | stop the model winning by saying "insufficient" |
| Cap length at 2,048 | fits the L4; ModernBERT can go to 8,192 later if the task needs it |
| Truncate evidence, never the question | a missing question makes the label meaningless |
| Title prefixes on Hotpot sentences | shuffling would otherwise orphan pronouns |
| L4, not A100/H100 | 150M parameters do not need a larger chip |
| On-demand after spot was unavailable | the run is long enough that a preemption in the middle is annoying, and spot had no capacity |
| `g2-standard-8` | the standard 1×L4 shape with enough CPU RAM to build data and load the model |
| Batch 4 × accum 4 | same update size as batch 16, a quarter of the activation memory |
| Gradient checkpointing | the other half of fitting 2,048-token backward passes on 24 GB |
| Resume from GCS, VM is disposable | zone stockouts and preemption are normal; the bucket is the source of truth |
| 16-hour watchdog on the LoRA resume | the remaining run is on the order of 9–14 hours; the cap is there so a hang cannot bill all week |
| Do not rerun frozen | it answered its question |
| Full fine-tune is optional now | only worth it if we still want to know whether LoRA left accuracy on the table |

---

## 13. What this run is not

- It is not a model that answers questions.
- It is not Jev. Jev scores a runtime list of options. Viveka scores one yes/no.
- The training labels are not "a frontier model got the question right." They are "the dataset's supporting sentences are present." Those are correlated and not identical.
- Validation numbers are not the final score. The eval file is.
- A good score on this data is not a measurement on support tickets, contracts, or device files. That needs a new eval set from that domain.
- We have not yet compared Viveka to Gemini-as-a-judge on the same examples. That comparison is the one that says whether the small model is actually the better gate.

---

## 14. Where the code for each piece lives

| piece | file |
|---|---|
| example format, JSONL, group split | `src/viveka/schema.py` |
| question/evidence text layout | `src/viveka/formatting.py` |
| the network | `src/viveka/model/sufficiency_model.py` |
| temperature fit | `src/viveka/model/calibration.py` |
| training loop | `src/viveka/train.py` |
| metrics | `src/viveka/metrics.py` |
| scoring a finished checkpoint | `src/viveka/evaluate.py` |
| the probability an agent would call | `src/viveka/inference.py` |
| turning one question into many labeled variants | `src/viveka/datasets/degrade.py` |
| HotpotQA and SQuAD builders | `src/viveka/datasets/hotpotqa.py`, `squad_v2.py` |
| frozen / LoRA / full settings | `configs/base.yaml`, `lora.yaml`, `full.yaml` |
| what the GPU machine does on boot | `scripts/gcp/startup.sh` |
| machine size, region, caps | `scripts/gcp/env.sh` |

---

## 15. The journey so far

This section is the log. Earlier sections describe the system. This one records what actually happened, what we concluded, and what we decided not to do. New rounds get added at the bottom.

### Round 1 — first trains (24–25 Sep 2026)

**What ran.** One dataset, three independent runs from the same pretrained ModernBERT, queue order frozen then LoRA then full. Data was HotpotQA + SQuAD 2.0, 30,000 source questions each, degraded by the curriculum in section 7. Train 201,648 examples (balanced 50/50), validation 11,219, held-out eval 13,967. No Gemini labels. The GPU was an on-demand L4 (`g2-standard-8`) because spot capacity was gone.

**Infrastructure, not results.** The first boots failed before any training: a retired VM image name, then a PyTorch / torchaudio / torchvision version mismatch that crashed on import, then CUDA out-of-memory at batch 16 and 2,048 tokens. Fixes that stuck: pin the image family, reinstall torch + torchvision + torchaudio from the same CUDA 12.9 index, batch 4 with 4-step gradient accumulation (same 37,809 optimizer steps), and gradient checkpointing. Zone stockouts meant the machine was deleted and recreated; checkpoints lived in the bucket the whole time. `lora-v1` is the run that finished under those fixes.

**Frozen (`frozen-v1`), held-out eval.** Encoder locked, only the 769-parameter head trained, mean pooling. It finished.

| | |
|---|---|
| log loss | 0.59 |
| AUROC | 0.75 |
| confident-yes error (P>0.98) | 29% |
| share of cases it would decide | 20% |
| short but complete evidence, accuracy | 0.28 |
| multi-hop sufficient, accuracy | 0.46 |
| long insufficient, accuracy | 0.57 |

Finding: the pretrained vectors do not already contain a sufficiency decision. The head learned a length shortcut. More text looked sufficient. Short complete answers looked insufficient. This run answered its question. We are not repeating it.

**LoRA (`lora-v1`), held-out eval, after 3 epochs.** Head plus rank-16 adapters, about 3.4 million trainable weights, CLS pooling. Temperature fitted on validation was about 1.44 (the raw model was overconfident; dividing logits by 1.44 fixed most of that). Checkpoint: `gs://viveka-propel-dev-486222/runs/lora-v1/best`. Do not overwrite this folder.

| | at end of epoch 1, validation only | finished, held-out eval |
|---|---|---|
| log loss | 0.20 | 0.27 |
| AUROC | 0.98 | 0.96 |
| accuracy at 0.5 | 0.93 | 0.90 |
| confident-yes error (P>0.98) | 1% | 6% |
| share of cases it would decide | 72% | 67% |
| short but complete, accuracy | — | 0.74 |
| multi-hop sufficient, accuracy | — | 0.90 |
| long insufficient, accuracy | — | 0.96 |

Finding: adapting the encoder is what makes the task learnable. The length shortcut mostly broke. Validation looked cleaner than the held-out eval, which is what validation is for: it is the same construction as training. The 6% is the number that counts. At 0.98 the model made 2,502 "answer now" calls and 149 of them were wrong. Recall at that bar was only 39%: it refuses to be sure on most truly sufficient cases, which is the cautious behavior we want, as long as the yeses it does emit are clean. Six percent is not clean enough to automate.

**Full fine-tune (`full-v1`).** Started, reached step 2,500 of 37,809, then died on GPU memory. At that early step its validation numbers were almost the same as LoRA's (AUROC about 0.95). We did not restart it. LoRA already adapts the encoder, and the remaining errors look like a kind of example, not like a model that is too small. A full fine-tune is optional later, only if a cleaned LoRA run stops improving.

### What the 6% actually is

We printed the confident false yeses from `runs/lora-v1/eval_preds.jsonl`. 145 rows. 133 were SQuAD `relevant_insufficient`. 12 were HotpotQA. Several scores were 0.995–0.999, so a higher threshold does not remove them.

The pattern is constraint mismatch. The paragraph contains an answer-shaped phrase, and the question adds a condition that phrase does not meet:

- A total for several countries is treated as the number who fled to England only.
- A brother's marriage is treated as the king's marriage.
- A list of towns is treated as an answer to "which had the most."
- A year in the paragraph is the year of a different event.

The model learned "there is a plausible answer span" more firmly than "that span satisfies every constraint in the question." SQuAD 2.0 unanswerable questions are built exactly this way, and they were already in training. The model saw them and still assigns the hard ones 0.99.

A minority of those 145 look like bad labels: the steam-cycle sentence really does list expansion, and the atmospheric-engine sentence really does name the first commercially successful engine. On those the model is right. They are not most of the pile.

### Label fix we did make, and what it changed

Reading training samples turned up a separate bug. On HotpotQA comparison questions both entities are marked as supporting facts, even when one sentence already answers the question. Dropping the other sentence produced a `missing_one_fact` example that was still sufficient (Vanished Planet was still described as the 2003 cooperative game after the Hex sentence was removed).

`degrade.py` now refuses that negative. It tries each sentence to drop and keeps the first deletion that actually removes the answer text from the evidence. If every deletion leaves the answer visible, it emits no negative. Yes/no answers and strings shorter than 8 characters are not used for this check, because "no" and "Bury" appear inside unrelated sentences.

The original data was copied to `gs://viveka-propel-dev-486222/data/processed-v1` before the rebuild. The cleaned build is `data/processed-v2`. Positives were unchanged. `missing_one_fact` shrank by about 5,700 train rows and 457 eval rows.

We then scored the **existing** `lora-v1` checkpoint on the cleaned eval, without any new training (`runs/lora-v1/eval-v2-report.json`):

| | original eval | cleaned eval, same checkpoint |
|---|---|---|
| confident-yes error | 5.9% | 6.0% |
| AUROC | 0.957 | 0.956 |
| log loss | 0.268 | 0.269 |

Finding: that Hotpot bug was real, and it was not the source of the 6%. The model was mostly not calling those comparison cases sufficient, so removing them did not shrink the error pile. Another train on the same recipe, even with this fix, should not be expected to move 6% to 2% by itself.

### Where we are, and the next experiment

Kept, and not to be overwritten:

- `runs/lora-v1/best` — the model we would actually load
- `runs/frozen-v1/` — the baseline that showed a frozen encoder is not enough
- `data/processed-v1` — the data `lora-v1` was trained on
- `data/processed-v2` — the same build with the comparison-question fix

Not worth doing next: rerun frozen, finish full fine-tune, raise the threshold, or train `lora-v2` on the same distribution and hope.

Worth doing next: build more negatives of the shape the model gets wrong (right entities, failed constraint), upweight the confident mistakes we already have, and train `lora-v2` from the original ModernBERT, not from the `lora-v1` weights. A fresh start keeps "better data" separate from "more steps on the old checkpoint." Score `lora-v2` on a held-out set that contains those constraint mismatches. The target discussed for this eval style is about 2% confident-yes error. That number on Wikipedia data would mean the model learned the constraint. It would still not be a product claim until the same measurement is done on real retrieval traces, against a cheap frontier-model judge.
