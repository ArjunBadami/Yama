# System 1 Models, Jev, Laya, and Our GCP Build Plan

## Executive Summary

The core idea behind a **System 1 model** is that many tasks currently assigned to large autoregressive language models do not actually require free-form language generation.

A great deal of workflow automation is fundamentally asking for one of a small number of mathematical objects:

- a probability over a set of choices,
- a binary probability,
- a score,
- a ranking,
- a stop/continue signal,
- a risk estimate,
- a verifier judgment,
- or a prediction about whether the available evidence is sufficient.

Today, developers often ask a generative LLM to produce those objects indirectly as text or JSON:

```json
{
  "route": "billing",
  "confidence": 0.91
}
```

But the model is still doing autoregressive token generation. The string `"0.91"` is generated text; it is not automatically the model's calibrated probability that the route is correct.

The System 1 idea is to instead make the **native output of the model the thing the software actually needs**.

For Jev, the primitive is approximately:

\[
(\text{state}, \text{question}, \text{options})
\rightarrow
P(\text{options})
\]

For the model we want to build, the primitive is:

\[
(\text{question}, \text{evidence})
\rightarrow
P(\text{sufficient})
\]

That model would answer one narrow but extremely important question for an agent:

> **Do I have enough evidence to answer this question yet?**

The goal is to train it on top of an existing pretrained encoder such as ModernBERT, run it cheaply on GCP, and use it inside agent/retrieval loops as a low-latency control signal.

---

# 1. Why System 1 Models Are Interesting

Modern LLM applications frequently use a huge autoregressive model for computations that are not inherently generative.

Examples:

- Which tool should I call?
- Which department should handle this ticket?
- Is this claim supported?
- Is this evidence enough?
- Should I stop retrieving?
- Should this action be allowed?
- How risky is this action?
- Which of these alternatives is most appropriate?
- Should this case be escalated to a human?

A typical LLM workflow might be:

```text
state
  ↓
large autoregressive LLM
  ↓
generated reasoning
  ↓
generated JSON
  ↓
parse JSON
  ↓
read generated confidence number
  ↓
branch in software
```

This works, but it is often an awkward computational path.

The software may only need:

\[
P(\text{tool}_1), P(\text{tool}_2), P(\text{tool}_3)
\]

Yet the system pays for a large decoder to generate many tokens explaining or serializing that decision.

The System 1 model philosophy is:

> **Ask what numerical object the software actually needs, and train a model to output that object directly.**

That can mean:

\[
\text{semantic input}
\rightarrow
\text{native mathematical output}
\]

rather than:

\[
\text{semantic input}
\rightarrow
\text{text generation}
\rightarrow
\text{parse text back into a decision}
\]

---

# 2. The Jev Idea

Jev, from TypeSafe AI, is an example of this approach.

Its key conceptual primitive is:

\[
(\text{state}, \text{question}, \text{candidate answers})
\rightarrow
P(\text{candidate answers})
\]

Instead of asking a decoder model:

> "Read this support ticket and return JSON containing the category and confidence."

you can pose something like:

```text
State:
"The customer says they were charged twice."

Question:
What type of issue is this?

Options:
- Billing
- Technical
- Account
```

and receive a probability distribution conceptually like:

```text
Billing      0.97
Technical    0.02
Account      0.01
```

The important point is that the output probabilities are intended to be the model's **native decision distribution**, not numbers generated as characters by an autoregressive model.

---

# 3. Why Jev Feels Brilliantly Simple

The surprising part is how little conceptual machinery is required.

A traditional LLM produces a hidden representation and then projects it into a fixed vocabulary:

\[
h_t
\rightarrow
W_{\text{vocab}}h_t
\rightarrow
z \in \mathbb{R}^{V}
\]

where \(V\) is the fixed vocabulary size.

Then:

\[
P(\text{token}_i)
=
\operatorname{softmax}(z)_i
\]

Every generation step chooses among the same fixed vocabulary.

A Jev-like model instead asks:

> Why should the final softmax be over a vocabulary at all?

If the task has 5 possible actions, score those 5 actions.

If the task has 17 possible actions, score those 17.

If the task has 100 possible actions, score those 100.

Conceptually:

\[
s_i = f_\theta(\text{state},\text{question},\text{option}_i)
\]

and then:

\[
P(i)
=
\frac{e^{s_i}}
{\sum_{j=1}^{N} e^{s_j}}
\]

where \(N\) is the number of choices supplied at runtime.

The final probability vector is therefore **dynamic in size**.

That is substantially different from an autoregressive decoder whose output projection is permanently tied to a fixed vocabulary.

---

# 4. Jev Is Still a Language-Understanding Model

An important distinction:

Jev can consume natural-language state and natural-language option descriptions, so it obviously requires strong language understanding.

But that does **not** imply that it needs to generate language.

The model can have:

- tokenization,
- embeddings,
- transformer layers,
- attention,
- hidden representations,
- softmax,

without being an autoregressive next-token generator.

The relevant distinction is not:

> "Does it process language?"

The distinction is:

> "What probability distribution is the model trained to produce?"

An autoregressive LLM models something like:

\[
P(x_1,\ldots,x_n)
=
\prod_t
P(x_t\mid x_{<t})
\]

A Jev-style decision model instead targets:

\[
P(\text{decision}\mid
\text{state},
\text{question},
\text{available decisions})
\]

That is a fundamentally different output primitive.

---

# 5. Our Likely Mental Model of Jev's Architecture

TypeSafe has not publicly disclosed every architectural detail, so we should distinguish between **public behavior** and **our architectural inference**.

A plausible architecture is:

```text
state + question + choices
          ↓
       tokenize
          ↓
  language encoder / transformer
          ↓
 contextual option representations
          ↓
   shared option-scoring head
          ↓
 one scalar logit per option
          ↓
 softmax over runtime choices
          ↓
 calibrated probabilities
```

Mathematically:

\[
h_i
=
\operatorname{Encoder}_\theta(
\text{state},
\text{question},
\text{choices}
)_i
\]

for option \(i\), then:

\[
s_i = f_\phi(h_i)
\]

and:

\[
P(i)
=
\operatorname{softmax}(s_1,\ldots,s_N)_i
\]

The scoring function can be shared across all options.

This means the model does **not** require a dedicated learned output neuron for every possible semantic answer.

For example, it does not need permanent class weights:

\[
w_{\text{billing}},
w_{\text{technical}},
w_{\text{account}}
\]

Instead the words themselves are encoded.

Tomorrow, the choices can be:

```text
- Send pricing
- Schedule demo
- Ask for more information
- Escalate to legal
```

without modifying the output head.

---

# 6. The 255-Choice Limitation

Jev currently imposes a practical limit of roughly **255 options for a single choice-style decision**.

This is not a fundamental mathematical limitation of variable-size softmax.

It is a product/model/runtime limit.

The key point is:

\[
N
\]

is dynamic, but it is not unlimited.

A request can have a variable number of options up to the supported ceiling.

For very high-cardinality problems, a practical system may need:

- hierarchical decisions,
- independent candidate scoring followed by a smaller choice stage,
- retrieval/ranking first,
- or multi-stage narrowing.

This matters because Jev is excellent for bounded semantic decisions but is not necessarily intended to directly choose among millions of possibilities in one softmax.

---

# 7. Jev Context Limits and Long Documents

A separate limitation is context.

The useful mental model from our discussion is that Jev is **not primarily a million-token "dump the entire corpus into the model" system**.

Its state is bounded enough that a typical 100-page dense PDF may not fit directly.

This exposes an important systems problem.

If we do:

```text
100-page PDF
    ↓
LLM chooses relevant passages
    ↓
Jev makes decision
```

then the LLM may have already made much of the difficult semantic judgment.

That can weaken the claim that Jev is replacing the expensive reasoning layer.

A more interesting pattern is exhaustive cheap semantic scanning:

```text
100-page PDF
      ↓
split into chunks
      ↓
run lightweight decision model over every chunk
      ↓
identify evidence-bearing chunks
      ↓
final decision
```

This avoids relying exclusively on embedding similarity or an LLM to decide what is relevant.

However, there is still a real limitation.

Suppose:

```text
Page 7:
Device contacts intact skin.

Page 39:
Contact duration exceeds 30 days.

Page 82:
Material is silicone.
```

The answer may depend on the **combination** of those facts.

No individual chunk necessarily contains enough information.

That means a compact-context System 1 model does not eliminate the general problem of:

\[
\text{large corpus}
\rightarrow
\text{discover relevant facts}
\rightarrow
\text{combine distant evidence}
\]

Long-context reasoning models still have an advantage on that problem.

---

# 8. Laya: The Open-Source Analogue

Laya is useful because it makes the System 1 idea much more concrete.

The architecture we discussed is essentially:

\[
\text{ModernBERT}
+
\text{decision layers}
+
\text{shared option scorer}
\]

The important difference from GPT is that ModernBERT is a **bidirectional encoder**, not an autoregressive decoder.

---

# 9. Remembering BERT

BERT-style training is different from GPT-style training.

GPT learns primarily through next-token prediction:

```text
The capital of France is
                      ↓
                   predict
                      ↓
                    Paris
```

More formally:

\[
P(x_t\mid x_{<t})
\]

BERT-style models instead learn from masked-language modeling.

Example:

```text
The capital of France is [MASK].
```

The model sees context on **both sides** of a position and learns to reconstruct the masked token.

Because every token can attend bidirectionally, the resulting representations are very useful for:

- classification,
- semantic matching,
- retrieval,
- entailment,
- reranking,
- entity relationships,
- and decision tasks.

The important point is that BERT-style models can still be pretrained on enormous corpora.

They do not need to learn language from a tiny decision dataset.

---

# 10. Why a Small Fine-Tuning Dataset Can Work

A model like ModernBERT already has a pretrained representation of concepts such as:

```text
customer
refund
billing
fraud
device
evidence
contract
risk
urgent
technical issue
```

The decision fine-tuning phase does not have to teach the model what English means.

It teaches the pretrained model:

> "Use your existing semantic representation in this new output interface."

This is similar in spirit to instruction tuning.

The expensive learning is:

\[
\boxed{\text{large-scale language pretraining}}
\]

The relatively cheap phase is:

\[
\boxed{\text{teach model how to express knowledge as decisions}}
\]

---

# 11. Laya's Option-Marker Architecture

The mental model we used for Laya is approximately:

```text
Question:
What type of issue is this?

[MASK] Billing
[MASK] Technical
[MASK] Account

State:
The customer says they were charged twice.
```

The key clarification is that `[MASK]` is being repurposed as an **option marker**.

It is not being used in the ordinary BERT sense of:

> "Predict the vocabulary token that belongs here."

Instead:

> "Use the hidden state at this position as the contextual representation of this candidate option."

After the bidirectional transformer:

\[
h_{\text{billing}},
h_{\text{technical}},
h_{\text{account}}
\]

are obtained at the option-marker positions.

A shared decision head then computes:

\[
s_i
=
w^\top h_i+b
\]

for every option.

Example logits:

\[
[5.8,1.2,-0.7]
\]

Then:

\[
P
=
\operatorname{softmax}(5.8,1.2,-0.7)
\]

might yield:

```text
Billing      0.986
Technical    0.010
Account      0.004
```

The probability vector is not generated as text.

It is the direct numerical output of the model.

---

# 12. What the Learned \(w\) Is

In the simplified formulation:

\[
s_i=w^\top h_i+b
\]

\(w\) is the learned weight vector of the shared decision-scoring head.

But an important point from our discussion is that training does not necessarily update only \(w\).

You can train:

- only the decision head,
- the head + LoRA adapters,
- some transformer layers,
- or the entire encoder.

Laya-style training can update the pretrained backbone as well.

So the system is not merely:

```text
frozen BERT + tiny classifier
```

It can be:

```text
pretrained BERT
      +
new decision layers
      +
fine-tuning of some/all BERT weights
```

The pretrained model supplies semantic intelligence.

Fine-tuning reshapes that intelligence around the decision interface.

---

# 13. Dynamic Softmax Is the Important Trick

For a traditional fixed classifier:

\[
h_{\text{CLS}}
\rightarrow
W
\]

where:

\[
W \in \mathbb{R}^{K\times d}
\]

and \(K\) is permanently fixed.

If \(K=3\), the model always has three outputs.

A Jev/Laya-style dynamic decision model instead scores each candidate with the same function:

\[
s_i=f(h_i)
\]

Then it applies:

\[
\operatorname{softmax}(s_1,\ldots,s_N)
\]

where \(N\) comes from the request.

That is what enables runtime-defined semantic choice spaces.

---

# 14. Calibration and RLCD

Accuracy and calibration are different.

A classifier can have excellent accuracy but terrible confidence estimates.

Suppose a model says:

\[
P(\text{correct})=0.95
\]

on 1,000 predictions.

If only 700 are correct, its confidence is badly miscalibrated.

A properly calibrated model should satisfy approximately:

> Among predictions assigned 0.9 confidence, roughly 90% should be correct.

That matters enormously for automation.

A system can then make engineering decisions such as:

```text
P > 0.995
    → fully automatic

0.80 < P <= 0.995
    → light review

P <= 0.80
    → human or larger reasoning model
```

Jev/Laya emphasize training for calibrated decisions.

The training approach we discussed is called **RLCD: Reinforcement Learning for Calibrated Decisions**.

At a high level, RLCD trains the probability distribution using scoring rules designed to reward honest probabilistic forecasts.

For our own first model, however, we should **not begin with RLCD**.

We should establish a strong baseline first using:

\[
L=-\log P(y)
\]

ordinary cross-entropy.

Then evaluate:

- accuracy,
- precision/recall/F1,
- log loss,
- Brier score,
- Expected Calibration Error,
- reliability diagrams.

Then apply:

- temperature scaling,
- and eventually RLCD-like training,

if calibration is not good enough.

---

# 15. Why This Took So Long Despite Being Simple

The underlying ingredients are not new.

We have had:

\[
\text{encoder}+\text{classification head}
\]

for many years.

The real changes are:

## 15.1 Strong general language encoders

Older classifiers often solved one narrow predetermined task.

Modern pretrained encoders are much better at understanding arbitrary natural-language task descriptions and candidate meanings.

## 15.2 LLMs temporarily made specialization unnecessary

Generative LLMs let developers write:

```text
Read this ticket.

Determine:
- urgency
- department
- churn risk
- whether legal review is needed

Return JSON.
```

That is incredibly convenient.

It allowed people to discover what kinds of semantic automation were valuable without training bespoke models.

## 15.3 Production exposed the drawbacks

Once these workflows moved into production, developers encountered:

- high latency,
- high cost,
- nondeterminism,
- malformed outputs,
- hallucinated fields,
- poor self-reported confidence,
- difficulty setting automation thresholds,
- unnecessary generation.

That creates room for a specialized fast decision layer.

## 15.4 Calibration became a first-class requirement

For automation, the model does not merely need to choose the right answer.

It must know **when it is uncertain**.

That makes calibrated decision models especially useful.

---

# 16. Other System 1 Primitives We Considered

The broader insight is not merely "clone Jev."

It is to find non-linguistic computational objects currently being produced awkwardly through text generation.

Examples:

## 16.1 Verifier

\[
(\text{state},\text{claim})
\rightarrow
P(\text{claim true})
\]

Useful for:

- factual checking,
- compliance,
- validation,
- agent self-checks.

## 16.2 Transition model

\[
(\text{state},\text{candidate action})
\rightarrow
P(\text{next state})
\]

Useful for predicting what happens if the agent performs an action.

## 16.3 Pairwise preference model

\[
(\text{state},A,B)
\rightarrow
P(A>B)
\]

Useful for ranking:

- documents,
- leads,
- search results,
- candidate actions.

## 16.4 Set selector

\[
(\text{state},\{o_i\})
\rightarrow
P(o_i\text{ should be selected})
\]

Unlike a softmax, choices are not mutually exclusive.

Multiple labels may simultaneously be true.

## 16.5 Ordinal model

\[
(\text{state},\text{dimension})
\rightarrow
P(1),P(2),\ldots,P(10)
\]

Useful for risk, severity, urgency, quality, etc.

## 16.6 Continuous-value model

\[
(\text{state},\text{question})
\rightarrow
\text{distribution over }y
\]

For example:

- expected time to resolution,
- expected revenue,
- expected delay,
- failure probability.

The native output could be:

- mean and variance,
- quantiles,
- mixture-distribution parameters.

## 16.7 Semantic action gate

\[
(\text{state},\text{policy})
\rightarrow
P(\text{allow}),
P(\text{review}),
P(\text{block})
\]

A lightweight learned firewall around an agent.

## 16.8 Tool router

\[
(\text{state},\{\text{tools}\})
\rightarrow
P(\text{tool}_i)
\]

Potentially replaces expensive LLM calls whose only purpose is selecting a tool.

## 16.9 Stop/continue model

\[
(\text{agent trajectory})
\rightarrow
P(\text{stop})
\]

Could prevent agents from wasting compute or stopping prematurely.

## 16.10 Evidence sufficiency model

\[
(\text{question},\text{evidence})
\rightarrow
P(\text{sufficient})
\]

This is the one we chose to focus on.

## 16.11 Contradiction detector

\[
(\text{fact set})
\rightarrow
P(\text{internally inconsistent})
\]

Useful for memory validation and knowledge-base hygiene.

## 16.12 Semantic join

\[
(A_i,B_j)
\rightarrow
P(\text{same entity/relation})
\]

Useful for enterprise data reconciliation.

## 16.13 Anomaly detector

\[
(\text{event},\text{normal context})
\rightarrow
P(\text{anomalous})
\]

Useful for fraud, observability, and operations.

## 16.14 Next-best-action / utility model

\[
(\text{state},\text{candidate action})
\rightarrow
P(\text{desired outcome}\mid a)
\]

Example:

```text
send pricing       → 0.62
schedule demo      → 0.81
send case study    → 0.47
```

This is subtly different from Jev.

Jev asks something like:

> Which option best fits this state?

A utility model asks:

> What outcome is likely if I take this action?

---

# 17. The System 1 Model We Want to Build

The model we liked most is:

\[
\boxed{
(\text{question},\text{evidence})
\rightarrow
P(\text{sufficient})
}
\]

It answers:

> **Does the currently available evidence contain enough information for a competent reasoner to answer the question?**

It does **not** need to answer the question itself.

That is what makes the task attractive.

---

# 18. Example

Suppose:

```text
Question:
Does this device require biocompatibility testing?

Evidence:
- Device contacts intact skin.
- Contact duration exceeds 30 days.
```

The model might output:

\[
P(\text{sufficient})=0.42
\]

The agent retrieves more.

After another retrieval:

```text
Evidence:
- Device contacts intact skin.
- Contact duration exceeds 30 days.
- Material is silicone.
- Predicate testing included cytotoxicity,
  sensitization, and irritation.
```

Now:

\[
P(\text{sufficient})=0.97
\]

The system allows the expensive reasoning/generation step.

---

# 19. Why This Model Is Different From Jev

If we make:

\[
(\text{state},\text{question},\text{arbitrary options})
\rightarrow
P(\text{options})
\]

then we are largely rebuilding the same generic problem Jev solves.

The evidence-sufficiency model is intentionally narrower.

Its output is one scalar:

\[
P(\text{sufficient})
\]

That means:

- no dynamic choice set,
- no 255-option problem,
- no arbitrary schema interpretation,
- no general-purpose semantic choice model,
- no need to reproduce Jev.

It is a specialized System 1 primitive.

---

# 20. Proposed Architecture

Version 1 should be extremely simple:

```text
question + evidence
       ↓
    tokenizer
       ↓
  ModernBERT encoder
       ↓
 [CLS] hidden state
       ↓
  linear / small MLP
       ↓
     one logit
       ↓
     sigmoid
       ↓
 P(sufficient)
```

Mathematically:

\[
h
=
\operatorname{ModernBERT}_\theta(
\text{question},
\text{evidence}
)_{\text{CLS}}
\]

Then:

\[
z=w^\top h+b
\]

and:

\[
P(\text{sufficient})
=
\sigma(z)
=
\frac{1}{1+e^{-z}}
\]

That is the whole inference path.

No decoder.

No token generation.

No JSON generation.

No variable-size softmax.

No text confidence score.

---

# 21. Why Start With ModernBERT

We should not train language understanding from scratch.

That would be expensive and unnecessary.

A pretrained encoder already has semantic representations for:

- questions,
- facts,
- relations,
- negation,
- entailment,
- relevance,
- temporal statements,
- domain concepts,
- ordinary language.

Our task is only to teach it:

> Given a question and a body of evidence, estimate whether that evidence is sufficient to resolve the question.

That is a much smaller problem.

---

# 22. Training Strategy

## Phase 0 — Build an evaluation set first

Before training anything, create a high-quality evaluation dataset that the training pipeline cannot touch.

Include:

- obvious sufficient examples,
- obvious insufficient examples,
- relevant-but-insufficient evidence,
- distractor-heavy evidence,
- contradictory evidence,
- evidence missing exactly one critical fact,
- long evidence,
- paraphrased questions,
- multi-hop examples.

This evaluation set is critical.

Otherwise a low loss can fool us into thinking the model understands sufficiency when it has merely learned superficial heuristics.

---

# 23. Dataset Schema

Each example can be as simple as:

```json
{
  "question": "Does the device require biocompatibility testing?",
  "evidence": [
    "The device contacts intact skin.",
    "The contact duration exceeds 30 days."
  ],
  "sufficient": false
}
```

or:

```json
{
  "question": "Does the device require biocompatibility testing?",
  "evidence": [
    "The device contacts intact skin.",
    "The contact duration exceeds 30 days.",
    "The device is made of silicone.",
    "The predicate underwent cytotoxicity, sensitization, and irritation testing."
  ],
  "sufficient": true
}
```

---

# 24. How We Can Generate Training Data

This is probably the most important part of the project.

The GPU is cheap.

Good labels are the scarce resource.

We can obtain data from several sources.

## 24.1 Public QA datasets

Use datasets containing:

- questions,
- answers,
- supporting passages,
- evidence annotations.

For each example:

```text
question + complete supporting evidence
→ sufficient
```

Then remove required evidence:

```text
question + incomplete evidence
→ insufficient
```

Potential source families include:

- question answering,
- natural language inference,
- fact verification,
- multi-hop QA,
- open-domain QA,
- reading comprehension.

---

# 25. Synthetic Data From Frontier LLMs

A frontier LLM can serve as a **teacher/data generator**.

We can ask it to create:

- questions,
- complete evidence,
- partially complete evidence,
- irrelevant distractors,
- missing-fact variants,
- contradictory variants.

The large model is used during training-data creation.

The small model is used at runtime.

That is a classic teacher/student economic tradeoff:

\[
\text{expensive intelligence once}
\rightarrow
\text{cheap inference millions of times}
\]

---

# 26. Hard Negatives Are Essential

The biggest danger is that the model learns:

\[
\text{relevance}
\approx
\text{sufficiency}
\]

We do not want that.

Example:

```text
Question:
Who won the 2024 championship?

Evidence:
The final was played in New York.
The championship had record attendance.
The favorite entered the final undefeated.
```

All of the evidence is highly relevant.

But it is still insufficient.

Therefore our dataset needs many examples where:

\[
\text{high semantic relevance}
\land
\text{insufficient information}
\]

This forces the model to learn actual evidential completeness rather than embedding similarity.

---

# 27. A Useful Synthetic Curriculum

Suppose a question requires three facts:

\[
F_1,F_2,F_3
\]

We can construct:

```text
0/3 facts
→ insufficient

1/3 facts
→ insufficient

2/3 facts
→ insufficient

3/3 facts
→ sufficient

3/3 + irrelevant noise
→ sufficient

2/3 + lots of highly relevant noise
→ insufficient

3/3 + contradiction
→ special hard case
```

This is an unusually nice property of the sufficiency task.

Training examples can be generated systematically.

---

# 28. Multi-Hop Sufficiency

We should eventually include examples where sufficiency depends on combining multiple facts.

For example:

```text
Fact A:
The user is on the Enterprise plan.

Fact B:
Enterprise plans permit API access.

Question:
Does the user have API access?
```

Neither fact alone is sufficient.

Together they are.

A stronger example:

```text
A implies B
B implies C
Question asks C
```

This tests whether the encoder is merely matching words or actually learning enough relational structure to judge completeness.

---

# 29. First Training Objective

Start with binary cross-entropy.

For target:

\[
y\in\{0,1\}
\]

and model output:

\[
p=P(\text{sufficient})
\]

use:

\[
L
=
-y\log(p)
-(1-y)\log(1-p)
\]

Do not begin with complicated reinforcement learning.

First answer:

> Can a pretrained encoder learn this task well at all?

---

# 30. Fine-Tuning Strategy

Start conservatively.

Possible progression:

## Experiment A

```text
Frozen ModernBERT
+
train only classification head
```

Cheap and fast.

## Experiment B

```text
ModernBERT
+
LoRA adapters
+
classification head
```

Likely the best first serious model.

## Experiment C

```text
Full fine-tuning
```

Only if needed.

This sequence tells us how much task-specific adaptation is actually necessary.

---

# 31. Model Sizes

We should start with **ModernBERT-base**, not large.

Reasons:

- cheaper,
- faster,
- easier to iterate,
- lower inference latency,
- smaller GPU requirement,
- easier deployment.

If the task saturates, scale to ModernBERT-large later.

The entire philosophy of the project is:

> Do not use more model than the computation requires.

---

# 32. Calibration

A useful System 1 model needs more than accuracy.

If the model says:

\[
P(\text{sufficient})=0.95
\]

we want that number to actually mean something.

We should measure:

## Accuracy

\[
\frac{\text{correct predictions}}{\text{all predictions}}
\]

## Precision / Recall / F1

Especially useful if sufficient/insufficient classes are imbalanced.

## Log loss

Penalizes confident wrong answers strongly.

## Brier score

For binary probability:

\[
\text{Brier}
=
\frac{1}{N}
\sum_i
(p_i-y_i)^2
\]

## Expected Calibration Error

Measures mismatch between confidence and observed accuracy.

## Reliability diagram

Bucket predictions by confidence and compare:

```text
predicted probability
vs.
actual fraction sufficient
```

---

# 33. Temperature Calibration

Before inventing RLCD, apply a simple calibration layer.

For a logit:

\[
z
\]

use:

\[
p
=
\sigma(z/T)
\]

where \(T\) is learned on a held-out calibration set.

This can substantially improve probability calibration without retraining the entire model.

If the base model performs well but remains poorly calibrated, then we can explore RLCD-like objectives.

---

# 34. Later: RLCD-Style Training

Once the baseline is strong, we can experiment with explicitly rewarding calibrated probability distributions.

The broad idea is to use **proper scoring rules**.

A proper scoring rule makes the expected reward maximal when the model reports its true belief.

This discourages systematic overconfidence.

This is important because an automation system may use thresholds like:

```text
P(sufficient) > 0.99
→ answer automatically

0.80 < P(sufficient) <= 0.99
→ maybe retrieve once more

P(sufficient) <= 0.80
→ continue retrieval / escalate
```

If probabilities are not calibrated, thresholds like this are meaningless.

---

# 35. Agent Architecture With the Sufficiency Model

The model becomes a cheap control layer inside a retrieval loop:

```text
user question
      ↓
retrieve evidence
      ↓
sufficiency model
      ↓
P(sufficient)
   ┌───────────────┐
   │               │
 low               high
   │               │
retrieve more      call expensive
   │               reasoning model
   └──────↺
```

This separates two jobs:

## System 1 model

> Do I have enough evidence?

## Large reasoning model

> Given sufficient evidence, what is the answer?

That separation is conceptually clean.

---

# 36. Why This Could Save Cost

Today a common retrieval loop does:

```text
retrieve
↓
LLM reads evidence
↓
LLM decides whether more retrieval is required
↓
retrieve
↓
LLM reads again
↓
...
```

Every loop may invoke a frontier decoder.

Instead:

```text
retrieve
↓
150M encoder
↓
retrieve again if needed
↓
150M encoder
↓
...
↓
one final large LLM call
```

The small model can potentially eliminate many expensive reasoning calls.

---

# 37. Why This Could Improve Quality

The benefit is not only cost.

A dedicated sufficiency model could reduce:

- premature answers,
- answers based on incomplete evidence,
- hallucinations caused by missing facts,
- endless retrieval loops,
- unnecessary tool calls,
- inconsistent LLM introspection.

It makes evidence completeness an explicit modeled quantity.

---

# 38. Important Failure Modes

We should expect several.

## 38.1 Relevance shortcut

Model learns:

> relevant-looking evidence = sufficient

Counter with hard negatives.

## 38.2 Length shortcut

Model learns:

> more evidence = more sufficient

Counter with:

- long insufficient examples,
- short sufficient examples.

## 38.3 Lexical overlap shortcut

Model learns:

> same words as question = sufficient

Counter with semantically relevant but unresolved evidence.

## 38.4 Dataset-generation artifacts

Synthetic examples may contain stylistic clues distinguishing positive and negative labels.

Counter with:

- multiple generators,
- paraphrasing,
- adversarial review,
- human-written validation examples.

## 38.5 Domain dependence

Sufficiency in medicine, finance, law, product support, and general QA may differ.

We need to determine whether one general model works or whether domain adapters are better.

## 38.6 Hidden reasoning requirements

Some evidence may technically contain all facts, but answering requires difficult reasoning.

We should decide whether "sufficient" means:

### Definition A

> All required facts are present.

or:

### Definition B

> A target downstream model can reliably answer from this evidence.

Those are not identical.

For an agent system, Definition B may ultimately be more useful.

---

# 39. A Stronger Operational Definition

An especially practical label definition is:

> Evidence is sufficient if the downstream target reasoner answers correctly and consistently when restricted to that evidence.

That allows us to generate labels empirically.

For example:

```text
question + evidence
      ↓
target frontier model
      ↓
answer
```

Run several times or with multiple strong models.

If competent reasoners reliably obtain the correct answer:

```text
sufficient = 1
```

If they cannot:

```text
sufficient = 0
```

This changes the problem from an abstract philosophical notion of sufficiency into an operationally measurable one.

---

# 40. GCP Is the Intended Build Environment

We explicitly want to build this on **Google Cloud Platform**.

The initial experiment does not require sophisticated infrastructure.

A very simple setup is sufficient:

```text
GCP Compute Engine GPU VM
        ↓
PyTorch
        ↓
Hugging Face Transformers
        ↓
ModernBERT-base
        ↓
LoRA + binary head
        ↓
training dataset in GCS
        ↓
checkpoints in GCS
```

---

# 41. Recommended Initial GCP Hardware

For a first pass:

```text
1× NVIDIA L4
```

is a sensible target.

ModernBERT-base + LoRA should be very manageable.

If we later full-fine-tune a larger model, move to:

```text
A100
```

or similar.

We do **not** need H100-class hardware for the initial model.

---

# 42. Why GPU Cost Is Not the Main Concern

Fine-tuning a pretrained encoder is dramatically cheaper than pretraining.

Our working expectation is that initial experiments should cost:

- single-digit dollars,
- or low tens of dollars,

rather than thousands.

The precise bill depends on:

- GPU type,
- sequence length,
- batch size,
- training set size,
- number of experiments,
- whether we use Spot VMs,
- whether we full-fine-tune or use LoRA.

Given that we already have substantial GCP credits, compute should not be the limiting factor.

The limiting factor will probably be:

\[
\boxed{\text{data quality}}
\]

not GPU budget.

---

# 43. Use Spot Instances Where Practical

For experimentation:

```text
GCP Spot GPU VM
```

can substantially reduce cost.

Because Spot instances can be preempted:

- checkpoint frequently,
- store checkpoints in GCS,
- make training resumable,
- save optimizer state if useful.

For a research prototype, this is entirely reasonable.

---

# 44. When to Move to Vertex AI

Do not begin with unnecessary infrastructure.

Start with Compute Engine.

Move to Vertex AI Custom Training when we want:

- repeatable managed jobs,
- hyperparameter sweeps,
- experiment tracking,
- reproducible environments,
- scheduled training,
- team workflows,
- easier productionization.

The model research should come first.

---

# 45. Suggested Repository Structure

```text
system1-sufficiency/
│
├── README.md
│
├── pyproject.toml
│
├── configs/
│   ├── base.yaml
│   └── lora.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
│
├── src/
│   ├── datasets/
│   │   ├── public_qa.py
│   │   ├── synthetic.py
│   │   └── hard_negatives.py
│   │
│   ├── model/
│   │   ├── sufficiency_model.py
│   │   └── calibration.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── scripts/
│   ├── generate_synthetic.py
│   ├── launch_gcp.sh
│   └── calibrate.py
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# 46. Concrete Version 0

The very first working model should intentionally be boring.

## Input

```text
[CLS]
question:
{question}

[SEP]

evidence:
{evidence}
[SEP]
```

## Encoder

```text
ModernBERT-base
```

## Output

```text
CLS hidden vector
↓
Linear(d, 1)
↓
sigmoid
↓
P(sufficient)
```

## Training

```text
binary cross-entropy
```

## Fine-tuning

```text
LoRA
```

## Deployment target

```text
single L4 or CPU depending on latency
```

This establishes the scientific baseline.

---

# 47. Concrete Version 1 Dataset Goal

A reasonable first serious dataset target:

```text
30k–100k examples
```

with deliberately balanced difficulty.

For example:

```text
20% obvious sufficient
20% obvious insufficient
25% relevant-but-insufficient
15% missing-one-critical-fact
10% distractor-heavy
10% multi-hop / adversarial
```

The exact distribution should be tuned after inspecting errors.

---

# 48. Evaluation Against an LLM

The most interesting benchmark is not merely:

> How accurate is the model?

It is:

> How often can this small non-generative model replace a frontier LLM control decision without increasing the error rate beyond an acceptable threshold?

Compare:

```text
Frontier LLM judge
vs.
System 1 sufficiency model
```

Measure:

- accuracy,
- false-sufficient rate,
- false-insufficient rate,
- calibration,
- p50 latency,
- p95 latency,
- cost per 1,000 decisions,
- throughput,
- downstream QA accuracy.

---

# 49. The Most Important Error: False Sufficient

For an agent, these two errors have different costs.

## False insufficient

Model says:

```text
not enough evidence
```

when evidence actually is sufficient.

Consequence:

- unnecessary retrieval,
- extra latency,
- extra cost.

Bad, but usually recoverable.

## False sufficient

Model says:

```text
enough evidence
```

when evidence is incomplete.

Consequence:

- downstream model answers prematurely,
- hallucination risk,
- factual error,
- potentially incorrect automated action.

Usually much worse.

Therefore the operating threshold should probably optimize:

\[
\text{very high precision on sufficient}
\]

rather than raw balanced accuracy.

For example:

```text
Only classify as sufficient when:
P(sufficient) > 0.98
```

The exact threshold should be chosen from validation data.

---

# 50. Selective Automation

This leads naturally to selective prediction.

Instead of forcing the model to act on every example:

```text
P > 0.98
→ sufficient

P < 0.20
→ clearly insufficient

otherwise
→ uncertain / use stronger model
```

This is exactly where calibrated System 1 models become powerful.

The system does not need the small model to solve every case.

It needs the small model to solve **easy cases very cheaply and know when not to trust itself**.

---

# 51. Longer-Term Version: Predict Missing Evidence

Once binary sufficiency works, we can extend the model.

Instead of only:

\[
P(\text{sufficient})
\]

predict:

\[
P(\text{sufficient})
\]

plus a representation of what is missing.

Examples:

```text
missing:
- temporal information
- identity/entity information
- numeric threshold
- causal evidence
- source provenance
- regulatory classification
```

This could directly steer retrieval.

Then the agent loop becomes:

```text
evidence
↓
System 1 model
↓
insufficient
+
"missing temporal information"
↓
retriever searches specifically
for date/time evidence
```

That becomes much more powerful.

But we should not start there.

---

# 52. Longer-Term Version: Query Generator + Sufficiency Gate

Another architecture:

```text
question
↓
retriever
↓
evidence
↓
sufficiency model
↓
if insufficient:
    small query-generation model
    or LLM
↓
retrieve missing evidence
↓
sufficiency model again
```

The expensive generative component is used only when the cheap decision model says it is necessary.

---

# 53. Relationship to Agent Harnesses

This entire direction fits naturally into an agent harness.

A modern agent can be viewed as:

```text
large reasoning model
        ↕
agent harness
        ↕
tools / filesystem / APIs / retrieval
```

System 1 models can live inside the harness as cheap control functions:

```text
tool router
risk gate
stop/continue
evidence sufficiency
verifier
retry decision
escalation decision
```

Then the large decoder is reserved for the places where generation and deep reasoning are genuinely required.

A future agent architecture might look like:

```text
                 ┌─────────────────────┐
                 │ Large reasoning LLM │
                 └──────────┬──────────┘
                            │
                     Agent harness
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
  tool router       sufficiency model       risk gate
  System 1          System 1                System 1
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                          tools
```

That is more compelling than using the same giant autoregressive model for every micro-decision.

---

# 54. The Deeper Principle

The big conceptual lesson from Jev is not merely:

> "Classification is cheaper than generation."

We already knew that.

The more interesting lesson is:

> **A pretrained semantic model can expose intelligence through a native output primitive other than language.**

Once that is accepted, many possibilities open up.

Instead of assuming:

\[
\text{intelligence}
=
\text{token generation}
\]

we can build:

\[
\text{intelligence}
\rightarrow
\text{probability}
\]

\[
\text{intelligence}
\rightarrow
\text{utility}
\]

\[
\text{intelligence}
\rightarrow
\text{ranking}
\]

\[
\text{intelligence}
\rightarrow
\text{stop signal}
\]

\[
\text{intelligence}
\rightarrow
\text{sufficiency}
\]

\[
\text{intelligence}
\rightarrow
\text{risk}
\]

The output representation should match the computation the application actually requires.

---

# 55. Why the Sufficiency Model Is a Good First Project

It satisfies several desirable properties:

## Narrow

We are not trying to recreate a generic foundation model.

## Useful

Retrieval and agent loops repeatedly face the question:

> Do we have enough information yet?

## Cheap

The model can be a relatively small encoder.

## Measurable

The output is one scalar probability.

## Calibratable

We can directly evaluate probability quality.

## Easy to integrate

The model simply gates whether the loop retrieves again or proceeds.

## Data can be generated systematically

Complete evidence can be degraded into incomplete evidence.

## Distinct from Jev

It is not merely another arbitrary-options engine.

## Suitable for GCP

Training and serving requirements are modest.

---

# 56. Recommended Build Order

## Step 1

Define precisely what "sufficient" means.

Recommended initial definition:

> Evidence is sufficient if a competent target reasoner can reliably answer the question correctly using only that evidence.

## Step 2

Create a hand-reviewed evaluation set before training.

Target:

```text
1k–3k examples
```

with many hard negatives.

## Step 3

Build a training-data pipeline from public QA and verification datasets.

## Step 4

Generate controlled synthetic examples.

Especially:

```text
complete evidence
→ remove one required fact
→ insufficient
```

## Step 5

Train:

```text
ModernBERT-base
+
LoRA
+
binary head
```

on a GCP L4.

## Step 6

Evaluate discrimination.

Can it separate:

```text
sufficient
vs.
insufficient
```

on truly unseen examples?

## Step 7

Evaluate calibration.

Measure:

- Brier,
- log loss,
- ECE,
- reliability curves.

## Step 8

Temperature-calibrate the model.

## Step 9

Benchmark against a frontier LLM judge.

Measure:

```text
quality
latency
cost
```

## Step 10

Insert it into a real retrieval loop.

Compare:

```text
LLM-only loop
vs.
System-1-gated loop
```

on end-to-end answer quality and cost.

## Step 11

Only then experiment with:

- full fine-tuning,
- larger ModernBERT,
- RLCD,
- missing-evidence prediction,
- domain specialization.

---

# 57. What Success Would Look Like

The project is successful if we can show something like:

```text
At a threshold of P(sufficient) > 0.98:

- false-sufficient rate is very low,
- the model handles X% of retrieval-loop decisions,
- p50 latency is dramatically below a frontier LLM,
- cost per decision is dramatically below a frontier LLM,
- end-to-end QA accuracy is unchanged or improved.
```

That would demonstrate the broader System 1 thesis:

> A large fraction of agent control decisions do not require autoregressive generation.

---

# 58. The Research Question

A concise framing for the project:

> **How often can a small non-generative semantic model replace a frontier LLM control decision inside an agent loop while maintaining a specified error rate?**

For our first model:

> **Can a pretrained encoder learn a calibrated estimate of whether a given evidence set is sufficient to answer a question?**

That is focused enough to build and benchmark, but broad enough to become a meaningful reusable primitive.

---

# 59. Final Mental Model

## LLM

```text
context
↓
decoder transformer
↓
fixed-vocabulary logits
↓
softmax over vocabulary
↓
sample next token
↓
repeat
↓
generated text / JSON / number
```

## Jev/Laya-style dynamic decision model

```text
state + question + runtime options
↓
semantic encoder
↓
contextual representation of each option
↓
shared scorer
↓
one logit per runtime option
↓
softmax across those options
↓
native probability distribution
```

## Our evidence-sufficiency System 1 model

```text
question + evidence
↓
ModernBERT
↓
CLS representation
↓
binary head
↓
one logit
↓
sigmoid
↓
P(sufficient)
```

That last model is the one we should build first.

---

# 60. One-Sentence Summary

**Jev's brilliant simplification is to stop using language generation as the universal output interface for intelligence; our first experiment takes the same philosophy and applies it to one extremely useful agent primitive—directly estimating the calibrated probability that the available evidence is sufficient to answer a question—using a pretrained encoder fine-tuned cheaply on GCP.**
