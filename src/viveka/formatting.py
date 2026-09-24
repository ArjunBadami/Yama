"""Turn an Example into the text the encoder sees.

Version 0 input layout (the tokenizer adds [CLS]/[SEP]):

    [CLS] question: {question} [SEP] evidence:
    - {evidence 1}
    - {evidence 2}
    ... [SEP]

We pass (question_text, evidence_text) as a *text pair* so the tokenizer's
pair handling and truncation strategy apply. Truncation only ever removes
evidence (the second segment), never the question.
"""

from __future__ import annotations

from typing import Any

from .schema import Example

QUESTION_PREFIX = "question: "
EVIDENCE_PREFIX = "evidence:\n"
EVIDENCE_BULLET = "- "
NO_EVIDENCE = "(no evidence)"


def format_pair(question: str, evidence: list[str]) -> tuple[str, str]:
    q = QUESTION_PREFIX + question.strip()
    if evidence:
        e = EVIDENCE_PREFIX + "\n".join(EVIDENCE_BULLET + s.strip() for s in evidence if s.strip())
    else:
        e = EVIDENCE_PREFIX + NO_EVIDENCE
    return q, e


def format_example(ex: Example) -> tuple[str, str]:
    return format_pair(ex.question, ex.evidence)


def encode_batch(
    tokenizer: Any,
    questions: list[str],
    evidences: list[list[str]],
    max_length: int,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    firsts, seconds = [], []
    for q, e in zip(questions, evidences, strict=False):
        a, b = format_pair(q, e)
        firsts.append(a)
        seconds.append(b)
    return tokenizer(
        firsts,
        seconds,
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors=return_tensors,
    )
