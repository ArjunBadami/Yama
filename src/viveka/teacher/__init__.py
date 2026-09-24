"""Teacher models: large models used only at data-creation time, never at runtime.

Two jobs:
  label.py     -> operational (Definition B) sufficiency labels: "does a strong
                  reasoner, restricted to this evidence, reliably answer correctly?"
  synthetic.py -> generate new (question, required facts, distractors) tuples that
                  the curriculum in datasets/degrade.py then expands.
"""

from .client import CachedTeacher, GeminiTeacher, Teacher, make_teacher

__all__ = ["CachedTeacher", "GeminiTeacher", "Teacher", "make_teacher"]
