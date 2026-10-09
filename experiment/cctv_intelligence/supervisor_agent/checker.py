"""Answer checker: every time and number in the answer must appear in some tool output.
Cheap guardrail against invented numbers (code, not an LLM)."""
from __future__ import annotations

import re

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
NUM_RE = re.compile(r"(?<![\w:.])\d+(?:\.\d+)?(?![\w:])")
IGNORE_NUMS = {"0", "1", "2", "3", "4", "5"}   # small counts and list markers are too common to check


def _norm_time(t: str) -> str:
    p = t.split(":")
    return f"{int(p[0]):02d}:{p[1]}" + (f":{p[2]}" if len(p) > 2 else "")


def unsupported(answer: str, tool_texts: list[str]) -> list[str]:
    corpus = "\n".join(tool_texts)
    corpus_times = {_norm_time(t) for t in TIME_RE.findall(corpus)}
    corpus_hm = {t[:5] for t in corpus_times}
    bad = []
    for t in TIME_RE.findall(answer):
        n = _norm_time(t)
        if n not in corpus_times and (len(n) > 5 or n not in corpus_hm):
            bad.append(t)
    stripped = TIME_RE.sub(" ", answer)
    corpus_nums = set(NUM_RE.findall(TIME_RE.sub(" ", corpus)))
    for n in NUM_RE.findall(stripped):
        if n in IGNORE_NUMS or n in corpus_nums:
            continue
        # allow simple rounding of a corpus value (e.g. 4.3 -> 4)
        try:
            v = float(n)
        except ValueError:
            continue
        cn = [float(c) for c in corpus_nums if c.replace(".", "", 1).isdigit()]
        if any(abs(c - v) <= 0.5 for c in cn) or any(abs(c / 60 - v) <= 0.15 for c in cn):
            continue   # rounding, or seconds shown as minutes
        bad.append(n)
    return sorted(set(bad))
