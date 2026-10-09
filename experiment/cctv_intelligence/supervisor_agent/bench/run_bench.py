"""Run the bench: the agent answers every question, a separate Claude call grades it with the rubric.

  export ANTHROPIC_API_KEY=... CCTV_DB_PATH=sim_data/sim.sqlite3 SA_TODAY=2026-10-09
  python -m supervisor_agent.bench.run_bench [--only q01,q02] [--model claude-...] [--judge-model claude-...]

Writes bench_results/<timestamp>.json and prints a table. Run after every prompt / tool / model change.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .. import agent as A

HERE = Path(__file__).resolve().parent

JUDGE = """You grade an answer from a factory CCTV assistant against the known truth.
Question: {q}
Truth: {truth}
Rubric: {rubric}
Answer:
<<<
{answer}
>>>
Also set invented=true if the answer states as fact any specific detail that contradicts the truth or is not supported
(invented people, activities, reasons, identities, wrong counts stated with certainty).
Reply with JSON only: {{"score": 0|1|2, "invented": true|false, "note": "max 20 words"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--model", default=A.MODEL)
    ap.add_argument("--judge-model", default=os.environ.get("SA_JUDGE_MODEL", A.MODEL))
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic()
    spec = json.loads((HERE / "questions.json").read_text())
    os.environ.setdefault("SA_TODAY", spec["today"])
    qs = [q for q in spec["questions"] if not a.only or q["id"] in a.only.split(",")]
    results = []
    for q in qs:
        t0 = time.time()
        events = list(A.run_turn(q["q"], [], client=client, model=a.model))
        answer = next((e["text"] for e in events if e["type"] == "chunk"), "")
        done = next((e for e in events if e["type"] == "done"), {})
        g = client.messages.create(model=a.judge_model, max_tokens=300, messages=[{"role": "user", "content": JUDGE.format(
            q=q["q"], truth=q["truth"], rubric=q["rubric"], answer=answer)}])
        txt = "".join(b.text for b in g.content if b.type == "text")
        try:
            grade = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        except ValueError:
            grade = {"score": 0, "invented": False, "note": "judge parse error"}
        r = dict(id=q["id"], q=q["q"], answer=answer, grade=grade, tool_calls=done.get("tool_calls"),
                 usage=done.get("usage"), unsupported=done.get("unsupported"), seconds=round(time.time() - t0, 1),
                 tools=[e["name"] for e in events if e["type"] == "tool"])
        results.append(r)
        print(f"{q['id']}  score {grade.get('score')}  invented {grade.get('invented')}  calls {r['tool_calls']}  {grade.get('note')}")
    tot = sum(r["grade"].get("score", 0) for r in results)
    print(f"\nTOTAL {tot}/{2 * len(results)}   invented: {sum(1 for r in results if r['grade'].get('invented'))}")
    out = HERE.parent / "bench_results"; out.mkdir(exist_ok=True)
    (out / f"{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
