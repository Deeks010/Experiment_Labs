"""Supervisor agent: Claude + the 7 tools, one main loop.

  export ANTHROPIC_API_KEY=...   CCTV_DB_PATH=...   [SA_MODEL=claude-...]
  python -m supervisor_agent.agent "Was Op 15 manned between 10:04 and 10:07 today?"
  python -m supervisor_agent.agent            # interactive chat

Safety nets: max tool calls, max seconds, max cost estimate; a code checker that flags numbers
not found in any tool output and asks for one revision.
`run_turn()` yields events {type: status|tool|image|chunk|done|error} like agent_web_server.py's NDJSON stream,
so the existing web UI can use it later.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from datetime import date
from pathlib import Path

from . import db as D
from . import tools as T
from .checker import unsupported
from .tool_specs import SPECS

MODEL = os.environ.get("SA_MODEL", "claude-sonnet-4-5")
MAX_TOOL_CALLS = int(os.environ.get("SA_MAX_TOOL_CALLS", 20))
MAX_SECONDS = int(os.environ.get("SA_MAX_SECONDS", 240))
MAX_TOKENS_OUT = 2500
LOG_DIR = Path(os.environ.get("SA_LOG_DIR") or (D.PKG_DIR / "agent_logs"))


def system_prompt() -> str:
    sem = D.load_semantics()
    tmpl = (D.PKG_DIR / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    defs = "; ".join(f"{k}: {v}" for k, v in sem["definitions"].items())
    shifts = ", ".join(f"{s['name']} {s['start']}–{s['end']}" for s in sem["shifts"])
    return (tmpl.replace("{today}", os.environ.get("SA_TODAY", date.today().isoformat()))
            .replace("{timezone}", os.environ.get("SA_TIMEZONE", "site local time"))
            .replace("{definitions}", defs).replace("{shifts}", shifts))


def _image_block(path: str) -> dict:
    mt = mimetypes.guess_type(path)[0] or "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": mt,
                                        "data": base64.b64encode(Path(path).read_bytes()).decode()}}


def run_turn(question: str, history: list | None = None, client=None, model: str = MODEL):
    """Generator of events. history (list of messages) is extended in place for multi-turn chat."""
    import anthropic
    client = client or anthropic.Anthropic()
    messages = history if history is not None else []
    messages.append({"role": "user", "content": question})
    conn = D.connect()
    sys_p = system_prompt()
    tool_texts, calls, t0, revised = [], 0, time.time(), False
    usage = {"input_tokens": 0, "output_tokens": 0}
    log = {"question": question, "model": model, "steps": []}
    while True:
        over = calls >= MAX_TOOL_CALLS or time.time() - t0 > MAX_SECONDS
        kwargs = dict(model=model, max_tokens=MAX_TOKENS_OUT, system=sys_p, tools=SPECS, messages=messages)
        if over:
            kwargs["tool_choice"] = {"type": "none"}
        resp = client.messages.create(**kwargs)
        usage["input_tokens"] += resp.usage.input_tokens; usage["output_tokens"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        if resp.stop_reason == "tool_use":
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                calls += 1
                yield {"type": "tool", "name": b.name, "input": b.input}
                if calls > MAX_TOOL_CALLS:
                    res = {"text": "Tool budget reached. Answer now with what you have and say what is missing.", "images": []}
                else:
                    res = T.call(conn, b.name, b.input)
                tool_texts.append(res["text"])
                content = [{"type": "text", "text": res["text"]}] + [_image_block(p) for p in res["images"] if Path(p).exists()]
                for p in res["images"]:
                    yield {"type": "image", "path": p}
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
                log["steps"].append({"tool": b.name, "input": b.input, "output": res["text"][:4000], "images": res["images"]})
            messages.append({"role": "user", "content": results})
            continue
        answer = "".join(b.text for b in resp.content if b.type == "text")
        bad = unsupported(answer, tool_texts) if tool_texts else []
        if bad and not revised:
            revised = True
            yield {"type": "status", "message": f"checking numbers not found in tool outputs: {', '.join(bad)}"}
            messages.append({"role": "user", "content":
                             "[automatic check] These numbers/times in your answer were not found in any tool output: "
                             f"{', '.join(bad)}. If a value is computed from tool values (for example a duration), keep it and show the "
                             "source values. Otherwise correct or remove it. Reply with the full corrected answer only."})
            continue
        log.update(answer=answer, tool_calls=calls, seconds=round(time.time() - t0, 1), usage=usage, unsupported=bad)
        _write_log(log)
        yield {"type": "chunk", "text": answer}
        yield {"type": "done", "tool_calls": calls, "usage": usage, "unsupported": bad}
        return


def _write_log(log: dict):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{date.today().isoformat()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, default=str) + "\n")
    except OSError:
        pass


def ask(question: str, history=None, verbose=True) -> str:
    final = ""
    for ev in run_turn(question, history):
        if ev["type"] == "tool" and verbose:
            print(f"  → {ev['name']}({json.dumps(ev['input'])})")
        elif ev["type"] == "status" and verbose:
            print(f"  · {ev['message']}")
        elif ev["type"] == "chunk":
            final = ev["text"]
        elif ev["type"] == "done" and verbose:
            print(f"  [{ev['tool_calls']} tool calls, {ev['usage']['input_tokens']} in / {ev['usage']['output_tokens']} out tokens]")
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    a = ap.parse_args()
    if a.question:
        print(ask(a.question)); return
    history = []
    print("Supervisor agent. Empty line to quit.")
    while True:
        q = input("\nQ> ").strip()
        if not q:
            break
        print("\n" + ask(q, history))


if __name__ == "__main__":
    main()
