# Research sources and what we took from them

## Agents and tools
- Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
  Simplest solution first; workflows vs agents; start on the API directly; invest in tool design (ACI).
- Writing tools for agents — https://www.anthropic.com/engineering/writing-tools-for-agents
  Few consolidated tools; namespacing; meaningful readable context; concise/detailed modes; actionable errors; evaluate.
- Effective context engineering — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  "Right altitude" prompts; start minimal, add for observed failures; sections; few canonical examples; just-in-time context.
- Multi-agent research system — https://www.anthropic.com/engineering/multi-agent-research-system
  Multi-agent ≈15× tokens; good for parallel breadth, bad for shared-context work; ~20 test queries show big effects;
  "think like your agents"; scale effort to query complexity; let Claude rewrite tool descriptions (−40% time).
- OpenAI practical guide (summary) — https://www.maginative.com/article/how-to-build-ai-agents-a-detailed-practical-guide-from-openai/
  Start single agent; most capable model first; layered guardrails; human intervention on repeated failure.
- Claude Agent SDK — https://code.claude.com/docs/en/agent-sdk/overview ; production notes:
  https://inference.net/content/claude-agent-sdk-production-guide/ (max_turns, cost logging, tracing).

## Data / semantic layer
- Why text-to-SQL fails — https://omni.co/blog/why-text-to-sql-fails  → governed metric definitions (semantic layer).

## Video analytics
- AVA (Microsoft Research) — https://www.microsoft.com/en-us/research/?p=1148911  → event knowledge graph + agentic retrieval.
- LLoVi https://arxiv.org/abs/2312.17235, VideoAgent https://arxiv.org/abs/2403.11481 and https://arxiv.org/abs/2403.10517,
  VideoTree https://arxiv.org/abs/2405.19209, DoraemonGPT https://arxiv.org/abs/2401.08392, Video-RAG https://arxiv.org/abs/2411.13093
  → structured text/event memory + tools beat raw frames.
- Plots vs text for numeric series: https://arxiv.org/abs/2410.02637 (plots can beat long raw number lists → give intervals, not raw series).

## People counting / tracking
- Ultralytics counting — https://docs.ultralytics.com/guides/object-counting/ ; Xovis — https://www.xovis.com/
- HOTA https://arxiv.org/abs/2009.07736 ; IDF1 https://arxiv.org/abs/1609.01775 ; Ma & Chan line crossing (CVPR 2013).
  → counts depend on detection; paths/dwell depend on association; line crossings survive most ID switches.

## Claude Code session transfer
- Teleport — https://code.claude.com/docs/en/claude-code-on-the-web#from-cloud-to-terminal
- Remote Control — https://code.claude.com/docs/en/remote-control
