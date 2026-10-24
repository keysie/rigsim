#!/usr/bin/env python3
"""Turnstone's request shape for the bench (pass 2, 2026-10-21): what Turnstone v1.8.5 puts in front of every model
call of a maker (interactive workstream) and of a sub-maker (task_agent), taken from Turnstone's own source.

  git clone --depth 1 --branch v1.8.5 https://github.com/turnstonelabs/turnstone.git /tmp/ts
  python3 infra/bench/tasks/turnstone-shape/make_shape.py --src /tmp/ts

Writes maker.json and sub.json here: {"source", "system", "skill_turn", "tools"} — the system message (Turnstone's
compose_system_message: persona base, env "web", context, tool patterns; the sub-agent's identity + operating
guidance), the skill (ours, infra/seed/skills: appended to the maker's system message; for a task agent its own user
turn with Turnstone's capability preamble) and the exact tool lanes (INTERACTIVE_TOOLS / TASK_AGENT_TOOLS, no MCP tools,
no cwd notes). Approximation: the skill placement for the maker and the omitted cwd notes / MCP tools; validated by
comparing the first-turn prompt_tokens with a real Turnstone request (usage report). The tool schemas and prompt texts
are Turnstone's (Apache License 2.0, see NOTICE here).
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = HERE.parents[2] / "seed" / "skills"
DROP = {"web_search"}          # no web search until leases (ops §16.10) — Turnstone hides it without a backend


def session_constants(path: Path) -> dict:
    """The task-agent prompt constants from session.py, read with ast (importing it needs Turnstone's dependencies)."""
    want = {"_TASK_DEFAULT_IDENTITY", "_TASK_OPERATING_GUIDANCE", "_TASK_SKILL_CAPABILITY_PREAMBLE"}
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = want - out.keys()
    if missing:
        raise SystemExit(f"not found in {path}: {sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Turnstone source checkout (tag v1.8.5)")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    from turnstone.core.tools import INTERACTIVE_TOOLS, TASK_AGENT_TOOLS          # noqa: E402
    from turnstone.prompts import SessionContext, ClientType, compose_system_message  # noqa: E402
    commit = subprocess.run(["git", "-C", a.src, "describe", "--tags", "--always"], capture_output=True,
                            text=True).stdout.strip()
    ctx = SessionContext(current_datetime="2026-10-21T12:00:00+02:00", timezone="CEST", username="user")
    skill = lambda n: json.loads((SEED / f"{n}.json").read_text())["content"]    # noqa: E731
    const = session_constants(Path(a.src) / "turnstone" / "core" / "session.py")

    tools_m = [t for t in INTERACTIVE_TOOLS if t["function"]["name"] not in DROP]
    sys_m = compose_system_message(client_type=ClientType.WEB, context=ctx,
                                   available_tools={t["function"]["name"] for t in tools_m}, kind="interactive")
    maker = {"source": f"turnstone {commit}", "system": sys_m + "\n\n" + skill("maker"), "skill_turn": "",
             "tools": tools_m}

    tools_s = [t for t in TASK_AGENT_TOOLS if t["function"]["name"] not in DROP]
    identity = const["_TASK_DEFAULT_IDENTITY"] + "\n\n" + const["_TASK_OPERATING_GUIDANCE"]
    sys_s = compose_system_message(client_type=ClientType.WEB, context=ctx,
                                   available_tools={t["function"]["name"] for t in tools_s}, kind="interactive",
                                   base_override=identity)
    sub = {"source": f"turnstone {commit}", "system": sys_s,
           "skill_turn": const["_TASK_SKILL_CAPABILITY_PREAMBLE"] + skill("sub-maker"), "tools": tools_s}
    for name, d in (("maker", maker), ("sub", sub)):
        (HERE / f"{name}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        print(f"{name}.json: system {len(d['system'])} chars, skill turn {len(d['skill_turn'])}, "
              f"{len(d['tools'])} tools ({len(json.dumps(d['tools']))} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
