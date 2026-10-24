"""Turnstone-shaped workload (v1.8.5 behaviour read from its source).

A workstream runs maker tasks back to back. Each maker turn writes an answer (mostly
thinking at effort low), then runs its tool calls in a 4-worker pool and waits for all of
them (fork-join). A tool call may need approval (judge first), runs, and its result goes
through the output guard. A `task_agent` call starts a sub-agent with its own conversation
on the sub-maker role; sub-agents cannot delegate. History grows until Turnstone compacts.

Task scripts are drawn before the run from a seed, so every layout sees the same work.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .core import Request

# ---------------------------------------------------------------- parameters

RANGE_KEYS = ["maker_turns", "answer_median", "tool_calls_mean", "p_delegate", "fanout", "tool_result_median",
              "tool_exec_median_s", "p_approval", "sub_turns", "sub_answer_median", "sub_tool_calls_mean",
              "sub_final_tokens", "task_tokens"]

FIXED = {
    "max_tokens": 8192,
    "compact_out": 1500,
    "call_tokens": 60,        # tool-call envelope per call in the history
    "pool": 4,                # tool calls run at once per turn (Turnstone task_agent pool)
    "judge_sys": 1500,
    "judge_read": 2048,       # each evidence read <= 2,048 tokens
    "guard_sys": 700,
    "approve_delay_s": 0.0,   # approvals resolved at once (the machine, not the human)
    "p_tool_search": 0.3,     # share of maker turns with a tool search (reranker)
    "rerank_candidates": 40,
    "rerank_pair_tokens": 250,
    "coord_plan_tokens": 6000,
    "coord_plan_out": 800,
    "coord_review_tokens": 8000,
    "coord_review_out": 500,
}


def sample_scenario(rng: random.Random, w) -> dict:
    """w: Workload (pydantic) with (lo, hi) ranges; ints stay ints."""
    s = {}
    for k in RANGE_KEYS:
        lo, hi = getattr(w, k)
        s[k] = rng.randint(int(lo), int(hi)) if isinstance(lo, int) and isinstance(hi, int) else rng.uniform(lo, hi)
    s.update(FIXED)
    s.update(maker_sys=w.maker_sys, sub_sys=w.sub_sys, window=w.window, compact_at=w.compact_at,
             visible_frac=w.visible_frac, tasks_per_ws=w.tasks_per_ws, judge_history_cap=w.window // 2,
             work_scale=float(getattr(w, "work_scale", 1.0) or 1.0), reruns=float(getattr(w, "reruns", 1.0) or 1.0))
    # the work lever scales what one task is: turns of the maker and of its sub-agents, and tool calls per turn
    if s["work_scale"] != 1.0:
        for k in ("maker_turns", "sub_turns"):
            s[k] = max(1, int(round(s[k] * s["work_scale"])))
        for k in ("tool_calls_mean", "sub_tool_calls_mean"):
            s[k] = s[k] * s["work_scale"] ** 0.5
    return s


# ---------------------------------------------------------------- scripts

@dataclass
class ToolCall:
    result: int
    exec_s: float
    approval: bool
    judge_turns: int


@dataclass
class AgentCall:
    task_tokens: int
    turns: list            # list[Turn]
    final: int


@dataclass
class Turn:
    answer: int
    calls: list = field(default_factory=list)
    search: bool = False


@dataclass
class Task:
    task_tokens: int
    turns: list
    reruns: int = 0           # extra attempts before the task counts


def _lognorm(rng, median, sigma, lo=1, hi=None):
    v = int(median * math.exp(rng.gauss(0.0, sigma)))
    v = max(lo, v)
    return min(v, hi) if hi else v


def _poisson(rng, mean, cap):
    # Knuth; small means only
    L, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return min(k, cap)
        k += 1


def _tool(rng, s, median_result):
    appr = rng.random() < s["p_approval"]
    return ToolCall(
        result=_lognorm(rng, median_result, 1.0, 20, 30000),
        exec_s=_lognorm(rng, s["tool_exec_median_s"] * 1000, 0.8, 50) / 1000.0,
        approval=appr,
        judge_turns=rng.randint(1, 5) if appr else 0,
    )


def make_task(rng: random.Random, s: dict) -> Task:
    turns = []
    n = max(2, int(s["maker_turns"]))
    for i in range(n):
        t = Turn(answer=_lognorm(rng, s["answer_median"], 0.8, 32, s["max_tokens"]),
                 search=rng.random() < s["p_tool_search"])
        if i < n - 1:
            if rng.random() < s["p_delegate"]:
                for _ in range(rng.randint(1, int(s["fanout"]))):
                    sub_turns = []
                    m = max(1, int(s["sub_turns"]))
                    for j in range(m):
                        st = Turn(answer=_lognorm(rng, s["sub_answer_median"], 0.8, 32, s["max_tokens"]))
                        if j < m - 1:
                            k = max(1, _poisson(rng, s["sub_tool_calls_mean"], s["pool"]))
                            st.calls = [_tool(rng, s, s["tool_result_median"]) for _ in range(k)]
                        sub_turns.append(st)
                    t.calls.append(AgentCall(task_tokens=_lognorm(rng, 400, 0.5, 50),
                                             turns=sub_turns,
                                             final=_lognorm(rng, s["sub_final_tokens"], 0.5, 50)))
            else:
                k = max(1, _poisson(rng, s["tool_calls_mean"], s["pool"]))
                t.calls = [_tool(rng, s, s["tool_result_median"]) for _ in range(k)]
        turns.append(t)
    return Task(task_tokens=_lognorm(rng, s["task_tokens"], 0.4, 50), turns=turns)


def make_scripts(seed: int, s: dict, n_ws: int) -> list[list[Task]]:
    """The reruns lever: a task counts once but is run `reruns` times on average (a weak model needs a
    second attempt). 1.5 means every other task runs twice; the extra runs are scheduled as work, not as tasks."""
    out = []
    reruns = float(s.get("reruns", 1.0) or 1.0)
    for w in range(n_ws):
        rng = random.Random(f"{seed}-{w}")
        tasks = []
        for _ in range(int(s["tasks_per_ws"])):
            t = make_task(rng, s)
            extra = int(reruns) - 1 + (1 if rng.random() < (reruns - int(reruns)) else 0)
            t.reruns = extra
            tasks.append(t)
        out.append(tasks)
    return out


# ---------------------------------------------------------------- processes

class Counter:
    def __init__(self):
        self.n = 0

    def next(self, prefix):
        self.n += 1
        return f"{prefix}{self.n}"


def _guard(s, ids, result, tag):
    if not s.get("_has_guard"):
        return
    ctx = s["guard_sys"] + result
    req = Request(conv=ids.next("g"), sys_id="guard", context=ctx, new=ctx, out=s["guard_out"],
                  kind="guard", meta={"tag": tag, "result": result})
    yield ("call", "guard", req)


def _judge(s, ids, maker_ctx, turns, tag):
    if not s.get("_has_judge"):
        return
    conv = ids.next("j")
    ctx = s["judge_sys"] + min(maker_ctx, s["judge_history_cap"])
    new = ctx
    for t in range(turns):
        req = Request(conv=conv, sys_id="judge", context=ctx, new=new, out=s["judge_out"],
                      kind="judge", meta={"tag": tag, "turn": t, "turns": turns})
        yield ("call", "judge", req)
        add = s["judge_out"] + (s["judge_read"] if t < turns - 1 else 0)
        ctx += add
        new = add


def _tool_proc(s, ids, call: ToolCall, maker_ctx, tag):
    if call.approval:
        yield from _judge(s, ids, maker_ctx, call.judge_turns, tag)
        if s["approve_delay_s"] > 0:
            yield ("sleep", s["approve_delay_s"])
    yield ("sleep", call.exec_s)
    yield from _guard(s, ids, call.result, tag)


def _chat(role, s, ids, conv, sys_id, sys_tokens, task_tokens, turns, kind, tag, log, compact_role=None, roles=None):
    """One agent conversation; returns nothing, logs per request."""
    window = s["window"]
    ctx = sys_tokens + task_tokens
    new = ctx
    gen = 0
    cur = conv
    for i, turn in enumerate(turns):
        req = Request(conv=cur, sys_id=sys_id, context=min(ctx, window), new=min(new, window),
                      out=turn.answer, kind=kind, meta={"tag": tag, "turn": i, "sys_tokens": sys_tokens})
        job = yield ("call", role, req)
        log.append(job)
        if roles and roles.get("reranker") and kind == "maker" and turn.search:
            yield from _rerank(s, ids, tag)
        vis = max(1, int(turn.answer * s["visible_frac"]))
        ctx += vis
        new = vis
        if not turn.calls:
            continue
        children = []
        for c in turn.calls:
            if isinstance(c, ToolCall):
                children.append(_tool_proc(s, ids, c, ctx, tag))
            else:
                children.append(_agent_proc(s, ids, c, tag, log, roles))
        # pool of 4: run in waves
        for k in range(0, len(children), s["pool"]):
            yield ("all", children[k:k + s["pool"]])
        add = 0
        for c in turn.calls:
            add += (c.result if isinstance(c, ToolCall) else c.final) + s["call_tokens"]
        ctx += add
        new += add
        if ctx > s["compact_at"] * window:
            creq = Request(conv=cur, sys_id=sys_id, context=min(ctx, window), new=min(new, window),
                           out=s["compact_out"], kind="compact", meta={"tag": tag, "sys_tokens": sys_tokens})
            job = yield ("call", compact_role or role, creq)
            log.append(job)
            gen += 1
            cur = f"{conv}~{gen}"                  # rewritten history: a new prefix after the system prompt
            ctx = sys_tokens + s["compact_out"]
            new = ctx


def _agent_proc(s, ids, call: AgentCall, tag, log, roles=None):
    conv = ids.next("s")
    yield from _chat("sub", s, ids, conv, "sub", s["sub_sys"], call.task_tokens, call.turns, "sub", tag, log, roles=roles)
    yield from _guard(s, ids, call.final, tag)


def _rerank(s, ids, tag):
    tok = s["rerank_candidates"] * s["rerank_pair_tokens"]
    req = Request(conv=ids.next("r"), sys_id="rerank", context=tok, new=tok, out=0, kind="rerank", meta={"tag": tag})
    yield ("call", "reranker", req)


def _coord(s, ids, conv, ctx, out, tag):
    req = Request(conv=conv, sys_id="coord", context=ctx, new=ctx, out=out, kind="coordinator", meta={"tag": tag})
    yield ("call", "coordinator", req)


def workstream(s, ids, w, tasks, log, task_log, now, roles):
    for ti, task in enumerate(tasks):
        t0 = now()
        for attempt in range(1 + int(getattr(task, "reruns", 0))):      # reruns: the same task again, fresh conversation
            conv = ids.next(f"m{w}-")
            if roles.get("coordinator"):
                yield from _coord(s, ids, f"c{w}-{ti}", s["coord_plan_tokens"], s["coord_plan_out"], (w, ti))
            yield from _chat("maker", s, ids, conv, "maker", s["maker_sys"], task.task_tokens, task.turns,
                             "maker", (w, ti), log, roles=roles)
            if roles.get("coordinator"):
                yield from _coord(s, ids, f"c{w}-{ti}", s["coord_review_tokens"], s["coord_review_out"], (w, ti))
        task_log.append({"ws": w, "task": ti, "start": t0, "end": now(), "attempts": 1 + int(getattr(task, "reruns", 0))})
