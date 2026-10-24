"""Simulation engines, built from an EnginePerf (the budgets derived in perf.py)."""
from __future__ import annotations

from collections import OrderedDict

from .core import EPS, Engine, Job


def interp(table: dict[int, float], n: int) -> float:
    keys = sorted(table)
    if not keys:
        return 0.0
    if n in table:
        return table[n]
    if n <= keys[0]:
        return table[keys[0]]
    if n >= keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k < n)
    hi = min(k for k in keys if k > n)
    return table[lo] + (table[hi] - table[lo]) * (n - lo) / (hi - lo)


def _prate(P: float, quad: float, tok: float, context: float) -> float:
    """Effective reading rate for `tok` new tokens over `context`: linear weights pass plus the
    attention products, which grow with the context."""
    secs = tok / P + quad * tok * context
    return tok / secs if secs > 0 and tok > 0 else P


class LRU:
    def __init__(self, cap: int):
        self.cap = max(0, cap)
        self.d: OrderedDict[str, None] = OrderedDict()

    def touch(self, key):
        if self.cap == 0:
            return
        self.d.pop(key, None)
        self.d[key] = None
        while len(self.d) > self.cap:
            self.d.popitem(last=False)

    def drop(self, key):
        self.d.pop(key, None)

    def __contains__(self, key):
        return key in self.d


class StrataEngine(Engine):
    """Admissions one at a time through the main session; slots write together; the prompt
    path gets 2/3 of the time beside writing slots (STRATA_BATCH_DECODE_SHARE 0.5)."""

    def __init__(self, perf, replay_thinking=False, no_replay_reread=7000, slot_prefix_bug=True):
        super().__init__(perf.engine_id)
        self.perf = perf
        self.slots = max(1, perf.slots)
        self.parking = LRU(perf.parking)
        self.slot_mem = LRU(self.slots if self.slots > 1 else 0)
        self.live = None
        self.live_sys = None
        self.P = max(1.0, perf.prefill_rate)
        self.D = perf.decode_total
        self.overhead_tokens = perf.overhead_s * self.P
        self.share = 0.5
        self.replay = replay_thinking
        self.no_replay_reread = no_replay_reread
        self.slot_bug = slot_prefix_bug
        self.reread_tokens = 0
        self.prefill_total = 0

    def _resident(self, conv):
        if conv == self.live or conv in self.parking:
            return True
        if conv in self.slot_mem:
            return not (self.slot_bug and not self.replay)
        return False

    def plan_prefill(self, job):
        r = job.req
        if self._resident(r.conv):
            tok = r.new if self.replay else min(r.context, r.new + self.no_replay_reread)
        else:
            job.reread = True
            tok = r.context
            if self.live_sys == r.sys_id:
                tok = max(r.new, r.context - r.meta.get("sys_tokens", 0))
        return int(tok)

    def admit(self, now):
        if not self.queue:
            return
        if self.slots == 1:
            if self.active:
                return
        else:
            if len(self.active) >= self.slots or any(j.prefill_left > EPS for j in self.active):
                return
        job = self.queue.pop(0)
        tok = self.plan_prefill(job)
        job.prefill_tokens = tok
        self.prefill_total += tok
        if job.reread:
            self.reread_tokens += tok
        job.prefill_left = tok + self.overhead_tokens
        job.prate = _prate(self.P, self.perf.prefill_quad, tok, job.req.context)
        job.out_left = job.req.out
        job.t_admit = now
        conv = job.req.conv
        if self.live is not None and self.live != conv:
            self.parking.touch(self.live)
        self.parking.drop(conv)
        self.live, self.live_sys = conv, job.req.sys_id
        self.active.append(job)
        self._admitted(job, now)

    def update_rates(self):
        self.rates.clear()
        pre = [j for j in self.active if j.prefill_left > EPS]
        dec = [j for j in self.active if j.prefill_left <= EPS]
        fp = 1.0 / (1.0 + self.share) if (pre and dec) else 1.0
        for j in pre:
            self.rates[id(j)] = (getattr(j, "prate", self.P) * fp, 0.0)
        if dec:
            total = interp(self.D, len(dec)) * ((1.0 - fp) if pre else 1.0)
            for j in dec:
                self.rates[id(j)] = (0.0, total / len(dec))

    def on_done(self, job):
        self.slot_mem.touch(job.req.conv)


class BatchEngine(Engine):
    """Continuous batching (vLLM), llama.cpp slots, hosted APIs and rerankers: up to `conc`
    requests at once, each writing at the per-request speed for that many writers; prompts
    read one at a time at the shared rate (hosted: in parallel)."""

    def __init__(self, perf, reuse=True):
        super().__init__(perf.engine_id)
        self.perf = perf
        self.conc = max(1, perf.max_parallel if perf.kind != "batch" else perf.slots)
        self.S = perf.speed_per_request
        self.P = max(1.0, perf.prefill_rate)
        self.overhead_tokens = (perf.overhead_s + perf.first_token_s) * self.P
        self.cache = LRU(perf.cache_convs if reuse else 0)
        self.window = perf.window
        self.hosted = perf.kind == "hosted"
        self.reread_tokens = 0
        self.prefill_total = 0
        self.tokens_in = 0
        self.tokens_cached = 0
        self.tokens_out = 0

    def submit(self, job, now):
        job.engine = self.name
        if self.window and job.req.context > self.window:
            job.skipped = True
            self.finished.append(job)
            return
        self.queue.append(job)

    def admit(self, now):
        while self.queue and len(self.active) < self.conc:
            job = self.queue.pop(0)
            r = job.req
            if r.conv in self.cache:
                tok = r.new
                self.tokens_cached += r.context - r.new
            else:
                tok = r.context
                job.reread = r.new < r.context
                if job.reread:
                    self.reread_tokens += tok
            self.tokens_in += tok
            self.tokens_out += r.out
            job.prefill_tokens = tok
            self.prefill_total += tok
            job.prefill_left = tok + self.overhead_tokens
            job.prate = _prate(self.P, self.perf.prefill_quad, tok, r.context)
            job.out_left = r.out
            job.t_admit = now
            self.active.append(job)
            self._admitted(job, now)

    def update_rates(self):
        self.rates.clear()
        pre = [j for j in self.active if j.prefill_left > EPS]
        dec = [j for j in self.active if j.prefill_left <= EPS]
        if pre:
            if self.hosted:
                for j in pre:
                    self.rates[id(j)] = (self.P, 0.0)
            else:
                self.rates[id(pre[0])] = (getattr(pre[0], "prate", self.P), 0.0)
        s = interp(self.S, len(dec)) if dec else 0.0
        for j in dec:
            self.rates[id(j)] = (0.0, s)

    def on_done(self, job):
        self.cache.touch(job.req.conv)
