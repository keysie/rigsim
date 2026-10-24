"""Event-driven fluid simulation.

Engines hold jobs whose work (prefill tokens, output tokens) drains at rates that are
constant between events. At every event each engine recomputes its rates; the next
event is the earliest job completion or timer. Agents are generator processes that
yield commands:

    ("call", engine_role, Request)   -> resumes with the finished Job
    ("sleep", seconds)               -> resumes after the delay
    ("all", [generator, ...])        -> resumes when every child process finished
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

EPS = 1e-9


@dataclass
class Request:
    conv: str               # conversation id (for history reuse)
    sys_id: str             # system-prompt family (root-checkpoint reuse)
    context: int            # tokens of the whole prompt
    new: int                # tokens added since this conversation's last request
    out: int                # tokens to write
    kind: str               # maker / sub / compact / guard / judge
    meta: dict = field(default_factory=dict)


@dataclass
class Job:
    req: Request
    t_submit: float
    prefill_left: float = 0.0
    out_left: float = 0.0
    prefill_tokens: int = 0
    reread: bool = False
    t_admit: float | None = None
    t_first: float | None = None
    t_done: float | None = None
    engine: str = ""
    skipped: bool = False   # e.g. guard prompt over its window (HTTP 400)
    _proc: object = None


class Engine:
    """Base class. Subclasses set rates in `update_rates` and admit in `admit`."""

    name = "engine"

    def __init__(self, name: str):
        self.name = name
        self.queue: list[Job] = []
        self.active: list[Job] = []
        self.rates: dict[int, tuple[float, float]] = {}   # id(job) -> (prefill rate, out rate)
        self.busy_time = 0.0
        self.work_time = 0.0      # integral of active jobs over time
        self.finished: list[Job] = []
        self.trace = None

    def _admitted(self, job: Job, now: float) -> None:
        if self.trace:
            self.trace({"t": now, "ev": "admit", "engine": self.name, "kind": job.req.kind, "conv": job.req.conv,
                        "prefill": job.prefill_tokens, "out": job.req.out, "reread": job.reread})

    # --- to override -------------------------------------------------------
    def plan_prefill(self, job: Job) -> int:
        return job.req.context

    def admit(self, now: float) -> None:
        raise NotImplementedError

    def update_rates(self) -> None:
        raise NotImplementedError

    def on_done(self, job: Job) -> None:
        pass

    # --- shared -----------------------------------------------------------
    def submit(self, job: Job, now: float) -> None:
        job.engine = self.name
        self.queue.append(job)

    def load(self) -> int:
        return len(self.queue) + len(self.active)

    def advance(self, dt: float) -> None:
        if self.active:
            self.busy_time += dt
            self.work_time += dt * len(self.active)
        for job in self.active:
            pr, orate = self.rates.get(id(job), (0.0, 0.0))
            if job.prefill_left > EPS:
                job.prefill_left = max(0.0, job.prefill_left - pr * dt)
            else:
                job.out_left = max(0.0, job.out_left - orate * dt)

    def next_completion(self) -> float:
        best = float("inf")
        for job in self.active:
            pr, orate = self.rates.get(id(job), (0.0, 0.0))
            if job.prefill_left > EPS:
                if pr > 0:
                    best = min(best, job.prefill_left / pr)
            elif orate > 0:
                best = min(best, job.out_left / orate)
        return best

    def collect(self, now: float) -> list[Job]:
        done = []
        for job in list(self.active):
            if job.prefill_left <= EPS and job.t_first is None:
                job.t_first = now
            if job.prefill_left <= EPS and job.out_left <= EPS:
                job.t_done = now
                self.active.remove(job)
                self.on_done(job)
                self.finished.append(job)
                done.append(job)
        return done


class Sim:
    def __init__(self, engines: dict[str, list[Engine]], t_max: float = 6 * 3600.0, trace=None):
        self.trace = trace                # optional callback(event: dict)
        self.engines = engines            # role -> list of engines (least loaded wins)
        self.all_engines = list({id(e): e for es in engines.values() for e in es}.values())
        for e in self.all_engines:
            e.trace = trace
        self.now = 0.0
        self.t_max = t_max
        self.timers: list[tuple[float, int, object]] = []
        self._seq = itertools.count()
        self.ready: list[tuple[object, object]] = []
        self.parents: dict[int, list] = {}   # child gen id -> [parent gen, counter cell]
        self.calls: list[Job] = []
        self.stopped = False

    # --- processes ----------------------------------------------------------
    def spawn(self, gen, parent=None):
        if parent is not None:
            self.parents[id(gen)] = parent
        self.ready.append((gen, None))

    def _step_proc(self, gen, value):
        try:
            cmd = gen.send(value)
        except StopIteration:
            par = self.parents.pop(id(gen), None)
            if par is not None:
                pgen, cell = par
                cell[0] -= 1
                if cell[0] == 0:
                    self.ready.append((pgen, None))
            return
        op = cmd[0]
        if op == "sleep":
            heapq.heappush(self.timers, (self.now + max(0.0, cmd[1]), next(self._seq), gen))
        elif op == "call":
            role, req = cmd[1], cmd[2]
            eng = min(self.engines[role], key=lambda e: e.load())
            job = Job(req=req, t_submit=self.now)
            job._proc = gen
            eng.submit(job, self.now)
            self.calls.append(job)
            if self.trace:
                self.trace({"t": self.now, "ev": "submit", "engine": eng.name, "kind": req.kind, "conv": req.conv,
                            "context": req.context, "out": req.out, "tag": str(req.meta.get("tag", ""))})
            if job.skipped:            # engine refused at once
                job.t_done = self.now
                self.ready.append((gen, job))
        elif op == "all":
            children = cmd[1]
            if not children:
                self.ready.append((gen, None))
                return
            cell = [len(children)]
            for ch in children:
                self.spawn(ch, parent=(gen, cell))
        else:
            raise ValueError(op)

    def _drain_ready(self):
        while self.ready:
            gen, val = self.ready.pop(0)
            self._step_proc(gen, val)

    # --- main loop ---------------------------------------------------------
    def run(self):
        while True:
            self._drain_ready()
            for e in self.all_engines:
                e.admit(self.now)
                e.update_rates()
            dt = min([e.next_completion() for e in self.all_engines] + [float("inf")])
            if self.timers:
                dt = min(dt, self.timers[0][0] - self.now)
            if dt == float("inf"):
                break
            dt = max(dt, 0.0)
            if self.now + dt > self.t_max:
                self.stopped = True
                break
            for e in self.all_engines:
                e.advance(dt)
            self.now += dt
            for e in self.all_engines:
                for job in e.collect(self.now):
                    self.ready.append((job._proc, job))
                    if self.trace:
                        self.trace({"t": self.now, "ev": "done", "engine": e.name, "kind": job.req.kind, "conv": job.req.conv,
                                    "wait": round((job.t_admit if job.t_admit is not None else self.now) - job.t_submit, 2),
                                    "prefill": job.prefill_tokens, "out": job.req.out, "reread": job.reread})
            while self.timers and self.timers[0][0] <= self.now + EPS:
                _, _, gen = heapq.heappop(self.timers)
                self.ready.append((gen, None))
        return self
