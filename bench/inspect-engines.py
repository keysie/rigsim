#!/usr/bin/env python3
"""Inspect the box's engines for the benchmarking stage's first step (agentic plan §9; software-stack EF1, SP1).
Read-only. Runs on the host that serves the engines. Standard library only.

Per engine (one ingress route each):
  EF1  the served chat template — from the engine (/props: llama.cpp and Strata) or the model's files (vLLM: the HF
       cache's chat_template.jinja / tokenizer_config.json) — and every line about effort / reasoning / thinking, with
       the quoted sentences near them (the "low effort" wording goes verbatim into the bench plans and skill-prompts §2);
  SP1  the sampling actually in force: engine flags in the container's command line (llama.cpp --temp/--top-k/…,
       vLLM --override-generation-config / --generation-config), the model's generation_config.json (vLLM applies it
       by default), the engine's own defaults (/props default_generation_settings), Strata's CONFIG file.
Turnstone sends no sampling (temperature empty) — so what is listed here is what every request gets.

  inspect-engines.py --ingress https://NAME:8443 --resolve NAME:8443:127.0.0.1 --engine ROUTE=SERVICE ... \
                     --models-dir /srv/ai/models --infra /path/to/infra [--json]
Route keys: env INGRESS_KEY_<ROUTE> (upper case).

--effort-probe ROUTE (repeatable, 2026-10-20): which effort channel the engine honours. The same small reasoning task
is sent N times (--effort-samples, default 3) each with: nothing; top-level `reasoning_effort: low`;
`chat_template_kwargs: {reasoning_effort: low}`; the template's low sentence as a system-prompt prefix; and
`chat_template_kwargs: {enable_thinking: false}`. Reported: median reasoning size (reasoning_tokens if the engine
reports them, else characters / 4) and completion tokens. A channel that does not move the numbers is not honoured."""
import argparse, glob, json, os, pathlib, re, socket, ssl, subprocess, sys, urllib.error, urllib.request

SAMPLING_FLAGS = ("--temp", "--temperature", "--top-k", "--top-p", "--min-p", "--presence-penalty", "--repeat-penalty",
                  "--frequency-penalty", "--override-generation-config", "--generation-config", "--sampling-seq",
                  "--samplers", "--reasoning-budget", "--reasoning-format", "--chat-template-kwargs", "--jinja")
SAMPLING_KEYS = ("temperature", "temp", "top_k", "top_p", "min_p", "presence_penalty", "repeat_penalty",
                 "repetition_penalty", "frequency_penalty", "samplers", "reasoning", "thinking", "effort")
EFFORT_RX = re.compile(r"effort|reasoning|thinking|enable_thinking|budget", re.I)


def get(url, key, timeout=20):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"{type(e).__name__}", None


def container_cmd(project, service):
    try:
        out = subprocess.run(["docker", "inspect", f"{project}-{service}-1"], capture_output=True, text=True, timeout=20)
        c = json.loads(out.stdout)[0]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, {}
    ep = list((c.get("Config") or {}).get("Entrypoint") or [])
    cmd = list((c.get("Config") or {}).get("Cmd") or [])
    return ep + cmd, {"cmd": cmd}


def flags(args):
    """Sampling-related flags and their values from a command line (both `--x v` and `--x=v`)."""
    out = {}
    for i, a in enumerate(args or []):
        name, _, val = a.partition("=")
        if name in SAMPLING_FLAGS:
            out[name] = val if val else (args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True)
    return out


def hf_files(models_dir, model_ref):
    """generation_config.json and the chat template of an HF model in the local cache (vLLM's model argument)."""
    if not model_ref:
        return {}
    cand = []
    if os.path.isdir(model_ref):
        cand.append(model_ref)
    org_name = model_ref.strip("/").replace("/", "--")
    cand += sorted(glob.glob(os.path.join(models_dir, "hf", "**", f"models--{org_name}", "snapshots", "*"), recursive=True))
    out = {}
    for d in cand:
        for f in ("generation_config.json", "chat_template.jinja", "tokenizer_config.json"):
            p = os.path.join(d, f)
            if f not in out and os.path.isfile(p):
                out[f] = p
    return out


def template_lines(tpl, context=1):
    """Lines of a chat template that are about effort / reasoning / thinking, with a little context."""
    if not tpl:
        return []
    lines = tpl.splitlines()
    keep = set()
    for i, l in enumerate(lines):
        if EFFORT_RX.search(l):
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    return [f"{i + 1:4d}  {lines[i]}" for i in sorted(keep)]   # untruncated (2026-10-20: a 200-char cut read like a typo)


def quoted_sentences(tpl):
    """Quoted strings in the template that read like an instruction about effort or thinking (the 'low' wording)."""
    out = []
    for m in re.finditer(r"'([^'\n]{12,400})'|\"([^\"\n]{12,400})\"", tpl or ""):
        s = m.group(1) or m.group(2)
        if EFFORT_RX.search(s) or re.search(r"\b(low|medium|high|brief|concise|quick)\b", s, re.I):
            if " " in s and s not in out:
                out.append(s)
    return out


def inspect(a, route, service):
    key = os.environ.get(f"INGRESS_KEY_{route.upper()}", "")
    base = f"{a.ingress.rstrip('/')}/{route}"
    rep = {"route": route, "service": service}
    st, models = get(f"{base}/v1/models", key)
    rep["models_status"] = st
    rep["model"] = ((models or {}).get("data") or [{}])[0].get("id") if isinstance(models, dict) else None
    st, props = get(f"{base}/props", key)
    rep["props_status"] = st
    args, env = container_cmd(a.project, service)
    rep["engine_flags"] = flags(args)
    rep["command"] = " ".join(args or [])[:600]
    tpl, tpl_src = None, None
    if isinstance(props, dict):
        tpl, tpl_src = props.get("chat_template"), "engine /props"
        dg = props.get("default_generation_settings") or {}
        params = dg.get("params", dg)
        rep["engine_defaults"] = {k: v for k, v in params.items() if any(s in k for s in SAMPLING_KEYS)}
    files = {}
    if args and service.startswith("vllm"):                     # vLLM: first positional argument = the model
        # the model is the first positional argument of the compose `command:` (Cmd), not of the image's entrypoint
        ref = next((x for x in env.get("cmd", []) if not x.startswith("-") and x not in ("serve", "vllm")), "")
        rep["model_ref"] = ref
        files = hf_files(a.models_dir, ref)
        if "generation_config.json" in files:
            try:
                gc = json.loads(pathlib.Path(files["generation_config.json"]).read_text())
                rep["model_generation_config"] = {k: v for k, v in gc.items() if any(s in k for s in SAMPLING_KEYS)}
            except ValueError:
                pass
        if not tpl and "chat_template.jinja" in files:
            tpl, tpl_src = pathlib.Path(files["chat_template.jinja"]).read_text(), files["chat_template.jinja"]
        if not tpl and "tokenizer_config.json" in files:
            try:
                ct = json.loads(pathlib.Path(files["tokenizer_config.json"]).read_text()).get("chat_template")
                tpl = ct if isinstance(ct, str) else json.dumps(ct)
                tpl_src = files["tokenizer_config.json"]
            except ValueError:
                pass
    if service == "strata":
        cfg_path = os.path.join(a.infra, "strata", "strata-maker.json")
        try:
            cfg = json.loads(pathlib.Path(cfg_path).read_text())
            flat = {}
            def walk(d, p=""):
                for k, v in (d.items() if isinstance(d, dict) else []):
                    q = f"{p}.{k}" if p else k
                    if isinstance(v, dict):
                        walk(v, q)
                    elif any(s in k.lower() for s in SAMPLING_KEYS) or "chat_template" in k.lower():
                        flat[q] = v
            walk(cfg)
            rep["strata_config"] = flat
        except (OSError, ValueError) as e:
            rep["strata_config"] = f"{type(e).__name__}: {cfg_path}"
    rep["template_source"] = tpl_src
    rep["template_effort_lines"] = template_lines(tpl)
    rep["template_quoted"] = quoted_sentences(tpl)
    rep["template_kwargs_seen"] = sorted(set(re.findall(r"\b(enable_thinking|reasoning_effort|thinking_budget|effort)\b", tpl or "")))
    return rep


LOW_SENTENCE = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                "without unnecessary elaboration.")
TASK = ("A warehouse ships 7 boxes per hour on weekdays and 4 per hour on weekends, 9 hours a day. How many boxes in "
        "two weeks? Answer with the number only.")


LONG_TASKS = [   # answers checked by brute force (2026-10-20)
    ("How many integers from 1 to 10000 are divisible by 3 or by 7, but not by both? Compute the  "
     "answer ten different ways to make sure it is correct. Answer with the number only, nothing else.", "3809"),
    ("In how many ways can a 3 x 8 rectangle be tiled completely with 2 x 1 dominoes (rotations of the whole board "
     "count as different tilings)? Compare three different approaches. Answer with the correct number only, nothing else.", "153"),
    ("Seven tasks: A 3 h, B 2 h, C 4 h, D 2 h, E 3 h, F 1 h, G 2 h. C needs A; D needs A and B; E needs C; F needs D; "
     "G needs E and F. Two workers, each does one task at a time, a task cannot be split. What is the minimum total "
     "time in hours to finish all tasks? Answer with the correct number only, nothing else.", "12"),
]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return nums[-1].replace(",", "") if nums else None


def post(url, key, body, timeout=1800):
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return json.loads(r.read())


def effort_probe(a, route, model, samples):
    """Small mode: one tiny question. Long mode (--effort-long): three harder questions with checked answers, a large
    token budget, requests in parallel per engine (--effort-concurrency) — tens of thousands of tokens per engine."""
    import concurrent.futures, statistics, time
    key = os.environ.get(f"INGRESS_KEY_{route.upper()}", "")
    url = f"{a.ingress.rstrip('/')}/{route}/v1/chat/completions"
    kw = lambda **k: {"chat_template_kwargs": k}
    variants = [("none (template default: xhigh)", {}, None),
                ("chat_template_kwargs reasoning_effort=low", kw(reasoning_effort="low"), None),
                # the template has no branch for medium: no effort instruction at all (2026-10-20)
                ("chat_template_kwargs reasoning_effort=medium", kw(reasoning_effort="medium"), None),
                ("prefix: the template's low sentence", {}, LOW_SENTENCE),
                ("chat_template_kwargs enable_thinking=false", kw(enable_thinking=False), None)]
    if not a.effort_long:
        variants.insert(1, ("top-level reasoning_effort=low", {"reasoning_effort": "low"}, None))
    tasks = LONG_TASKS if a.effort_long else [(TASK, "774")]   # 7·9·10 + 4·9·4 (was wrongly 1080 until 2026-10-20)

    def one(job):
        name, extra, prefix, (q, want) = job
        msgs = ([{"role": "system", "content": prefix}] if prefix else []) + [{"role": "user", "content": q}]
        t0 = time.monotonic()
        try:
            r = post(url, key, {"model": model, "messages": msgs, "max_tokens": a.effort_max_tokens, **extra})
        except (urllib.error.URLError, OSError, ValueError) as e:
            return name, {"error": f"{type(e).__name__} {getattr(e, 'code', '')}".strip()}
        dt = time.monotonic() - t0
        u = r.get("usage") or {}
        ch = (r.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if rt is None:
            rt = len(m.get("reasoning_content") or m.get("reasoning") or "") // 4
        ct = u.get("completion_tokens") or 0
        return name, {"reasoning": rt, "completion": ct, "s": dt, "ok": last_number(m.get("content")) == want,
                      "truncated": ch.get("finish_reason") == "length"}

    jobs = [(n, e, p, t) for n, e, p in variants for t in tasks for _ in range(samples)]
    res = {n: [] for n, _, _ in variants}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.effort_concurrency)) as ex:
        for name, r in ex.map(one, jobs):
            res[name].append(r)
    out = []
    for name, _, _ in variants:
        rs = [r for r in res[name] if "error" not in r]
        errs = [r["error"] for r in res[name] if "error" in r]
        med = lambda k: int(statistics.median([r[k] for r in rs])) if rs else None
        out.append({"variant": name, "reasoning_median": med("reasoning"), "completion_median": med("completion"),
                    "samples": [r["reasoning"] for r in rs], "correct": f"{sum(r['ok'] for r in rs)}/{len(rs)}",
                    "truncated": sum(r["truncated"] for r in rs), "tokens": sum(r["completion"] for r in rs),
                    "seconds_median": round(statistics.median([r["s"] for r in rs]), 1) if rs else None,
                    "error": errs[0] if errs else None})
    return out


def human(r):
    out = [f"== {r['route']} ({r['service']}) — model {r.get('model') or '?'}  [models {r['models_status']}, props {r['props_status']}]"]
    out.append("  SP1 sampling in force (no request-level sampling from Turnstone):")
    out.append(f"    engine flags:            {r.get('engine_flags') or 'none'}")
    if "engine_defaults" in r:
        out.append(f"    engine defaults (/props): {r['engine_defaults']}")
    if "model_generation_config" in r:
        out.append(f"    model generation_config: {r['model_generation_config']}  (vLLM applies it unless --generation-config vllm)")
    if "strata_config" in r:
        out.append(f"    Strata CONFIG:           {r['strata_config'] or 'no sampling keys'}")
    out.append(f"  EF1 chat template: {r.get('template_source') or 'not found'}; kwargs {r.get('template_kwargs_seen') or 'none'}")
    for l in r.get("template_effort_lines") or []:
        out.append(f"    {l}")
    for e in r.get("effort_probe") or []:
        out.append(f"  EF1 probe  {e['variant']:44s} reasoning ~{e['reasoning_median']}  completion {e['completion_median']}"
                   f"  correct {e['correct']}  cut {e['truncated']}  ~{e['seconds_median']} s  ({e['tokens']} tokens)"
                   + (f"  samples {e['samples']}" if e["samples"] and len(e["samples"]) <= 6 else "")
                   + (f"  ERROR {e['error']}" if e["error"] else ""))
    if r.get("template_quoted"):
        out.append("  quoted wording near effort/thinking (verbatim candidates for the plans):")
        out += [f"    «{q}»" for q in r["template_quoted"][:12]]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ingress", required=True)
    ap.add_argument("--resolve", help="HOST:PORT:IP — connect to IP for HOST:PORT (TLS still checks HOST)")
    ap.add_argument("--engine", action="append", default=[], help="ROUTE=SERVICE (repeatable)")
    ap.add_argument("--models-dir", default="/srv/ai/models")
    ap.add_argument("--infra", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--project", default="turnstone")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--effort-probe", action="append", default=[], help="ROUTE: measure which effort channel it honours")
    ap.add_argument("--effort-samples", type=int, default=3)
    ap.add_argument("--effort-max-tokens", type=int, default=4096)
    ap.add_argument("--effort-long", action="store_true", help="three harder checked questions, big budget")
    ap.add_argument("--effort-concurrency", type=int, default=1, help="parallel requests per engine")
    a = ap.parse_args(argv)
    if a.resolve:
        host, port, ip = a.resolve.rsplit(":", 2)
        orig = socket.getaddrinfo
        socket.getaddrinfo = lambda h, p, *r, **k: orig(ip if (h == host and str(p) == port) else h, p, *r, **k)
    reps = [inspect(a, *e.split("=", 1)) for e in a.engine]
    import concurrent.futures
    todo = [r for r in reps if r["route"] in a.effort_probe and r.get("model")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(todo))) as ex:   # engines sit on other GPUs
        for r, res in zip(todo, ex.map(lambda r: effort_probe(a, r["route"], r["model"], a.effort_samples), todo)):
            r["effort_probe"] = res
    print(json.dumps(reps, indent=1) if a.json else "\n\n".join(human(r) for r in reps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
