"""Mock OpenAI server for the bench tests: streams reasoning_content, then content. Like Strata (pass 3): earlier
assistant turns are rendered with their reasoning_content in a think block, and cached_tokens is the common prefix
with what was generated before (so sending the thinking back extends the reuse); GET /metrics gives a live slot view,
the request history and conversation-cache counters.
Reasoning length: 60 chunks by default, 20 with chat_template_kwargs effort 'low', 25 if the first user message starts
with 'THINK LESS'. finish_reason 'length' if max_tokens is below reasoning + answer. Answers are canned by keyword."""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWERS = [("kebab-case", "myreport-final.pdf\nsetup-notes.md\nq3-budget-v2.xlsx"),
           ("host, port, user", '{"host": "db07.internal", "port": 5433, "user": "svc_report"}'),
           ("contain ERROR", "3"), ("TOTAL", "TOTAL: 100"), ("CAUSE", "CAUSE: run.sh is not executable"),
           ("tool_output_", '{"risk_level": "none", "flags": [], "reasoning": "data", "confidence": 0.9}')]

HELD, HIST, LOCK = [], [], threading.Lock()


def render(msgs):
    out = []
    for m in msgs:
        c = m.get("content") if isinstance(m.get("content"), str) else ""
        if m["role"] == "assistant":
            out.append("<assistant><think>" + (m.get("reasoning_content") or "") + "</think>" + c)
        else:
            out.append(f"<{m['role']}>" + c)
    return "".join(out)


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, d, code=200):
        b = json.dumps(d).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/metrics"):
            with LOCK:
                hist = list(HIST)
            return self._send({"live": {"state": "generating", "tok_s": 50.0, "running": 1, "outside_slots": 0,
                                        "waiting": 0, "slots": [{"slot": 0, "state": "decoding", "prompt_tokens": 10,
                                                                 "generated": 3, "held_tokens": 0}]},
                               "requests": hist[::-1], "conversation_cache": {"parked": 0, "restores": 0}})
        self._send({"data": [{"id": "mock-reasoner"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.rstrip("/").endswith("/rerank"):           # pass 4: llama.cpp's rerank endpoint
            docs = body.get("documents") or []
            time.sleep(0.002 * len(docs))
            return self._send({"results": [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(len(docs))],
                               "usage": {"prompt_tokens": sum(len(d) // 4 for d in docs)}})
        msgs = body["messages"]
        first_user = next(m["content"] for m in msgs if m["role"] == "user")
        eff = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        n = 20 if eff == "low" else 25 if first_user.startswith("THINK LESS") else 60
        text = " ".join(m["content"] for m in msgs if isinstance(m.get("content"), str))
        ans = next((a for k, a in ANSWERS if k in text), "no idea")
        maxt = body.get("max_tokens", 10**6); finish = "stop"
        if maxt < n + 1:
            n, finish, ans = maxt, "length", ""
        P = render(msgs)
        prompt = len(P) // 4                   # the rendered prompt, in the same units as cached_tokens
        t_start = time.time()
        with LOCK:
            cached = max((lcp(P, h) for h in HELD), default=0) // 4
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def ev(d): self.wfile.write(f"data: {json.dumps(d)}\n\n".encode()); self.wfile.flush()
        for _ in range(n):
            ev({"choices": [{"delta": {"reasoning_content": "hm "}}]}); time.sleep(0.001)
        tc_mode = body.get("tool_choice")
        if tc_mode == "none" and body.get("tools"):          # pass 4 judge: the verdict
            ans = '{"verdict": "approve", "risk_level": "low", "confidence": 0.9, "reasoning": "ok"}'
        if body.get("tools") and (tc_mode == "required" or (tc_mode is None and len(msgs) % 2 == 1)):
            # agentloop with a tool lane: every other turn a tool call; the judge's forced turns: always
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"c{len(msgs)}", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}]}}]})
        elif ans:
            ev({"choices": [{"delta": {"content": ans}}]})
        ev({"choices": [{"delta": {}, "finish_reason": finish}]})
        ev({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": n + 1,
            "completion_tokens_details": {"reasoning_tokens": n}, "prompt_tokens_details": {"cached_tokens": cached}}})
        with LOCK:
            HELD.append(P + "<assistant><think>" + "hm " * n + "</think>" + (ans if finish == "stop" else ""))
            HIST.append({"time": t_start, "output_tokens": n + 1, "hit_rate": 0.9, "pcie_share": 0.01,
                         "reused": cached, "prompt_read": prompt - cached, "decode_tok_s": 50.0})
        self.wfile.write(b"data: [DONE]\n\n")

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
