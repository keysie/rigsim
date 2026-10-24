"""inspect-engines.py against a fake ingress (/v1/models, /props), a fake `docker inspect` and a fake HF cache."""
import importlib.util, json, os, pathlib, stat, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location("ie", pathlib.Path(__file__).resolve().parents[1] / "inspect-engines.py")
ie = importlib.util.module_from_spec(spec); spec.loader.exec_module(ie)
TPL = ("{%- if reasoning_effort is defined and reasoning_effort == 'low' %}\n"
       "{{- '<|im_start|>system\\nThink briefly: keep the reasoning short and go to the answer.<|im_end|>' }}\n"
       "{%- endif %}\n{%- if enable_thinking is false %}<think>\\n\\n</think>{%- endif %}\nplain line\n")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        low = (b.get("chat_template_kwargs") or {}).get("reasoning_effort") == "low" or b["messages"][0]["role"] == "system"
        off = (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        n = 0 if off else 40 if low else 400          # this fake honours kwargs and the prefix, not the top-level field
        body = {"choices": [{"message": {"content": "so the answer is\n774", "reasoning_content": "x" * n}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": n // 4 + 3}}
        out = json.dumps(body).encode(); self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer k":
            self.send_response(401); self.end_headers(); return
        if self.path.endswith("/v1/models"):
            body = {"data": [{"id": "m-" + self.path.split("/")[1]}]}
        elif self.path == "/guard/props":
            body = {"chat_template": TPL, "default_generation_settings": {"params": {"temperature": 0.8, "top_k": 40, "n_predict": -1}}}
        else:
            self.send_response(404); self.end_headers(); return
        b = json.dumps(body).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


class Inspect(unittest.TestCase):
    def test_report(self):
        d = pathlib.Path(tempfile.mkdtemp())
        snap = d / "models/hf/hub/models--org--sub/snapshots/abc"; snap.mkdir(parents=True)
        (snap / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "eos_token_id": 1}))
        (snap / "chat_template.jinja").write_text(TPL)
        (d / "infra/strata").mkdir(parents=True)
        (d / "infra/strata/strata-maker.json").write_text(json.dumps({"model": "x", "sampling": {"temperature": 0.7}, "ctx": 32768}))
        fake = d / "bin/docker"; fake.parent.mkdir()
        fake.write_text("#!/bin/sh\ncase \"$2\" in\n"
                        " *vllm-sub*) echo '[{\"Config\":{\"Entrypoint\":[\"vllm\",\"serve\"],\"Cmd\":[\"org/sub\",\"--served-model-name=qwen\",\"--max-model-len=32768\"]}}]';;\n"
                        " *llama-guard*) echo '[{\"Config\":{\"Entrypoint\":[\"/app/llama-server\"],\"Cmd\":[\"-m\",\"/models/g.gguf\",\"--temp\",\"0.15\",\"--jinja\"]}}]';;\n"
                        " *) echo '[]';;\nesac\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        os.environ["PATH"] = f"{fake.parent}:{os.environ['PATH']}"
        for r in ("sub", "guard", "strata"):
            os.environ[f"INGRESS_KEY_{r.upper()}"] = "k"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        import io, contextlib
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                ie.main(["--ingress", f"http://box.test:{port}", "--resolve", f"box.test:{port}:127.0.0.1", "--json",
                         "--engine", "sub=vllm-sub", "--engine", "guard=llama-guard", "--engine", "strata=strata",
                         "--models-dir", str(d / "models"), "--infra", str(d / "infra"), "--effort-probe", "sub", "--effort-samples", "2"])
        finally:
            srv.shutdown(); srv.server_close()
        reps = {r["route"]: r for r in json.loads(out.getvalue())}
        sub, guard, strata = reps["sub"], reps["guard"], reps["strata"]
        self.assertEqual(sub["model"], "m-sub"); self.assertEqual(sub["model_ref"], "org/sub")
        self.assertEqual(sub["model_generation_config"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertEqual(sub["template_kwargs_seen"], ["enable_thinking", "reasoning_effort"])
        self.assertTrue(any("Think briefly" in q for q in sub["template_quoted"]))
        self.assertEqual(guard["engine_flags"], {"--temp": "0.15", "--jinja": True})
        self.assertEqual(guard["engine_defaults"], {"temperature": 0.8, "top_k": 40})
        self.assertEqual(guard["template_source"], "engine /props")
        self.assertEqual(strata["strata_config"], {"sampling.temperature": 0.7})
        self.assertIn("template_effort_lines", strata)
        ep = {e["variant"]: e["reasoning_median"] for e in sub["effort_probe"]}
        self.assertEqual(ep, {"none (template default: xhigh)": 100, "top-level reasoning_effort=low": 100, "chat_template_kwargs reasoning_effort=low": 10,
                              "chat_template_kwargs reasoning_effort=medium": 100,
                              "prefix: the template's low sentence": 10, "chat_template_kwargs enable_thinking=false": 0})
        self.assertTrue(all(e["correct"] == "2/2" for e in sub["effort_probe"]))
        self.assertEqual(ie.last_number("12 h? no: the minimum is\n12."), "12"); self.assertEqual(ie.last_number("3,809"), "3809")
        self.assertNotIn("effort_probe", guard)
        txt = ie.human(sub); self.assertIn("SP1", txt); self.assertIn("EF1 probe", txt)


if __name__ == "__main__":
    unittest.main()
