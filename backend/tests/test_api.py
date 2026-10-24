from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)

BOX = {"cpus": [{"cpu_id": "i9-10900x", "ram_gb": 128, "ram_mts": 3200, "ram_channels": 4}],
       "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16},
                   {"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16},
                   {"id": "gpu-3", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 8},
                   {"id": "gpu-4", "kind": "gpu", "model_id": "rtx-3060-12g", "link_gen": 3, "link_width": 4},
                   {"id": "ssd-1", "kind": "ssd", "model_id": "nvme-gen3", "link_gen": 3, "link_width": 4, "capacity_tb": 2}]}
ENGINES = [
    {"id": "maker", "name": "Strata", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next",
     "gpus": ["gpu-1", "gpu-2"], "cpu_cores": 9, "args": {"batch": 4}},
    {"id": "sub", "name": "27B", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-3"],
     "args": {"tensor_parallel": "1", "max_num_seqs": 4}},
    {"id": "guard", "engine": "llama.cpp", "version": "b11552", "model_id": "ministral-3-3b", "gpus": ["gpu-4"],
     "args": {"parallel": 1, "quantization": "q8_0"}},
    {"id": "coord", "engine": "hosted-api", "version": "provider", "model_id": "deepseek-v4.1-flash"},
]


def test_catalog():
    r = c.get("/api/catalog").json()
    assert len(r["cpus"]) > 30 and len(r["gpus"]) > 30 and "strata" in r["engines"]
    assert all(2011 <= x["year"] <= 2026 for x in r["cpus"])


def test_validate_and_derive():
    r = c.post("/api/machine/validate", json=BOX).json()
    assert r["ok"] and r["cpus"]["0"]["lanes_used"] == 48
    d = c.post("/api/engines/derive", json={"machine": BOX, "engines": ENGINES}).json()
    assert not d["engines"]["maker"]["errors"] and d["engines"]["maker"]["decode_total"]["1"] > 80


def test_simulate_end_to_end():
    body = {"machine": BOX, "engines": ENGINES,
            "roles": {"coordinator": "coord", "maker": "maker", "sub": "sub", "guard": "guard"},
            "concurrency": [1, 2], "scenarios": 3}
    r = c.post("/api/simulate", json=body).json()
    assert not r["errors"], r["errors"]
    assert len(r["results"]) == 2
    first = r["results"][0]
    assert first["tasks_per_h"]["p50"] > 0 and "maker" in first["engines"] and first["bottlenecks"]
    assert "cost_usd" in first["engines"]["coord"]


def test_simulate_reports_bad_config():
    body = {"machine": BOX, "engines": ENGINES, "roles": {"maker": "nope"}, "concurrency": [1], "scenarios": 1}
    r = c.post("/api/simulate", json=body).json()
    assert r["errors"]
