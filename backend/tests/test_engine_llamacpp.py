"""Engine layer, first slice: llama.cpp CUDA on one GPU, dense models (docs/overhaul/engine.md §4, §6)."""
import statistics
import struct

from app.engine import gguf
from app.engine.llamacpp import cuda
from app.engine.model import from_config
from app.hw.execute import Op, execute
from app.hw.yardsticks import validate_llamabench as V


def test_model_sizes_match_what_llama_bench_reports():
    # llama-bench prints size and parameter count per model: Llama 2 7B Q4_0 3.56 GiB / 6.74 B, Llama 3.1 8B Q4_0
    # 4.33 GiB / 8.03 B
    for key, gib, params in (("llama-2-7b", 3.56, 6.74), ("llama-3.1-8b", 4.33, 8.03)):
        m = from_config(key, "q4_0")
        assert round(m.weight_bytes / 2**30, 2) == gib
        assert round(sum(t.n for t in m.tensors) / 1e9, 2) == params


def _gguf(path, tensors, meta):
    def s(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    for name, dims, t in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", t, 0)
    path.write_bytes(out)


def test_gguf_header_gives_exact_tensors(tmp_path):
    f = tmp_path / "m.gguf"
    _gguf(f, [("token_embd.weight", (64, 1000), 14), ("blk.0.attn_q.weight", (64, 64), 2)],
          {"general.architecture": "llama", "llama.embedding_length": 64, "llama.attention.head_count": 4,
           "llama.block_count": 1, "llama.feed_forward_length": 128, "llama.attention.head_count_kv": 2})
    m = gguf.model(f)
    assert (m.d_model, m.heads, m.kv_heads, m.head_dim, m.vocab) == (64, 4, 2, 16, 1000)
    assert m.tensor("token_embd.weight").type == "q6_k" and m.tensor("blk.0.attn_q.weight").type == "q4_0"
    assert m.tensor("blk.0.attn_q.weight").bytes == 64 * 64 * 18 / 32


def test_engine_states_work_only_and_switches_kernel_path():
    m = from_config("tinyllama-1.1b", "q4_0")
    ops = cuda.forward(m, 1, 0)
    assert all(isinstance(o, Op) for o in ops) and not any(hasattr(o, "seconds") for o in ops)
    q = next(o for o in ops if o.name == "blk.0.attn_q")
    assert q.factor_basis.startswith("mmvq") and q.op_class is None                 # memory-bound dot product
    big = next(o for o in cuda.forward(m, 512, 0) if o.name == "blk.0.attn_q")
    assert big.factor_basis.startswith("mmq") and big.op_class == "int8_tensor"     # int8 tensor-core tiles
    assert next(o for o in cuda.forward(m, 8, 0) if o.name == "blk.0.attn_q").factor_basis.startswith("mmvq")


def test_execute_adds_launch_cost_and_reports_the_bound():
    m = V.machine("rtx-3090")
    r = execute(m, "gpu0", [Op("tiny", "elementwise", bytes=1e3, kernels=3), Op("big", "matmul", bytes=1e9)], cuda.RUNTIME)
    tiny, big = r.ops
    assert tiny.bound == "launch" and big.bound == "memory"
    assert abs(big.memory_s - 1e9 / r.vram_bps) < 1e-12 and "unmeasured" in r.basis["launch"]


def test_validation_writing_close_prompt_reading_a_known_miss():
    res = V.compare()
    tg = [r["error"] for r in res if r["test"].startswith("tg")]
    pp = [r["error"] for r in res if r["test"].startswith("pp")]
    assert len(tg) >= 8 and abs(statistics.median(tg)) < 0.05 and max(abs(e) for e in tg) < 0.15
    # known miss (yardsticks.md): MMQ kernel factor unmeasured (1.0 x the cuBLAS-int8 default); the model reads
    # prompts about twice as fast as measured. Replace this bound when per-kernel factors are measured.
    assert len(pp) >= 8 and 0.4 < statistics.median(pp) < 1.6
