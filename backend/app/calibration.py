"""Reproduce published measurements with the performance model.

Two groups: CAL points were used (directly or indirectly) to pick the model's constants; VAL points
were held out. Sources: the owner's bench passes 1-4 (box), Strata community reports under
bench/results/ in the Strata repository (hardware and settings as reported), and the llama.cpp
CUDA table posted with build numbers on r/LocalLLaMA (2024-11, FA on, Q4_0).
Run: PYTHONPATH=. python -m app.calibration
"""
from .machine import validate
from .perf import derive
from .schemas import CpuSlot, EngineSpec, Machine, PcieDevice


def machine(cpu, ram_gb, mts, ch, gpus, gen=4, widths=None, os="linux", power=None):
    widths = widths or [16] * len(gpus)
    power = power or [None] * len(gpus)
    return validate(Machine(os=os, cpus=[CpuSlot(cpu_id=cpu, ram_gb=ram_gb, ram_mts=mts, ram_channels=ch)],
                            devices=[PcieDevice(id=f"gpu-{i + 1}", kind="gpu", model_id=g, link_gen=gen, link_width=w, power_w=pw)
                                     for i, (g, w, pw) in enumerate(zip(gpus, widths, power))]))


def strata(info, n, cores, **args):
    return derive(EngineSpec(id="s", engine="strata", version="0.1.43.1", model_id="qwen3.8-flash-next",
                             gpus=[f"gpu-{i + 1}" for i in range(n)], cpu_cores=cores, args=args), info)


def llama(info, model, quant, **args):
    return derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id=model, gpus=["gpu-1"], cpu_cores=8,
                             args={"quantization": quant, "parallel": 1, "ctx_size": 2048, **args}), info)   # tg128 / pp512 benches


def vllm(info, model, gpus, **args):
    return derive(EngineSpec(id="v", engine="vllm", version="0.31.0", model_id=model, gpus=gpus,
                             args={"tensor_parallel": str(len(gpus)), "max_num_seqs": 4, **args}), info)


STRATA = "https://github.com/Niko1221/Strata/tree/main/bench/results/"
SOURCES = {
    "box": dict(name="<keysie>", url=None, note="Bench passes 1–4 on the owner's box: i9-10900X, 4-ch DDR4-3200, 3 x RTX 3090 "
                "(x16/x16/x8, power-limited to 250 W) + RTX 3060 (x4), Strata 0.1.43.1, vLLM 0.31.0, llama.cpp b11552, Linux."),
    "strata-3060": dict(name="Strata community report: RTX 3060 12 GB, Core i7-12700, 64 GB DDR4 (Windows)",
                        url=STRATA + "2026-10-09-community-rtx-3060-0141", note="Strata 0.1.41, four quants, 128→260K prompt sweep, MTP on/off."),
    "strata-4090": dict(name="Strata community report: RTX 4090, Ryzen 9 7950X, 46 GB (Linux)",
                        url=STRATA + "2026-10-03-community-rtx4090-iq3s-140k-code", note="IQ3_S at 143K context, 4K/32K/128K prompts, gen-only run."),
    "strata-5090": dict(name="Strata community report: RTX 5090, Ryzen 9 9950X3D, 96 GB (Windows 11)",
                        url=STRATA + "2026-10-02-community-rtx-5090-iq3s", note="IQ3_S at 262K, KV resident 65536, 1K–131K prompts."),
    "strata-titan": dict(name="Strata community report: 2 x TITAN RTX, Xeon E5-2696 v4, 125 GB (Linux)",
                         url=STRATA + "2026-10-01-community-2x-titan-rtx", note="IQ3_S, layer split, NVLink present but unused, 4K/32K/128K prompts."),
    "strata-2080ti": dict(name="Strata community report: RTX 2080 Ti 11 GB, 24-core Zen 2, 128 GB DDR4-2133 (Linux)",
                          url=STRATA + "2026-10-02-community-rtx2080ti-11gb", note="IQ3_S at 262K with KV streaming and a 3.25 GiB expert cache."),
    "strata-4000ada": dict(name="Strata community report: 2 x RTX 4000 Ada, EPYC 7313P, 125 GB (Linux)",
                           url=STRATA + "2026-10-02-community-rtx4000ada-layer-split", note="IQ3_S, 2- vs 3-card layer split, 16.6K prompt."),
    "llamacpp-table": dict(name="llama.cpp CUDA table with build numbers (r/LocalLLaMA, 2024-11)",
                           url="https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/", note="Llama 3.1 8B Q4_0, b4011, flash attention on: pp512 / tg128 for 3090, 4090, 3050, M2, EPYC."),
    "llamacpp-apple": dict(name="llama.cpp discussion #4167: Performance of llama.cpp on Apple Silicon M-series (ggerganov)",
                           url="https://github.com/ggml-org/llama.cpp/discussions/4167",
                           note="Llama 2 7B F16/Q8_0/Q4_0, pp512 / tg128, Metal, every M-series chip; pinned builds (8e672ef for M1–M4, "
                                "c1d0e7a with flash attention for the M5 family and the 2026 re-runs). The 2023 build is 15–30 % slower than today's."),
    "llamacpp-rocm": dict(name="llama.cpp discussion #15021: Performance of llama.cpp on AMD ROCm (HIP)",
                          url="https://github.com/ggml-org/llama.cpp/discussions/15021",
                          note="Llama 2 7B Q4_0 pp512 / tg128 scoreboard with commits and ROCm versions; Strix Halo (Ryzen AI Max+ 395) and "
                               "discrete RDNA cards; Qwen3.8-27B Q5_K_M on Strix Halo from a user matrix with scripts."),
    "llamacpp-spark": dict(name="llama.cpp discussion #16578: Performance of llama.cpp on NVIDIA DGX Spark (ggerganov)",
                           url="https://github.com/ggml-org/llama.cpp/discussions/16578",
                           note="gpt-oss-120b MXFP4 and others, pp2048 / tg32 at context depths 0–32K, build 6767; a second user's run of the "
                                "same model came out 20 % lower (42 vs 53 tok/s), which bounds the reproducibility."),
    "kunal-32b": dict(name="kunalganglani.com LLM benchmarks (marked 'measured')",
                      url="https://kunalganglani.com/llm-benchmarks", note="Qwen2.5-32B Q4_K_M on 3090 / 4090; used only as a held-out check."),
}

POINTS = []   # (group, label, modelled, measured, source key)


def add(group, label, modelled, measured, source="box"):
    POINTS.append((group, label, modelled, measured, source))


def run():
    # --- the owner's box: i9-10900X, 4-ch DDR4-3200, 3 x 3090 (x16/x16/x8), 3060 x4 ----------------
    box = machine("i9-10900x", 128, 3200, 4, ["rtx-3090", "rtx-3090", "rtx-3090", "rtx-3060-12g"], 3, [16, 16, 8, 4],
                  power=[250, 250, 250, None])      # the 3090s run power-limited to 250 W
    p = strata(box, 1, 9, batch=1)
    add("CAL", "box: Strata 1x3090 IQ3_S decode", p.decode_total[1], 95)
    add("CAL", "box: Strata 1x3090 prompt tok/s", p.prefill_rate, 1900)
    p2 = strata(box, 2, 9, batch=4)
    add("CAL", "box: Strata 2x3090 decode, 1 slot", p2.decode_total[1], 125)
    add("CAL", "box: Strata 2x3090 decode, 4 slots total", p2.decode_total[4], 175)
    add("CAL", "box: Strata 2x3090 prompt tok/s", p2.prefill_rate, 3300)
    add("VAL", "box: Strata 2x3090 115K prompt, seconds", 115000 / p2.prefill_rate + 0.7, 35)
    p3 = strata(box, 3, 9, batch=1)
    add("VAL", "box: Strata 3x3090 prompt tok/s (+25 %)", p3.prefill_rate, 4000)
    v = vllm(box, "qwen3.8-27b", ["gpu-3"])
    add("CAL", "box: vLLM 27B AWQ 3090 decode", v.speed_per_request[1], 39)
    add("CAL", "box: vLLM 27B AWQ 3090 prompt tok/s", v.prefill_rate, 1050)
    add("VAL", "box: vLLM 27B AWQ KV room (K tokens)", v.memory["kv_room_tokens"] / 1000, 21)
    v2 = vllm(box, "qwen3.8-27b", ["gpu-1", "gpu-2"], max_num_seqs=8)
    add("VAL", "box: vLLM 27B TP2 over PCIe, decode each", v2.speed_per_request[1], 50)
    add("VAL", "box: vLLM 27B TP2 prompt tok/s", v2.prefill_rate, 1400)
    l = derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="qwen3.8-27b", gpus=["gpu-3"], cpu_cores=4,
                          args={"quantization": "iq3_s", "parallel": 3, "ctx_size": 32768}), box)
    add("VAL", "box: llama.cpp 27B IQ3_S, 3 slots, each", l.speed_per_request[3], 29)
    g = derive(EngineSpec(id="g", engine="llama.cpp", version="b11552", model_id="ministral-3-3b", gpus=["gpu-4"], cpu_cores=1,
                          args={"quantization": "q8_0", "parallel": 1, "ctx_size": 32768}), box)
    add("CAL", "box: guard 3B prompt tok/s on 3060", g.prefill_rate, 5000)
    def guard_s(n):
        return n / g.prefill_rate + g.prefill_quad * n * n + 120 / g.speed_per_request[1]
    add("VAL", "box: guard 8K output on 3060, seconds", guard_s(8000), 3.38)
    add("VAL", "box: guard 28K output on 3060, seconds", guard_s(28000), 17.1)
    r = derive(EngineSpec(id="r", engine="llama.cpp", version="b11552", model_id="bge-reranker-v2-m3", gpus=[], cpu_cores=4,
                          args={"n_gpu_layers": "0"}), box)
    add("CAL", "box: reranker on CPU, s per 40-pair search", 40 / r.decode_total[1], 23)

    # --- Strata community reports --------------------------------------------------------------------
    m = machine("i7-12700-ddr4", 64, 3200, 2, ["rtx-3060-12g"], 4, os="windows")   # the report ran on Windows
    add("CAL", "3060 + i7-12700 2ch DDR4: IQ3_S decode (MTP on)", strata(m, 1, 12, batch=1).decode_total[1], 18.8, "strata-3060")
    add("VAL", "3060 + i7-12700: IQ3_S decode, MTP off", strata(m, 1, 12, batch=1, mtp=False).decode_total[1], 17.0, "strata-3060")
    add("VAL", "3060 + i7-12700: Q2_0 decode", strata(m, 1, 12, batch=1, quantization="q2_0").decode_total[1], 43, "strata-3060")
    add("VAL", "3060 + i7-12700: Q2_0 prompt tok/s (4K-32K)", strata(m, 1, 12, batch=1, quantization="q2_0").prefill_rate, 930, "strata-3060")
    m = machine("ryzen9-7950x", 48, 5200, 2, ["rtx-4090"], 4)
    add("CAL", "4090 + 7950X: IQ3_S decode (gen-only)", strata(m, 1, 16, batch=1).decode_total[1], 88.7, "strata-4090")
    add("VAL", "4090 + 7950X: IQ3_S prompt tok/s (32K)", strata(m, 1, 16, batch=1).prefill_rate, 2948, "strata-4090")
    m = machine("ryzen9-9950x3d", 96, 5600, 2, ["rtx-5090"], 5, os="windows")        # Windows 11, WDDM
    add("VAL", "5090 + 9950X3D: IQ3_S decode", strata(m, 1, 16, batch=1, kv_resident=65536, context=262144).decode_total[1], 142, "strata-5090")
    add("VAL", "5090 + 9950X3D: IQ3_S prompt tok/s (32K)", strata(m, 1, 16, batch=1, kv_resident=65536, context=262144).prefill_rate, 4168, "strata-5090")
    m = machine("xeon-e5-2696v4", 125, 2400, 4, ["titan-rtx", "titan-rtx"], 3)
    add("VAL", "2x TITAN RTX + E5-2696 v4: IQ3_S decode", strata(m, 2, 22, batch=1).decode_total[1], 68, "strata-titan")
    add("VAL", "2x TITAN RTX: IQ3_S prompt tok/s (32K)", strata(m, 2, 22, batch=1).prefill_rate, 1612, "strata-titan")
    m = machine("tr-3960x", 128, 2133, 4, ["rtx-2080ti"], 3)
    add("CAL", "2080 Ti 11GB + TR 3960X DDR4-2133: IQ3_S decode", strata(m, 1, 24, batch=1, kv_resident=32768, context=262144).decode_total[1], 39.6, "strata-2080ti")
    m = machine("epyc-7313p", 125, 3200, 8, ["rtx-4000-ada", "rtx-4000-ada"], 4)
    add("VAL", "2x RTX 4000 Ada + EPYC 7313P: IQ3_S decode", strata(m, 2, 16, batch=1).decode_total[1], 70, "strata-4000ada")
    add("VAL", "2x RTX 4000 Ada: 16.6K prompt tok/s", strata(m, 2, 16, batch=1).prefill_rate, 2117, "strata-4000ada")

    # --- unified-memory prebuilts: Apple Silicon (llama.cpp #4167) ----------------------------------------
    def soc(cpu, gpu, ram, mts=8533):
        return validate(Machine(cpus=[CpuSlot(cpu_id=cpu, ram_gb=ram, ram_mts=mts, ram_channels=8)],
                                devices=[PcieDevice(id="gpu-1", kind="gpu", model_id=gpu, link_gen=4, link_width=16)]))

    def mac(info, quant, ctx=512):          # tg128 / pp512 benches run at a short context
        return derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="llama-2-7b", gpus=["gpu-1"], cpu_cores=8,
                                 args={"quantization": quant, "parallel": 1, "ctx_size": ctx}), info)
    m2u = soc("apple-m2-ultra", "m2-ultra-gpu", 192, 6400)
    add("CAL", "M2 Ultra 76c (2026 build): Llama 2 7B Q4_0 tg128", mac(m2u, "q4_0").speed_per_request[1], 125.2, "llamacpp-apple")
    add("CAL", "M2 Ultra 76c (2026 build): Llama 2 7B F16 tg128", mac(m2u, "fp16").speed_per_request[1], 49.6, "llamacpp-apple")
    add("VAL", "M2 Ultra 76c (2026 build): Llama 2 7B F16 pp512", mac(m2u, "fp16").prefill_rate, 1628, "llamacpp-apple")
    add("VAL", "M2 Ultra 76c (2026 build): Llama 2 7B Q4_0 pp512", mac(m2u, "q4_0").prefill_rate, 1489, "llamacpp-apple")
    m1u = soc("apple-m1-ultra", "m1-ultra-gpu", 128, 6400)
    add("VAL", "M1 Ultra 64c (2026 build): Llama 2 7B Q4_0 tg128", mac(m1u, "q4_0").speed_per_request[1], 113.8, "llamacpp-apple")
    add("VAL", "M1 Ultra 64c (2026 build): Llama 2 7B F16 tg128", mac(m1u, "fp16").speed_per_request[1], 47.8, "llamacpp-apple")
    m5m = soc("apple-m5-max", "m5-max-gpu", 128)
    add("CAL", "M5 Max 40c (tensor units): Llama 2 7B Q4_0 tg128", mac(m5m, "q4_0").speed_per_request[1], 119.9, "llamacpp-apple")
    add("VAL", "M5 Max 40c: Llama 2 7B F16 pp512", mac(m5m, "fp16").prefill_rate, 3158, "llamacpp-apple")
    add("VAL", "M5 Max 40c: Llama 2 7B Q8_0 tg128", mac(m5m, "q8_0").speed_per_request[1], 72.4, "llamacpp-apple")
    m5u = soc("apple-m5-ultra", "m5-ultra-gpu", 256)
    add("VAL", "M5 Ultra 80c: Llama 2 7B Q4_0 tg128", mac(m5u, "q4_0").speed_per_request[1], 179.1, "llamacpp-apple")
    add("VAL", "M5 Ultra 80c: Llama 2 7B F16 pp512", mac(m5u, "fp16").prefill_rate, 5207, "llamacpp-apple")
    m4m = soc("apple-m4-max", "m4-max-gpu", 128)
    add("VAL", "M4 Max 40c (2023 build): Llama 2 7B Q4_0 tg128", mac(m4m, "q4_0").speed_per_request[1], 83.1, "llamacpp-apple")
    add("VAL", "M4 Max 40c (2023 build): Llama 2 7B F16 pp512", mac(m4m, "fp16").prefill_rate, 923, "llamacpp-apple")
    m3u = soc("apple-m3-ultra", "m3-ultra-gpu", 512)
    add("VAL", "M3 Ultra 80c (2023 build): Llama 2 7B Q4_0 tg128", mac(m3u, "q4_0").speed_per_request[1], 92.1, "llamacpp-apple")

    # --- Strix Halo and discrete RDNA (llama.cpp #15021) ------------------------------------------------
    sh = soc("amd-ai-max-395", "strix-halo-gpu", 128, 8000)
    add("CAL", "Ryzen AI Max+ 395 (Strix Halo, ROCm 7.14): Llama 2 7B Q4_0 tg128", mac(sh, "q4_0").speed_per_request[1], 49.1, "llamacpp-rocm")
    add("VAL", "Strix Halo: Llama 2 7B Q4_0 pp512", mac(sh, "q4_0").prefill_rate, 1144, "llamacpp-rocm")
    q27 = derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="qwen3.8-27b", gpus=["gpu-1"], cpu_cores=8,
                            args={"quantization": "q5_k_m", "parallel": 1, "ctx_size": 4096, "mtp": False}), sh)
    add("VAL", "Strix Halo: Qwen3.8-27B Q5_K_M decode, MTP off", q27.speed_per_request[1], 10.6, "llamacpp-rocm")
    q27m = derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="qwen3.8-27b", gpus=["gpu-1"], cpu_cores=8,
                             args={"quantization": "q5_k_m", "parallel": 1, "ctx_size": 4096, "mtp": True}), sh)
    add("VAL", "Strix Halo: Qwen3.8-27B Q5_K_M decode, MTP on (Vulkan)", q27m.speed_per_request[1], 17.0, "llamacpp-rocm")
    xtx = machine("ryzen9-7950x", 64, 5200, 2, ["rx-7900xtx"], 4)
    add("VAL", "RX 7900 XTX (ROCm): Llama 2 7B Q4_0 tg128", mac(xtx, "q4_0").speed_per_request[1], 125, "llamacpp-rocm")
    add("VAL", "RX 7900 XTX (ROCm): Llama 2 7B Q4_0 pp512", mac(xtx, "q4_0").prefill_rate, 3550, "llamacpp-rocm")

    # --- DGX Spark (llama.cpp #16578) ------------------------------------------------------------------
    sp = soc("gb10", "gb10-gpu", 128)
    oss = derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="gpt-oss-120b", gpus=["gpu-1"], cpu_cores=16,
                            args={"quantization": "mxfp4", "parallel": 1, "ctx_size": 4096}), sp)
    add("CAL", "DGX Spark: gpt-oss-120b MXFP4 tg32", oss.speed_per_request[1], 52.9, "llamacpp-spark")
    add("VAL", "DGX Spark: gpt-oss-120b MXFP4 pp2048", oss.prefill_rate, 1689, "llamacpp-spark")

    # --- llama.cpp dense models, public tables -------------------------------------------------------
    m = machine("ryzen9-7950x", 64, 5200, 2, ["rtx-3090"], 4)
    add("VAL", "llama.cpp Llama-3.1-8B Q4_0 on 3090: tg128", llama(m, "llama-3.1-8b", "q4_0").speed_per_request[1], 167, "llamacpp-table")
    add("VAL", "llama.cpp Llama-3.1-8B Q4_0 on 3090: pp512", llama(m, "llama-3.1-8b", "q4_0").prefill_rate, 6073, "llamacpp-table")
    add("VAL", "llama.cpp Qwen2.5-32B Q4_K_M on 3090: tg", llama(m, "qwen2.5-32b", "q4_k_m").speed_per_request[1], 28, "kunal-32b")
    m = machine("ryzen9-7950x", 64, 5200, 2, ["rtx-4090"], 4)
    add("VAL", "llama.cpp Llama-3.1-8B Q4_0 on 4090: tg128", llama(m, "llama-3.1-8b", "q4_0").speed_per_request[1], 188, "llamacpp-table")
    add("VAL", "llama.cpp Llama-3.1-8B Q4_0 on 4090: pp512", llama(m, "llama-3.1-8b", "q4_0").prefill_rate, 13944, "llamacpp-table")
    add("VAL", "llama.cpp Qwen2.5-32B Q4_K_M on 4090: tg", llama(m, "qwen2.5-32b", "q4_k_m").speed_per_request[1], 42, "kunal-32b")


CLAIMS = [
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 4090, 7B Q4_K_M: 135 tok/s",
         claimed=135, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-4090"), model="llama-3.1-8b", quant="q4_k_m", verdict="consistent"),
    dict(source="kunalganglani.com", url="https://kunalganglani.com/llm-benchmarks", claim="RTX 4090, Llama 3.1 8B Q4_K_M: 145 tok/s",
         claimed=145, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-4090"), model="llama-3.1-8b", quant="q4_k_m", verdict="consistent"),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 3090, 7B Q4_K_M: 95 tok/s",
         claimed=95, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-3090"), model="llama-3.1-8b", quant="q4_k_m",
         verdict="too low: the 3090 has 93 % of the 4090's bandwidth, so a bandwidth-bound decode should land near 120–165 (the trusted table measured 167 for Q4_0). Both pages show the 4090 1.5× ahead, which no bandwidth figure supports."),
    dict(source="kunalganglani.com", url="https://kunalganglani.com/llm-benchmarks", claim="RTX 3090, Llama 3.1 8B Q4_K_M: 92 tok/s",
         claimed=92, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-3090"), model="llama-3.1-8b", quant="q4_k_m", verdict="too low, same argument as above."),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 4060 Ti 16 GB, 7B Q4_K_M: 55 tok/s",
         claimed=55, machine=("ryzen9-5950x", 64, 3200, 2, "rtx-4060ti-16g"), model="llama-3.1-8b", quant="q4_k_m",
         verdict="doubtful: 288 GB/s over 4.9 GB of weights allows 59 tok/s at 100 % of the bus; 55 would be 93 %. Real kernels reach 65–80 %. Plausible only with speculative decoding."),
    dict(source="kunalganglani.com", url="https://kunalganglani.com/llm-benchmarks", claim="RTX 4060 Ti 16 GB, Llama 3.1 8B Q4_K_M: 50 tok/s",
         claimed=50, machine=("ryzen9-5950x", 64, 3200, 2, "rtx-4060ti-16g"), model="llama-3.1-8b", quant="q4_k_m", verdict="doubtful: 85 % of the bus; see above."),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 3060 12 GB, 7B Q4_K_M: 45 tok/s",
         claimed=45, machine=("ryzen9-5950x", 64, 3200, 2, "rtx-3060-12g"), model="llama-3.1-8b", quant="q4_k_m", verdict="plausible but low (60 % of the bus): fits Windows and K-quants on an older build."),
    dict(source="kunalganglani.com", url="https://kunalganglani.com/llm-benchmarks", claim="RTX 3060 12 GB, Llama 3.1 8B Q4_K_M: 38 tok/s",
         claimed=38, machine=("ryzen9-5950x", 64, 3200, 2, "rtx-3060-12g"), model="llama-3.1-8b", quant="q4_k_m", verdict="plausible but low (50 % of the bus)."),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 3060 12 GB, 13B Q4_K_M: 22 tok/s",
         claimed=22, machine=("ryzen9-5950x", 64, 3200, 2, "rtx-3060-12g"), model="phi-4-14b", quant="q4_k_m", verdict="plausible: 8.5 GB of weights plus KV barely fit 12 GB, so a little offload drags it toward 22."),
    dict(source="willitrunai.com (estimator)", url="https://willitrunai.com/can-run/llama-3.1-8b-on-rtx-4090-24gb", claim="RTX 4090, Llama 3.1 8B Q4_K_M: 'expect ~112 tok/s'",
         claimed=112, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-4090"), model="llama-3.1-8b", quant="q4_k_m", verdict="low but within noise; an estimate, not a measurement."),
    dict(source="kunalganglani.com", url="https://kunalganglani.com/llm-benchmarks", claim="RTX 4090, Qwen2.5 14B Q4_K_M: 92 tok/s",
         claimed=92, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-4090"), model="phi-4-14b", quant="q4_k_m", verdict="consistent (Phi-4 14B stands in for the 14B)."),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 4090, Llama 70B 'Q2': 18 tok/s",
         claimed=18, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-4090"), model="llama-3.3-70b", quant="q2_k", layers="75%",
         verdict="not with Q2_K: ~25 GB for 70B does not fit 24 GB; 2–4 GB must stream from host RAM and on 2-channel DDR5 that alone caps you near 10 tok/s. Reachable only with an IQ2_XXS-class quant (~19 GB) fully in VRAM. The table's 'Q2' hides which quant it was."),
    dict(source="mustafa.net", url="https://mustafa.net/llm-tokens-per-second-benchmarks/", claim="RTX 3090, Llama 70B 'Q2': 10 tok/s",
         claimed=10, machine=("ryzen9-7950x", 64, 5200, 2, "rtx-3090"), model="llama-3.3-70b", quant="q2_k", layers="75%",
         verdict="same caveat: fits an IQ2 quant resident in VRAM, not Q2_K with offload."),
]


def check_claims():
    out = []
    for c in CLAIMS:
        cpu, ram, mts, ch, gpu = c["machine"]
        m = machine(cpu, ram, mts, ch, [gpu], 4)
        kw = {"n_gpu_layers": c["layers"], "ctx_size": 4096} if c.get("layers") else {}
        p = llama(m, c["model"], c["quant"], **kw)
        v = p.speed_per_request.get(1, float("nan")) if not p.errors else float("nan")
        out.append({k: c[k] for k in ("source", "url", "claim", "claimed", "verdict")} | {"model": round(v, 1)})
    return out


def data_points():
    POINTS.clear()
    run()
    return [dict(group=g, label=l, model=round(m, 1), measured=t, error=round((m - t) / t, 3), source=src) for g, l, m, t, src in POINTS]


def report():
    run()
    out = []
    for grp in ("CAL", "VAL"):
        rows = [p for p in POINTS if p[0] == grp]
        errs = [abs(m - t) / t for _, _, m, t, _ in rows]
        out.append(f"\n{grp} ({len(rows)} points): median |error| {sorted(errs)[len(errs) // 2]:.0%}, "
                   f"worst {max(errs):.0%}, within 30 %: {sum(e <= .3 for e in errs)}/{len(errs)}")
        out.append(f"{'point':58s} {'model':>9s} {'measured':>9s} {'error':>7s}")
        for _, label, m, t, _ in rows:
            out.append(f"{label:58s} {m:9.1f} {t:9.1f} {(m - t) / t:+7.0%}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
