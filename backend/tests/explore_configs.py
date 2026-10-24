import json, time
from app.schemas import *
from app.machine import validate
from app.sim.run import simulate, trace
from app.perf import derive

def dev(i, model, cpu=0, gen=4, w=16, kind="gpu", cap=None):
    return PcieDevice(id=f"{kind}-{i}", kind=kind, model_id=model, cpu=cpu, link_gen=gen, link_width=w, capacity_tb=cap)
def run(label, m, engines, roles, conc=(1,2,4)):
    info = validate(m)
    r = simulate(SimRequest(machine=m, engines=engines, roles=roles, concurrency=list(conc), scenarios=6))
    print(f"\n== {label}")
    print("   machine:", "ok" if info["ok"] else info["errors"], "| warnings:", len(info["warnings"]), "| lanes:", {k:(c["lanes_used"],c["lanes_total"]) for k,c in info["cpus"].items()}, "| RAM GB/s:", {k:c["ram_bw_gbs"] for k,c in info["cpus"].items()})
    for e in engines:
        pf = derive(e, info); print(f"   {e.name:22s} {'ERR '+pf.errors[0][:90] if pf.errors else f'read {pf.prefill_rate:6.0f} tok/s, write {pf.decode_total.get(1,0):5.0f} alone, {max(pf.decode_total.values()) if pf.decode_total else 0:5.0f} max total, slots {pf.slots}'}")
    if r["errors"]: print("   SIM ERRORS:", r["errors"]); return
    for x in r["results"]:
        print(f"   {x['concurrency']} ws: {x['tasks_per_h']['p50']:.1f} tasks/h  gpus {dict((k, round(v,2)) for k,v in x['gpus'].items())}")
        print("      -", x["bottlenecks"][0] if x["bottlenecks"] else "nothing saturates")
    return r

# 1. Threadripper PRO 7995WX, 8ch DDR5-5200, 6 x RTX 4090, 4 NVMe gen5
m = Machine(cpus=[CpuSlot(cpu_id="tr-pro-7995wx", ram_gb=512, ram_mts=5200, ram_channels=8)],
            devices=[dev(i, "rtx-4090") for i in range(1,7)] + [dev(i, "nvme-gen5", kind="ssd", gen=5, w=4, cap=4) for i in range(1,5)])
eng = [EngineSpec(id="mk", name="Strata 3x4090", engine="strata", version="0.2.2", model_id="qwen3.8-flash-next", gpus=["gpu-1","gpu-2","gpu-3"], cpu_cores=32, args={"batch":6,"parking_mib":16384,"parking_slots":8}),
       EngineSpec(id="sub", name="27B TP2 4090", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-4","gpu-5"], args={"tensor_parallel":"2","max_num_seqs":8}),
       EngineSpec(id="jg", name="judge 32B 4090", engine="vllm", version="0.31.0", model_id="qwen3-32b", gpus=["gpu-6"], args={"tensor_parallel":"1","max_num_seqs":4}),
       EngineSpec(id="gd", name="guard 3B cpu", engine="llama.cpp", version="b11552", model_id="ministral-3-3b", gpus=[], cpu_cores=16, args={"n_gpu_layers":"0","parallel":2,"quantization":"q8_0","ctx_size":32768}),
       EngineSpec(id="rr", name="reranker gpu-6", engine="llama.cpp", version="b11552", model_id="bge-reranker-v2-m3", gpus=["gpu-6"], cpu_cores=1, args={"parallel":2}),
       EngineSpec(id="co", name="coordinator", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash")]
run("TR PRO 7995WX, 6 x 4090, 4 NVMe", m, eng, Roles(coordinator="co", maker="mk", sub="sub", guard="gd", judge="jg", reranker="rr"))

# 2. Dual EPYC 9654 with unequal RAM, 4 x A100 80G nvlinked pairs, 8 U.2
m2 = Machine(cpus=[CpuSlot(cpu_id="epyc-9654", ram_gb=768, ram_mts=4800, ram_channels=12), CpuSlot(cpu_id="epyc-9654", ram_gb=192, ram_mts=4800, ram_channels=6)],
             devices=[dev(1,"a100-80g",0,5), dev(2,"a100-80g",0,5), dev(3,"a100-80g",1,5), dev(4,"a100-80g",1,5)] + [dev(i,"nvme-u2-ent",i%2,4,4,"ssd",8) for i in range(1,9)],
             nvlinks=[("gpu-1","gpu-2"),("gpu-3","gpu-4")])
eng2 = [EngineSpec(id="mk", name="235B-A22B TP2 A100", engine="vllm", version="0.31.0", model_id="qwen3-235b-a22b", gpus=["gpu-1","gpu-2"], cpu=0, args={"tensor_parallel":"2","max_num_seqs":16,"quantization":"fp8"}),
        EngineSpec(id="sub", name="27B TP2 A100 cpu2", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-3","gpu-4"], cpu=1, args={"tensor_parallel":"2","max_num_seqs":16,"quantization":"fp8"}),
        EngineSpec(id="gd", name="guard on cpu2", engine="llama.cpp", version="b11552", model_id="ministral-3-3b", gpus=[], cpu=1, cpu_cores=24, args={"n_gpu_layers":"0","parallel":4,"quantization":"q8_0","ctx_size":32768}),
        EngineSpec(id="co", name="coordinator", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash")]
run("Dual EPYC 9654 (768+192 GB), 4 x A100 80G, 8 U.2", m2, eng2, Roles(coordinator="co", maker="mk", sub="sub", guard="gd", judge="sub"))

# 3. Old HEDT: i7-6950X (40 lanes, DDR4-2400) with 2 x 1080 Ti + 2080 Ti: should mostly not fit
m3 = Machine(cpus=[CpuSlot(cpu_id="i7-6950x", ram_gb=64, ram_mts=2400, ram_channels=4)],
             devices=[dev(1,"gtx-1080ti",0,3), dev(2,"gtx-1080ti",0,3), dev(3,"rtx-2080ti",0,3,8), dev(1,"nvme-gen3",0,3,4,"ssd",1)])
eng3 = [EngineSpec(id="mk", name="Strata on 1080Ti+1080Ti", engine="strata", version="0.1.43.1", model_id="qwen3.8-flash-next", gpus=["gpu-1","gpu-2"], cpu_cores=9, args={"batch":1}),
        EngineSpec(id="sub", name="27B llama 2080Ti", engine="llama.cpp", version="b11552", model_id="qwen3.8-27b", gpus=["gpu-3"], cpu_cores=4, args={"parallel":1,"n_gpu_layers":"50%","quantization":"iq3_xxs","ctx_size":16384}),
        EngineSpec(id="co", name="coordinator", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash")]
run("i7-6950X, 2 x 1080 Ti + 2080 Ti", m3, eng3, Roles(coordinator="co", maker="mk", sub="sub"))

# 4. Consumer 9950X with 2 x 5090 (lanes 24: x8+x8+x4 ssd), everything on vLLM, mixed
m4 = Machine(cpus=[CpuSlot(cpu_id="ryzen9-9950x", ram_gb=96, ram_mts=5600, ram_channels=2)],
             devices=[dev(1,"rtx-5090",0,5,8), dev(2,"rtx-5090",0,5,8), dev(1,"nvme-gen5",0,5,4,"ssd",2), dev(2,"nvme-gen4",0,4,4,"ssd",2)])
eng4 = [EngineSpec(id="mk", name="gpt-oss-120b vLLM 5090", engine="vllm", version="0.31.0", model_id="gpt-oss-120b", gpus=["gpu-1"], args={"tensor_parallel":"1","max_num_seqs":8,"quantization":"awq4"}),
        EngineSpec(id="sub", name="27B vLLM 5090", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-2"], args={"tensor_parallel":"1","max_num_seqs":8}),
        EngineSpec(id="co", name="coordinator", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash")]
run("9950X, 2 x 5090 at x8 (lane-limited), 2 NVMe", m4, eng4, Roles(coordinator="co", maker="mk", sub="sub", judge="sub"))

# 5. Same box but Strata with measured benchmarks entered by the user
m5 = Machine(cpus=[CpuSlot(cpu_id="i9-10900x", ram_gb=128, ram_mts=3200, ram_channels=4)],
             devices=[dev(1,"rtx-3090",0,3), dev(2,"rtx-3090",0,3), dev(3,"rtx-3090",0,3,8), dev(4,"rtx-3060-12g",0,3,4)])
eng5 = [EngineSpec(id="mk", name="Strata measured", engine="strata", version="0.1.43.1", model_id="qwen3.8-flash-next", gpus=["gpu-1","gpu-2"], cpu_cores=9, args={"batch":4}, benchmarks={"decode_1":132,"decode_4":175,"prefill":3300}),
        EngineSpec(id="sub", name="27B measured", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-3"], args={"tensor_parallel":"1","max_num_seqs":4}, benchmarks={"decode_1":39,"prefill":1050,"kv_room_tokens":21000}),
        EngineSpec(id="co", name="coordinator", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash")]
run("Box with measured values entered", m5, eng5, Roles(coordinator="co", maker="mk", sub="sub"))
t=trace(SimRequest(machine=m5, engines=eng5, roles=Roles(coordinator="co", maker="mk", sub="sub"), concurrency=[2], scenarios=2), 2, 0)
print("   trace:", len(t["events"]), "events,", round(t["makespan_s"]/60), "min")
