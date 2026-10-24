const REPO = "https://github.com/keysie/rigsim/blob/main";

function Code({ children }: { children: string }) {
  return <pre className="mt-2 overflow-x-auto rounded border border-line bg-paper p-3 text-xs leading-relaxed"><code>{children}</code></pre>;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="mt-6 rounded-lg border border-line bg-panel p-5"><h2>{title}</h2>{children}</section>;
}

export default function MeasurePage() {
  return (
    <div className="max-w-4xl">
      <h1>How to measure</h1>
      <p className="mt-1 text-muted">The simulator wants four numbers per engine, taken with the engine alone on its cards: writing speed with one request,
        writing speed with N requests at once (total), prompt-reading speed, and the KV room. Enter them in the engine's <em>Measured</em> block.
        Everything below reports medians over at least three runs; one run is an anecdote.</p>

      <Section title="Before any run">
        <ul className="mt-2 list-disc pl-5 text-sm leading-relaxed">
          <li>Fix what varies: GPU power limit (<code>nvidia-smi -pl 250</code>), fans, no other GPU process, no desktop compositor on the card (Linux: run headless or on another GPU).</li>
          <li>Record the environment once: GPU model, driver, link width under load, CPU, RAM channels and speed, OS, engine version and command line.</li>
          <li>Warm up: one 512-token request after each engine start, discarded.</li>
          <li>Separate prompt reading from writing. Engines report both; wall time divided by tokens mixes them.</li>
        </ul>
        <Code>{`nvidia-smi --query-gpu=name,driver_version,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv
lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | sed 's/Flags:.*\\(avx512f\\|avx2\\).*/vector: \\1/'
sudo dmidecode -t memory | grep -E 'Configured Memory Speed|Locator: DIMM' | sort | uniq -c`}</Code>
      </Section>

      <Section title="Memory bandwidth of a CPU pool">
        <p className="mt-1 text-sm text-muted">The CPU expert path is capped by what the memory pool really delivers. Measure it instead of trusting channels × speed.</p>
        <Code>{`# the box's calibrator: STREAM-style read/copy over the whole pool, cached by hardware fingerprint
python3 bench/membw-calibrate.py            # prints read / copy GB/s and the fingerprint
gcc -O3 -march=native -fopenmp bench/membw_probe.c -o membw_probe && ./membw_probe 4096`}</Code>
        <p className="mt-2 text-sm">Scripts: <a className="underline" href={`${REPO}/bench/membw-calibrate.py`} target="_blank" rel="noreferrer">membw-calibrate.py</a>,{" "}
          <a className="underline" href={`${REPO}/bench/membw_probe.c`} target="_blank" rel="noreferrer">membw_probe.c</a>. Expect 70–75 % of the theoretical figure on a healthy 4-channel DDR4 system; much less means a channel is not populated.</p>
      </Section>

      <Section title="Strata">
        <p className="mt-1 text-sm text-muted">Strata ships a benchmark that writes a report in the community format; its log also prints the expert-cache hit and the window times.</p>
        <Code>{`./bench.sh                                   # or: python tools/strata_bench.py --url http://127.0.0.1:8080
# log lines to keep: "strata batch:" (rows, window ms), "expert cache ... hit", "INFO batch_groups="
# N parallel: send N identical long generations at once and read the engine's decode tok/s per slot
curl -s localhost:8080/metrics | grep -E 'live.slots|hit'`}</Code>
        <p className="mt-2 text-sm">The calibration points were produced with <a className="underline" href={`${REPO}/bench/benchw.py`} target="_blank" rel="noreferrer">benchw.py</a> (workloads: tasks, multiturn, throughput, agentloop, longtask, guard, judge, rerank),
          driven by the queue under <a className="underline" href={`${REPO}/bench/queue/`} target="_blank" rel="noreferrer">bench/queue/</a> with the pass definitions in{" "}
          <a className="underline" href={`${REPO}/bench/queue/plans/`} target="_blank" rel="noreferrer">plans/</a>. <a className="underline" href={`${REPO}/bench/to-measured.py`} target="_blank" rel="noreferrer">to-measured.py</a> turns a result file into the values the engine block takes.</p>
        <p className="mt-2 text-sm">What to enter: <em>writes alone</em> = decode tok/s with one request and MTP on; <em>writes with N</em> = total tok/s with N slots busy; <em>reads</em> = prompt tok/s for a fresh 32K prompt.</p>
      </Section>

      <Section title="llama.cpp">
        <Code>{`# weights only, no KV effects: text generation and prompt processing
llama-bench -m model.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 5          # pp512 and tg128, medians of 5
llama-bench -m model.gguf -ngl 99 -fa 1 -p 512 -n 128 -b 2048 -ub 512
# N slots: start the server with -np N -c N*ctx, then fire N concurrent requests and sum tokens/s
llama-server -m model.gguf -ngl 99 -fa -np 3 -c 98304 --port 8081
python3 bench/benchw.py --url http://127.0.0.1:8081 --kind throughput --clients 3 --tokens 512`}</Code>
        <p className="mt-2 text-sm">tg128 is close to <em>writes alone</em> at a short context; for agent work also measure at a 16K prompt, where KV reads cost 10–20 %.</p>
      </Section>

      <Section title="vLLM">
        <Code>{`# vLLM's own tools, in the vllm repo (benchmarks/)
python benchmarks/benchmark_serving.py --backend openai --base-url http://127.0.0.1:8000 \\
  --model Qwen3.8-27B-AWQ --dataset-name random --random-input-len 16384 --random-output-len 512 \\
  --num-prompts 8 --max-concurrency 4
# KV room: read it from the startup log
grep -E 'GPU KV cache size|Maximum concurrency' vllm.log`}</Code>
        <p className="mt-2 text-sm">Enter output tokens/s per request at concurrency 1 as <em>writes alone</em>, the total at concurrency N as <em>writes with N</em>, the prefill tok/s as <em>reads</em>, and "GPU KV cache size" in tokens as <em>KV room</em>.</p>
      </Section>

      <Section title="Guard, judge, reranker">
        <p className="mt-1 text-sm text-muted">These are latency, not throughput, questions: seconds per call at the sizes the workload produces.</p>
        <Code>{`# guard: one call per tool result; try 8K and 28K outputs (the simulator's p90 / p99)
python3 bench/benchw.py --url http://127.0.0.1:8453 --kind guard --sizes 8192,28000 --repeat 3
# judge: history 4K / 12K with 0, 2, 5 evidence reads, thinking off and low
python3 bench/benchw.py --url http://127.0.0.1:8454 --kind judge --history 4096,12288 --reads 0,2,5
# reranker: pairs per second over 40 candidates
python3 bench/benchw.py --url http://127.0.0.1:8455 --kind rerank --candidates 40`}</Code>
        <p className="mt-2 text-sm">The box's pass 4 did exactly this; the plan files are in <a className="underline" href={`${REPO}/bench/queue/plans/`} target="_blank" rel="noreferrer">plans/</a> (gen_phase4.py, p4-small.toml, p4-cpu.toml).</p>
      </Section>

      <Section title="Workload numbers">
        <p className="mt-1 text-sm text-muted">The ranges on the roles page are assumptions until real traces replace them. From a day of normal use:</p>
        <Code>{`# judge verdicts, rounds per verdict, latency: the agent harness's verdict table
psql -c "select tier, count(*), percentile_cont(0.5) within group (order by latency_ms) from intent_verdicts group by tier"
# guard calls per minute and prompt sizes: the reverse proxy's access log for the guard route
jq -r 'select(.request.uri|test("/guard/")) | [.ts, .size] | @tsv' access.log`}</Code>
        <p className="mt-2 text-sm">Turns per task, tool calls per turn, delegation share and fan-out come from the session logs; approval share from <code>intent_verdicts</code>.
          How to contribute a measurement: <a className="underline" href={`${REPO}/bench/README.md`} target="_blank" rel="noreferrer">bench/README.md</a>.</p>
      </Section>

      <Section title="Before you submit a number">
        <ul className="mt-2 list-disc pl-5 text-sm leading-relaxed">
          <li>Three runs or more, medians, ranges noted.</li>
          <li>Engine-reported tok/s, not wall time. Prompt and decode kept apart.</li>
          <li>Context length, quant, parallel count and power limit written next to the number.</li>
          <li>Reproducible command line saved with it.</li>
        </ul>
        <p className="mt-2 text-sm text-muted">That is what separates the trusted points on the calibration page from the claims below them.</p>
      </Section>
    </div>
  );
}
