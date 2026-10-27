# rigsim CPU / storage probe: likwid-bench core ladders, pointer-chase latency, fio.
FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY src/chase.c /src/chase.c
RUN gcc -O2 -o /chase /src/chase.c

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends likwid fio python3-minimal && rm -rf /var/lib/apt/lists/*
COPY --from=build /chase /usr/local/bin/chase
COPY src/cpuprobe /usr/local/bin/cpuprobe
RUN chmod 0755 /usr/local/bin/cpuprobe && useradd -u 10001 -M probe
USER probe
ENTRYPOINT []
LABEL org.opencontainers.image.title="rigsim-hwbench-cpu" rigsim.hwbench.probe="cpu"
