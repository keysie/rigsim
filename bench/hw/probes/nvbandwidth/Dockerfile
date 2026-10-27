# NVIDIA nvbandwidth (host <-> device and device <-> device copy bandwidth), built from source at a pinned tag.
ARG CUDA=12.8.1
ARG NVBANDWIDTH_TAG=v0.7
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG NVBANDWIDTH_TAG
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libboost-program-options-dev \
    ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${NVBANDWIDTH_TAG} https://github.com/NVIDIA/nvbandwidth.git /src
WORKDIR /src
RUN cmake -DCMAKE_BUILD_TYPE=Release . && make -j"$(nproc)"

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libboost-program-options1.83.0 python3-minimal \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/nvbandwidth /usr/local/bin/nvbandwidth
COPY nvb /usr/local/bin/nvb
RUN chmod 0755 /usr/local/bin/nvb && useradd -u 10001 -M probe
USER probe
ENTRYPOINT []
LABEL org.opencontainers.image.title="rigsim-hwbench-nvbandwidth" rigsim.hwbench.probe="nvbandwidth"
