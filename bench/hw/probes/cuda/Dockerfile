# rigsim hardware probe: cuBLAS GEMM, VRAM streaming, sustained load at a busy share.
# Built on the build host (ansible/images.yml), pushed to the internal registry, pulled by digest on the box.
ARG CUDA=12.8.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
WORKDIR /src
COPY src/probe.cu .
# sm_86 = consumer Ampere (3060-3090), sm_89 = Ada, sm_80 = A100, sm_90 = Hopper; PTX for anything newer
RUN nvcc -O3 -std=c++17 -o /probe probe.cu -lcublas \
    -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
COPY --from=build /probe /usr/local/bin/probe
RUN useradd -u 10001 -M probe
USER probe
ENTRYPOINT []
LABEL org.opencontainers.image.title="rigsim-hwbench-cuda" rigsim.hwbench.probe="cuda"
