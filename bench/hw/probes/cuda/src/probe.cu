// rigsim hardware probe (CUDA + cuBLAS). Prints one JSON object per result line on stdout.
//   probe gemm   --prec fp32|tf32|fp16_fp32acc|fp16|bf16|int8 --shape MxNxK [--reps 50] [--warmup 5]
//   probe stream --op read|write|copy [--mib 4096] [--reps 20]
//   probe load   --kind compute|memory [--duty 1.0] [--seconds 30] [--period-ms 20]
// Inputs are random (power depends on the data: zeros would draw less and run at a higher clock).
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at line %d\n", (int)s_, __LINE__); exit(3); } } while (0)

static std::string arg(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 2; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return dflt;
}

// ---- random fill ----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (uint32_t)x;
}
__global__ void fill_f32(float* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (mix(i) & 0xffff) / 65536.0f - 0.5f;
}
__global__ void fill_f16(__half* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half((mix(i) & 0xffff) / 65536.0f - 0.5f);
}
__global__ void fill_bf16(__nv_bfloat16* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2bfloat16((mix(i) & 0xffff) / 65536.0f - 0.5f);
}
__global__ void fill_i8(int8_t* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (int8_t)((mix(i) % 255) - 127);
}

// ---- GEMM -------------------------------------------------------------------------------------------------------
struct Prec {
    const char* name; cudaDataType_t ab, c; cublasComputeType_t compute; size_t ab_bytes, c_bytes; int scale; // 0 f32 1 f16 2 i32
};
static const Prec PRECS[] = {
    {"fp32", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 4, 4, 0},
    {"tf32", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4, 4, 0},
    {"fp16_fp32acc", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 2, 2, 0},
    {"fp16", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, 2, 2, 1},
    {"bf16", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 2, 2, 0},
    {"int8", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, 1, 4, 2},
};

static void fill(void* p, cudaDataType_t t, size_t n) {
    switch (t) {
        case CUDA_R_32F: fill_f32<<<1024, 256>>>((float*)p, n); break;
        case CUDA_R_16F: fill_f16<<<1024, 256>>>((__half*)p, n); break;
        case CUDA_R_16BF: fill_bf16<<<1024, 256>>>((__nv_bfloat16*)p, n); break;
        case CUDA_R_8I: fill_i8<<<1024, 256>>>((int8_t*)p, n); break;
        default: CK(cudaMemset(p, 0, n * 4));
    }
    CK(cudaGetLastError());
}

struct Gemm {
    const Prec* p; int m, n, k; void *A, *B, *C; cublasHandle_t h;
    float af = 1.f, bf = 0.f; __half ah, bh; int32_t ai = 1, bi = 0;
    Gemm(const Prec* p_, int m_, int n_, int k_) : p(p_), m(m_), n(n_), k(k_) {
        CB(cublasCreate(&h));
        ah = __float2half(1.f); bh = __float2half(0.f);
        CK(cudaMalloc(&A, (size_t)m * k * p->ab_bytes)); CK(cudaMalloc(&B, (size_t)k * n * p->ab_bytes));
        CK(cudaMalloc(&C, (size_t)m * n * p->c_bytes));
        fill(A, p->ab, (size_t)m * k); fill(B, p->ab, (size_t)k * n);
        CK(cudaMemset(C, 0, (size_t)m * n * p->c_bytes));
        CK(cudaDeviceSynchronize());
    }
    ~Gemm() { cudaFree(A); cudaFree(B); cudaFree(C); cublasDestroy(h); }
    cublasStatus_t call() {
        const void* alpha = p->scale == 0 ? (const void*)&af : p->scale == 1 ? (const void*)&ah : (const void*)&ai;
        const void* beta = p->scale == 0 ? (const void*)&bf : p->scale == 1 ? (const void*)&bh : (const void*)&bi;
        // column-major C(m x n) = op(A) (m x k) * B (k x n). INT8 (IMMA) wants A transposed: A stored k x m.
        bool t = p->ab == CUDA_R_8I;
        return cublasGemmEx(h, t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                            A, p->ab, t ? k : m, B, p->ab, k, beta, C, p->c, m, p->compute, CUBLAS_GEMM_DEFAULT);
    }
    double ops() const { return 2.0 * m * (double)n * k; }
};

static const Prec* prec_by_name(const std::string& s) {
    for (auto& p : PRECS) if (s == p.name) return &p;
    fprintf(stderr, "unknown precision %s\n", s.c_str()); exit(1);
}

static int gemm_mode(int argc, char** argv) {
    const Prec* p = prec_by_name(arg(argc, argv, "--prec", "fp16"));
    int m, n, k;
    if (sscanf(arg(argc, argv, "--shape", "4096x4096x4096").c_str(), "%dx%dx%d", &m, &n, &k) != 3) return 1;
    int reps = atoi(arg(argc, argv, "--reps", "50").c_str()), warm = atoi(arg(argc, argv, "--warmup", "5").c_str());
    Gemm g(p, m, n, k);
    cublasStatus_t first = g.call();
    if (first == CUBLAS_STATUS_NOT_SUPPORTED || first == CUBLAS_STATUS_INVALID_VALUE) {
        // a result, not an error: this precision / shape has no path on this card and library
        printf("{\"mode\":\"gemm\",\"prec\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"unsupported\":true,\"status\":%d}\n",
               p->name, m, n, k, (int)first);
        return 0;
    }
    CB(first);
    for (int i = 0; i < warm; ++i) CB(g.call());
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms(reps);
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); CB(g.call()); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::vector<float> s = ms; std::sort(s.begin(), s.end());
    double med = s[reps / 2] / 1e3;
    int ver; CB(cublasGetVersion(g.h, &ver)); int rt; CK(cudaRuntimeGetVersion(&rt));
    printf("{\"mode\":\"gemm\",\"prec\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"reps\":%d,\"ops\":%.0f,"
           "\"t_median_s\":%.9g,\"t_min_s\":%.9g,\"t_max_s\":%.9g,\"ops_per_s\":%.6g,\"cublas\":%d,\"cudart\":%d}\n",
           p->name, m, n, k, reps, g.ops(), med, s[0] / 1e3, s[reps - 1] / 1e3, g.ops() / med, ver, rt);
    return 0;
}

// ---- streaming ----------------------------------------------------------------------------------------------------
__global__ void k_read(const float4* __restrict__ x, size_t n, float* out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = x[i]; acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;                       // keeps the loads; practically never stores
}
__global__ void k_write(float4* __restrict__ x, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] = make_float4(1.f, 2.f, 3.f, (float)i);
}
__global__ void k_copy(const float4* __restrict__ x, float4* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = x[i];
}

struct Stream {
    size_t bytes, n; float4 *x, *y; float* out; int blocks;
    Stream(size_t mib) : bytes(mib << 20), n((mib << 20) / sizeof(float4)) {
        CK(cudaMalloc(&x, bytes)); CK(cudaMalloc(&y, bytes)); CK(cudaMalloc(&out, 4));
        fill_f32<<<1024, 256>>>((float*)x, bytes / 4); CK(cudaDeviceSynchronize());
        int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0)); blocks = sms * 8;
    }
    ~Stream() { cudaFree(x); cudaFree(y); cudaFree(out); }
    double run(const std::string& op) {                       // bytes moved
        if (op == "read") { k_read<<<blocks, 256>>>(x, n, out); return (double)bytes; }
        if (op == "write") { k_write<<<blocks, 256>>>(y, n); return (double)bytes; }
        k_copy<<<blocks, 256>>>(x, y, n); return 2.0 * bytes;
    }
};

static int stream_mode(int argc, char** argv) {
    std::string op = arg(argc, argv, "--op", "read");
    size_t mib = strtoull(arg(argc, argv, "--mib", "4096").c_str(), nullptr, 10);
    int reps = atoi(arg(argc, argv, "--reps", "20").c_str());
    Stream s(mib);
    for (int i = 0; i < 3; ++i) s.run(op);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms(reps); double moved = 0;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); moved = s.run(op); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms[i], a, b));
    }
    CK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    double med = ms[reps / 2] / 1e3;
    printf("{\"mode\":\"stream\",\"op\":\"%s\",\"bytes\":%.0f,\"reps\":%d,\"t_median_s\":%.9g,\"t_min_s\":%.9g,"
           "\"bytes_per_s\":%.6g}\n", op.c_str(), moved, reps, med, ms[0] / 1e3, moved / med);
    return 0;
}

// ---- sustained load at a busy share -----------------------------------------------------------------------------
static int load_mode(int argc, char** argv) {
    std::string kind = arg(argc, argv, "--kind", "compute");
    double duty = atof(arg(argc, argv, "--duty", "1.0").c_str());
    double seconds = atof(arg(argc, argv, "--seconds", "30").c_str());
    double period = atof(arg(argc, argv, "--period-ms", "20").c_str()) / 1e3;
    if (duty <= 0 || duty > 1) { fprintf(stderr, "duty must be in (0, 1]\n"); return 1; }
    Gemm* g = nullptr; Stream* s = nullptr;
    if (kind == "compute") g = new Gemm(prec_by_name("fp16"), 8192, 8192, 8192);
    else s = new Stream(4096);
    using clk = std::chrono::steady_clock;
    auto t_end = clk::now() + std::chrono::duration<double>(seconds);
    double busy = 0, work = 0; long calls = 0;
    auto t0 = clk::now();
    while (clk::now() < t_end) {
        auto p0 = clk::now();
        auto p_busy_end = p0 + std::chrono::duration<double>(period * duty);
        do {                                                   // whole calls only; a call is ~1-10 ms
            if (g) { CB(g->call()); work += g->ops(); } else { work += s->run("read"); }
            ++calls;
            CK(cudaDeviceSynchronize());
        } while (clk::now() < p_busy_end);
        busy += std::chrono::duration<double>(clk::now() - p0).count();
        auto p_end = p0 + std::chrono::duration<double>(period);
        if (duty < 1.0 && clk::now() < p_end) std::this_thread::sleep_until(p_end);
    }
    double total = std::chrono::duration<double>(clk::now() - t0).count();
    printf("{\"mode\":\"load\",\"kind\":\"%s\",\"duty_set\":%.3f,\"busy_share\":%.4f,\"seconds\":%.3f,\"calls\":%ld,"
           "\"%s\":%.6g}\n", kind.c_str(), duty, busy / total, total, calls,
           g ? "ops_per_busy_s" : "bytes_per_busy_s", work / busy);
    delete g; delete s;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: probe gemm|stream|load ...\n"); return 1; }
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    std::string mode = argv[1];
    if (mode == "gemm") return gemm_mode(argc, argv);
    if (mode == "stream") return stream_mode(argc, argv);
    if (mode == "load") return load_mode(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
