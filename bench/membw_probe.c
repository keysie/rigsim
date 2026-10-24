/* Memory bandwidth probe for the bench's calibration (2026-10-21, user: "we know where 100 % actually is").
 * Own code (not STREAM): kernels read (sum), write (store), copy (a = b), triad (a = b + s*c) over arrays far larger
 * than the caches, OpenMP threads from OMP_NUM_THREADS, each kernel repeated until --seconds have passed.
 * Prints one CSV line per kernel: kernel,threads,counted_gbs,seconds  (counted = bytes the kernel names, without the
 * write-allocate reads the hardware adds — pcm-memory measures those; the calibration takes pcm's figure as the peak).
 *   gcc -O3 -march=native -fopenmp membw_probe.c -o membw_probe && OMP_NUM_THREADS=10 ./membw_probe 1024 10
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], 0, 10) : 1024;      /* per array */
    double secs = argc > 2 ? atof(argv[2]) : 10.0;
    const char *only = argc > 3 ? argv[3] : "";
    size_t n = mib * 1024 * 1024 / sizeof(double);
    double *a = aligned_alloc(64, n * sizeof(double)), *b = aligned_alloc(64, n * sizeof(double)),
           *c = aligned_alloc(64, n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }     /* first touch per thread */
    const char *names[] = {"read", "write", "copy", "triad"};
    const double bytes[] = {8.0, 8.0, 16.0, 24.0};
    volatile double sink = 0;
    int th = omp_get_max_threads();
    for (int k = 0; k < 4; k++) {
        if (*only && strcmp(only, names[k])) continue;
        long reps = 0; double t0 = now(), t;
        do {
            if (k == 0) { double s = 0;
#pragma omp parallel for reduction(+:s) schedule(static)
                for (size_t i = 0; i < n; i++) s += a[i];
                sink += s; }
            else if (k == 1) {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) a[i] = 3.0; }
            else if (k == 2) {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) a[i] = b[i]; }
            else {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) a[i] = b[i] + 1.5 * c[i]; }
            reps++; t = now() - t0;
        } while (t < secs);
        printf("%s,%d,%.2f,%.2f\n", names[k], th, bytes[k] * n * reps / t / 1e9, t);
        fflush(stdout);
    }
    return sink == 42.0;   /* keep the reads */
}
