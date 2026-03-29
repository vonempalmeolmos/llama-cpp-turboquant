#include "turbo-wht.cuh"
#include "common.cuh"

// Sign arrays — must match ggml-cpu/ops.cpp turbo_wht_s1/s2 exactly.
static __device__ __constant__ float turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __device__ __constant__ float turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// One block per 128-element group, 128 threads per block.
// Applies: signs → 7-stage WHT butterfly → normalize → signs.
static __global__ void k_turbo_wht_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        int64_t n_groups,
        int     direction) {
    __shared__ float x[128];

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int i = threadIdx.x;  // 0..127

    const float * s_first  = (direction == 0) ? turbo_wht_s1 : turbo_wht_s2;
    const float * s_second = (direction == 0) ? turbo_wht_s2 : turbo_wht_s1;

    // Load and apply first signs
    x[i] = src[g * 128 + i] * s_first[i];
    __syncthreads();

    // WHT butterfly (7 stages), each thread handles the lower element of its pair
    for (int h = 1; h < 128; h *= 2) {
        if ((i % (h * 2)) < h) {
            const float a = x[i];
            const float b = x[i + h];
            x[i]     = a + b;
            x[i + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs
    dst[g * 128 + i] = x[i] * 0.08838834764831845f * s_second[i];
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(src) % 128 == 0);

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_total  = ggml_nelements(src);
    const int64_t n_groups = n_total / 128;

    if (n_groups == 0) return;

    k_turbo_wht_f32<<<(unsigned)n_groups, 128, 0, ctx.stream()>>>(
        (const float *) src->data,
        (float *)       dst->data,
        n_groups,
        direction
    );
}
