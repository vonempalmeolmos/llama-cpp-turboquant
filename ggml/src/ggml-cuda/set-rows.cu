#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "ggml-common.h"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// Forward declaration — defined below after TurboQuant sign arrays and kernel.
template<typename idx_t>
static void set_rows_cuda_turbo3_wht(
        const float *, const idx_t *, block_turbo3_0 *,
        int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t,
        size_t, size_t, size_t,
        size_t, size_t, size_t,
        size_t, size_t, size_t,
        cudaStream_t);

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3_wht<idx_t>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


// ---- TurboQuant fused WHT+quantize kernel for SET_ROWS ----

// Sign arrays duplicated from turbo-wht.cu — static so each CUDA module gets its own copy.
static __device__ __constant__ float k_sr_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __device__ __constant__ float k_sr_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// Fused WHT + turbo3 quantize: one CUDA block (128 threads) per 128-element WHT group.
// Eliminates the graph-side TURBO_WHT(v_cur) op — WHT is applied in shared memory before
// packing, saving one full VRAM round-trip per attention layer.
template<typename idx_t>
static __global__ void k_set_rows_turbo3_wht_quant(
        const float       * __restrict__ src0,
        const idx_t       * __restrict__ src1,
        block_turbo3_0    * __restrict__ dst,
        const int64_t  ne_groups,
        const int64_t  ne10,
        const int64_t  ne11,
        const int64_t  ne12,
        const int64_t  ne13,
        const int64_t  s01,
        const int64_t  s02,
        const int64_t  s03,
        const int64_t  s10,
        const int64_t  s11,
        const int64_t  s12,
        const int64_t  s1,
        const int64_t  s2,
        const int64_t  s3,
        const uint3    ne00,
        const uint3    ne01,
        const uint3    ne02,
        const uint3    ne11_fd,
        const uint3    ne12_fd) {

    __shared__ float x[128];

    const int64_t g = blockIdx.x;
    if (g >= ne_groups) return;

    const int t       = threadIdx.x;   // 0..127
    const int warp_id = t >> 5;        // 0..3  (warp 0-3 each own one turbo3 block)
    const int lane    = t & 31;

    // Decompose group base element index → source / dest row addresses
    const int64_t i_base = g * 128;
    uint32_t      tmp    = (uint32_t)i_base;
    uint2         dm;

    dm                = fast_div_modulo(tmp, ne00);
    const int64_t i00 = dm.y;   // element start within row (multiple of 128)
    tmp               = dm.x;

    dm                = fast_div_modulo(tmp, ne01);
    const int64_t i01 = dm.y;
    tmp               = dm.x;

    dm                = fast_div_modulo(tmp, ne02);
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float    * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_blk = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo3_0)
                                   + (i00 / QK_TURBO3) + warp_id;

    // ---- WHT in shared memory (identical to k_turbo_wht_f32) ----
    x[t] = src_row[i00 + t] * k_sr_wht_s1[t];
    __syncthreads();

    for (int h = 1; h < 128; h <<= 1) {
        if ((t % (h << 1)) < h) {
            const float a = x[t];
            const float b = x[t + h];
            x[t]     = a + b;
            x[t + h] = a - b;
        }
        __syncthreads();
    }

    const float x_val = x[t] * 0.08838834764831845f * k_sr_wht_s2[t];

    // ---- Per-warp quantization (warp = 32 threads = one block_turbo3_0) ----

    // Warp-level norm reduction
    float norm_sq = x_val * x_val;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        norm_sq += __shfl_xor_sync(0xFFFFFFFF, norm_sq, offset);
    const float norm = sqrtf(norm_sq);
    const float inv  = (norm >= 1e-10f) ? (1.0f / norm) : 0.0f;

    // Scalar quantize this lane's element
    int idx = 0;
    if (inv > 0.0f) {
        const float v = x_val * inv;
        if      (v < -0.154259f) idx = 0;
        else if (v < -0.091775f) idx = 1;
        else if (v < -0.043589f) idx = 2;
        else if (v <  0.0f     ) idx = 3;
        else if (v <  0.043589f) idx = 4;
        else if (v <  0.091775f) idx = 5;
        else if (v <  0.154259f) idx = 6;
        else                     idx = 7;
    }

    const int low2 = idx & 0x3;
    const int hi1  = (idx >> 2) & 0x1;

    // Pack 32 sign bits via __ballot_sync
    const uint32_t sign_mask = __ballot_sync(0xFFFFFFFF, hi1);
    // Pack 32 pairs of 2-bit indices: bit0 plane and bit1 plane separately
    const uint32_t qs_bit0   = __ballot_sync(0xFFFFFFFF, low2 & 1);
    const uint32_t qs_bit1   = __ballot_sync(0xFFFFFFFF, (low2 >> 1) & 1);

    if (lane == 0) {
        dst_blk->norm = __float2half(norm);

        // signs[4]: 32-bit mask → 4 bytes
        dst_blk->signs[0] = (uint8_t)((sign_mask >>  0) & 0xFF);
        dst_blk->signs[1] = (uint8_t)((sign_mask >>  8) & 0xFF);
        dst_blk->signs[2] = (uint8_t)((sign_mask >> 16) & 0xFF);
        dst_blk->signs[3] = (uint8_t)((sign_mask >> 24) & 0xFF);

        // qs[8]: each byte packs 4 elements' 2-bit indices
        // Element k*4+j → bits j*2+0 (from qs_bit0) and j*2+1 (from qs_bit1) in byte k
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint8_t n0 = (uint8_t)((qs_bit0 >> (k * 4)) & 0xF);
            const uint8_t n1 = (uint8_t)((qs_bit1 >> (k * 4)) & 0xF);
            uint8_t byte_k = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                byte_k |= (uint8_t)(((n0 >> j) & 1) << (j * 2));
                byte_k |= (uint8_t)(((n1 >> j) & 1) << (j * 2 + 1));
            }
            dst_blk->qs[k] = byte_k;
        }
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turbo3_wht(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % 128 == 0);
    const int64_t ne_groups = (ne00 * ne01 * ne02 * ne03) / 128;
    if (ne_groups == 0) return;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t)ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);

        k_set_rows_turbo3_wht_quant<idx_t><<<(unsigned)ne_groups, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_groups,
            ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// ---- TurboQuant device-side quantize functions for SET_ROWS ----

// Quantize QK_TURBO3=32 pre-rotated floats into one block_turbo3_0.
// Input x is already WHT-rotated (used only when falling back to the non-fused path).
static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO3; i++) norm_sq += src[i] * src[i];
    const float norm = sqrtf(norm_sq);
    dst->norm = __float2half(norm);

    uint8_t qs_out[QK_TURBO3 / 4];
    uint8_t si_out[QK_TURBO3 / 8];
    for (int i = 0; i < QK_TURBO3 / 4; i++) qs_out[i] = 0;
    for (int i = 0; i < QK_TURBO3 / 8; i++) si_out[i] = 0;

    if (norm >= 1e-10f) {
        const float inv = 1.0f / norm;
        for (int i = 0; i < QK_TURBO3; i++) {
            const float x = src[i] * inv;
            int idx;
            if      (x < -0.154259f) idx = 0;
            else if (x < -0.091775f) idx = 1;
            else if (x < -0.043589f) idx = 2;
            else if (x <  0.0f     ) idx = 3;
            else if (x <  0.043589f) idx = 4;
            else if (x <  0.091775f) idx = 5;
            else if (x <  0.154259f) idx = 6;
            else                     idx = 7;

            qs_out[i / 4] |= (uint8_t)((idx & 0x3) << ((i % 4) * 2));
            si_out[i / 8] |= (uint8_t)(((idx >> 2) & 0x1) << (i % 8));
        }
    }

    for (int i = 0; i < QK_TURBO3 / 4; i++) dst->qs[i]    = qs_out[i];
    for (int i = 0; i < QK_TURBO3 / 8; i++) dst->signs[i] = si_out[i];
}

// Quantize QK_TURBO4=128 pre-rotated floats into one block_turbo4_0 (4-bit, nibble-packed).
static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src, block_turbo4_0 * __restrict__ dst) {
    constexpr float CENTROIDS[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };

    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO4; i++) norm_sq += src[i] * src[i];
    const float norm = sqrtf(norm_sq);

    uint8_t indices[QK_TURBO4];
    if (norm < 1e-10f) {
        dst->norm = 0;
        for (int i = 0; i < QK_TURBO4 / 2; i++) dst->qs[i] = 0;
        return;
    }

    const float inv = 1.0f / norm;
    for (int i = 0; i < QK_TURBO4; i++) {
        const float x = src[i] * inv;
        int idx;
        if      (x < -0.145560f) idx =  0;
        else if (x < -0.103361f) idx =  1;
        else if (x < -0.079142f) idx =  2;
        else if (x < -0.060009f) idx =  3;
        else if (x < -0.043430f) idx =  4;
        else if (x < -0.028293f) idx =  5;
        else if (x < -0.013963f) idx =  6;
        else if (x <  0.0f     ) idx =  7;
        else if (x <  0.013963f) idx =  8;
        else if (x <  0.028293f) idx =  9;
        else if (x <  0.043430f) idx = 10;
        else if (x <  0.060009f) idx = 11;
        else if (x <  0.079142f) idx = 12;
        else if (x <  0.103361f) idx = 13;
        else if (x <  0.145560f) idx = 14;
        else                     idx = 15;
        indices[i] = (uint8_t)idx;
    }

    // Norm correction
    float recon_sq = 0.0f;
    for (int i = 0; i < QK_TURBO4; i++) recon_sq += CENTROIDS[indices[i]] * CENTROIDS[indices[i]];
    const float recon = sqrtf(recon_sq);
    dst->norm = __float2half((recon > 1e-10f) ? norm / recon : norm);

    for (int i = 0; i < QK_TURBO4; i += 2) dst->qs[i / 2] = indices[i] | (indices[i + 1] << 4);
}

// ---- end TurboQuant ----

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
