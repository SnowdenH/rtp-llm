#include "rtp_llm/cpp/kernels/mask_logits_csr.h"
#include "rtp_llm/cpp/kernels/mask_logits.h"  // 复用 NegativeInfinity<T>

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: csr_mask_logits
//
// 每个 block 处理一个 beam（blockIdx.x = beam_idx）。
// 执行两阶段：
//   Phase 1 — 将整行 logits 置 -inf（所有线程按步长扫描 vocab）
//   Phase 2 — 将 col_idx[start..end) 对应位置恢复原值（unmask 合法 token）
//
// 这样避免了 CPU 构建 uint8 mask 再 H2D 的开销。
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void csr_mask_logits_kernel(
    T*         logits,      // [batch_size, vocab_size]
    const int* states,      // [batch_size]  当前状态 id
    const int* row_ptr,     // [num_states+1]
    const int* col_idx,     // [nnz]
    int        vocab_size)
{
    int beam_idx = blockIdx.x;
    int state    = states[beam_idx];

    // 状态无效（已退出树模式），不做任何操作
    if (state < 0) return;

    T* row_logits = logits + (size_t)beam_idx * vocab_size;

    int start = row_ptr[state];
    int end   = row_ptr[state + 1];

    // ── Phase 1: 整行置 -inf ──────────────────────────────────────────────
    // blockDim.x 个线程协作扫描 vocab_size 个位置
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        row_logits[v] = NegativeInfinity<T>();
    }

    // 等待 Phase 1 完成（所有线程都写完后再 unmask）
    __syncthreads();

    // ── Phase 2: unmask 合法 token ────────────────────────────────────────
    // 合法 token 数量（end - start）通常远小于 vocab_size，
    // 直接让线程 0 到 (end-start-1) 各负责一个 token
    int num_candidates = end - start;
    for (int k = threadIdx.x; k < num_candidates; k += blockDim.x) {
        int token_id = col_idx[start + k];
        if (token_id >= 0 && token_id < vocab_size) {
            // 恢复原始 logit（通过原始 logits buffer 已被覆盖，所以需要在 Phase 1 之前保存）
            // ── 注意：由于 Phase 1 已把整行置 -inf，这里只能设为 0 作为占位，
            //    实际使用中调用方应在 Phase 1 前先 gather 原始 logprobs 到临时 buffer，
            //    或者改用"只 mask 不合法 token"策略（见下方说明）。
            //
            // ★ 实际采用策略：先保存原始 logits → 全置 -inf → 再从保存值恢复合法位置
            //    由于 shared memory 受限，这里改为"只 mask 不合法 token"策略：
            //    不做 Phase 1 全置 -inf，改用 mask 方式（见 csr_mask_logits_kernel_v2）
            (void)token_id;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 (v2, 正式版): csr_mask_logits_kernel_v2
//
// 策略改为"只 mask 不合法 token"：
//   先将整行置 -inf，同时通过 __ldg 缓存原始值；
//   但因为 Phase1 和 Phase2 使用同一块 HBM，正确做法是：
//
//   【分两步 kernel】或【shared memory 缓冲】。
//
// 实际最简实现：
//   block = 1 个 beam，shared memory 存该 beam 候选 token 的原始 logit。
//   步骤：
//     1. 线程 0..num_candidates-1 先从 logits 里读出合法 token 的原始值存入 shmem
//     2. __syncthreads()
//     3. 整行置 -inf
//     4. __syncthreads()
//     5. 将 shmem 里的值写回合法位置
//
// limit = 当前层最大分支数，用于静态 shmem 分配（模板参数）
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void csr_mask_logits_kernel_v2(
    T*         logits,      // [batch_size, vocab_size]，读写
    const int* states,      // [batch_size]
    const int* row_ptr,     // [num_states+1]
    const int* col_idx,     // [nnz]
    int        vocab_size,
    int        limit)       // 当前层最大分支数（shmem 上界）
{
    extern __shared__ char shmem_raw[];
    T*   shmem_logits = reinterpret_cast<T*>(shmem_raw);       // 原始 logit 值缓存
    int* shmem_tokens = reinterpret_cast<int*>(shmem_logits + limit); // 对应 token id

    int beam_idx = blockIdx.x;
    int state    = states[beam_idx];

    // 状态无效（已退出树模式），不做任何操作
    if (state < 0) return;

    T*  row_logits    = logits + (size_t)beam_idx * vocab_size;
    int start         = row_ptr[state];
    int end           = row_ptr[state + 1];
    int num_candidates = end - start;

    // ── Phase 1: 将合法 token 的原始 logit 读入 shmem ─────────────────────
    // 用 __ldg() 走只读缓存路径，减少随机散读对 L1 data cache 的污染
    for (int k = threadIdx.x; k < num_candidates; k += blockDim.x) {
        int token_id        = col_idx[start + k];
        shmem_tokens[k]     = token_id;
        shmem_logits[k]     = (token_id >= 0 && token_id < vocab_size)
                                  ? __ldg(&row_logits[token_id])
                                  : NegativeInfinity<T>();
    }
    __syncthreads();

    // ── Phase 2: 整行置 -inf ──────────────────────────────────────────────
    // 尝试用 uint4（128bit）向量写提升内存带宽利用率；
    // 运行时检查地址对齐，未对齐则退化为标量写（避免 misaligned store UB）
    const T neg_inf = NegativeInfinity<T>();
    constexpr int elems_per_vec = 16 / sizeof(T);  // float:4, half:8, bf16:8

    // 检查 row_logits 是否 16 字节对齐
    bool aligned = (reinterpret_cast<uintptr_t>(row_logits) % 16 == 0);
    if (aligned) {
        int vec_count = vocab_size / elems_per_vec;
        int rem_start = vec_count * elems_per_vec;
        // 构造填满 neg_inf 的 uint4
        uint4 fill_vec;
        {
            T* fill_ptr = reinterpret_cast<T*>(&fill_vec);
            for (int e = 0; e < elems_per_vec; ++e) fill_ptr[e] = neg_inf;
        }
        uint4* row_vec = reinterpret_cast<uint4*>(row_logits);
        for (int v = threadIdx.x; v < vec_count; v += blockDim.x) {
            row_vec[v] = fill_vec;
        }
        // 尾部不能整除部分标量写
        for (int v = rem_start + threadIdx.x; v < vocab_size; v += blockDim.x) {
            row_logits[v] = neg_inf;
        }
    } else {
        // 地址未对齐：退化到标量写
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            row_logits[v] = neg_inf;
        }
    }
    __syncthreads();

    // ── Phase 3: 将 shmem 里的合法 logit 写回 ─────────────────────────────
    for (int k = threadIdx.x; k < num_candidates; k += blockDim.x) {
        int token_id = shmem_tokens[k];
        if (token_id >= 0 && token_id < vocab_size) {
            row_logits[token_id] = shmem_logits[k];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: csr_update_states
//
// 每个线程处理一个 beam。
// 对每个 beam，在 col_idx[row_ptr[s]..row_ptr[s+1]) 中二分查找 new_token：
//   - 找到 → states[i] = next_state[edge_pos]
//   - 未找到 → states[i] = -1（触发 CPU 侧退出树模式）
//   - 遇到 end_token → states[i] = -2（正常结束标记）
// ─────────────────────────────────────────────────────────────────────────────
__global__ void csr_update_states_kernel(
    int*       states,       // [batch_size]，读写
    const int* new_tokens,   // [batch_size]
    const int* row_ptr,      // [num_states+1]
    const int* col_idx,      // [nnz]
    const int* next_state,   // [nnz]
    int        end_token_id,
    int        batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    int state = states[i];
    if (state < 0) return;  // 已退出，跳过

    int token = new_tokens[i];

    // 遇到结束 token：正常结束
    if (token == end_token_id) {
        states[i] = -2;
        return;
    }

    int seg_start = row_ptr[state];
    int seg_end   = row_ptr[state + 1];

    // 在 col_idx[seg_start..seg_end) 中二分查找 token（该区间已升序排列）
    int lo = seg_start, hi = seg_end;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (col_idx[mid] < token) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo < seg_end && col_idx[lo] == token) {
        // 找到：推进状态
        states[i] = next_state[lo];
    } else {
        // 未找到：违反约束
        states[i] = -1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host 启动函数
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
void invokeCSRMaskLogits(
    T*           logits,
    const int*   states,
    const int*   row_ptr,
    const int*   col_idx,
    int          batch_size,
    int          vocab_size,
    int          limit,
#if USING_CUDA
    cudaStream_t stream)
#elif USING_ROCM
    hipStream_t  stream)
#endif
{
    // Phase 2（全行 -inf 写）需要尽可能多的线程来满足内存带宽。
    // Phase 1/3 只处理 num_candidates ≤ limit 个元素，线程数按 max(limit, 256) 取、上限 1024。
    // 多余线程在 Phase 1/3 的循环里自然空跑，不影响正确性。
    int block_x = ((limit + 31) / 32 * 32);
    if (block_x < 256) block_x = 256;   // 至少 256 线程保证 Phase 2 带宽
    if (block_x > 1024) block_x = 1024;

    dim3 block(block_x);
    dim3 grid(batch_size);

    // shmem = limit 个 T（保存原始 logit）+ limit 个 int（保存 token id）
    size_t shmem_bytes = (size_t)limit * (sizeof(T) + sizeof(int));

    csr_mask_logits_kernel_v2<T><<<grid, block, shmem_bytes, stream>>>(
        logits, states, row_ptr, col_idx, vocab_size, limit);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

void invokeCSRUpdateStates(
    int*         states,
    const int*   new_tokens,
    const int*   row_ptr,
    const int*   col_idx,
    const int*   next_state,
    int          end_token_id,
    int          batch_size,
#if USING_CUDA
    cudaStream_t stream)
#elif USING_ROCM
    hipStream_t  stream)
#endif
{
    // 每个线程处理一个 beam，使用 1D grid
    int block_x = 256;
    int grid_x  = (batch_size + block_x - 1) / block_x;

    csr_update_states_kernel<<<grid_x, block_x, 0, stream>>>(
        states, new_tokens, row_ptr, col_idx, next_state, end_token_id, batch_size);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

// 显式实例化三种精度
template void invokeCSRMaskLogits<float>(
    float*, const int*, const int*, const int*, int, int, int,
#if USING_CUDA
    cudaStream_t);
#elif USING_ROCM
    hipStream_t);
#endif

template void invokeCSRMaskLogits<half>(
    half*, const int*, const int*, const int*, int, int, int,
#if USING_CUDA
    cudaStream_t);
#elif USING_ROCM
    hipStream_t);
#endif

template void invokeCSRMaskLogits<__nv_bfloat16>(
    __nv_bfloat16*, const int*, const int*, const int*, int, int, int,
#if USING_CUDA
    cudaStream_t);
#elif USING_ROCM
    hipStream_t);
#endif

}  // namespace rtp_llm
