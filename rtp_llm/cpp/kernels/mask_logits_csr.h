#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

/**
 * GPU CSR 约束 logits masking kernel 的启动接口
 *
 * 逻辑（对应参考实现 generate_and_apply_logprobs_mask）：
 *   1. 对每个 beam，读取 row_ptr[state] 得到候选区间 [start, end)
 *   2. 先将整行 logits 置 -inf
 *   3. 再将 col_idx[start..end) 对应位置恢复原值（unmask 合法 token）
 *
 * @param logits        [batch_size, vocab_size]  float/half/bf16，GPU 上
 * @param states        [batch_size]              int32，每个 beam 当前状态，GPU 上
 * @param row_ptr       [num_states + 1]          int32，CSR 行指针，GPU 上
 * @param col_idx       [nnz]                     int32，候选 token id，GPU 上
 * @param batch_size    beam 总数（batch_size * num_beams）
 * @param vocab_size    词表大小
 * @param limit         当前层最大分支数（kernel 内用于分配寄存器循环上界）
 * @param stream        CUDA/HIP stream
 */
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
    cudaStream_t stream);
#elif USING_ROCM
    hipStream_t  stream);
#endif

/**
 * GPU CSR 状态更新 kernel 的启动接口
 *
 * 对每个 beam，根据刚生成的 token 推进当前状态：
 *   1. 在 col_idx[row_ptr[s] .. row_ptr[s+1]) 中二分查找 token
 *   2. 找到则更新 states[i] = next_state[edge_pos]
 *   3. 未找到则将 states[i] 设为 -1（触发退出树模式）
 *
 * @param states        [batch_size]  int32，读写，GPU 上
 * @param new_tokens    [batch_size]  int32，刚生成的 token，GPU 上
 * @param row_ptr       [num_states + 1] int32，GPU 上
 * @param col_idx       [nnz]         int32，GPU 上
 * @param next_state    [nnz]         int32，GPU 上
 * @param end_token_id  结束 token id，遇到则将 state 设为 -2（特殊：正常结束）
 * @param batch_size    beam 总数
 * @param stream        CUDA/HIP stream
 */
void invokeCSRUpdateStates(
    int*         states,
    const int*   new_tokens,
    const int*   row_ptr,
    const int*   col_idx,
    const int*   next_state,
    int          end_token_id,
    int          batch_size,
#if USING_CUDA
    cudaStream_t stream);
#elif USING_ROCM
    hipStream_t  stream);
#endif

}  // namespace rtp_llm
