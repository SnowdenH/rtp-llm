#pragma once

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

namespace rtp_llm {

// GPU 上的 CSR 状态转移矩阵，每个请求独立实例，请求入队时一次性构建上传
struct TreeDFAGPUData {
    // CPU 侧（用于 updateStatus 中的 CPU 二分查找）
    std::vector<int32_t> row_ptr;    // [num_states + 1]
    std::vector<int32_t> col_idx;    // [nnz]，每行内升序排列
    std::vector<int32_t> next_state; // [nnz]，转移后的目标状态，-1 表示叶子节点
    int32_t              start_state_id = 0;
    int32_t              end_token_id   = -1;

    // GPU 侧（阶段二 kernel 直接读取）
    rtp_llm::BufferPtr gpu_row_ptr;   // GPU 上的 row_ptr
    rtp_llm::BufferPtr gpu_col_idx;   // GPU 上的 col_idx
    rtp_llm::BufferPtr gpu_next_state; // GPU 上的 next_state（GPU 路径状态更新用）

    // 每层的最大分支数（即对应层所有 state 的 row 宽度最大値）
    // 层编号 = key 中 "_" 分隔符的个数（根节点为第 0 层）
    // 用于 GPU kernel 需要静态分配共享内存的场景
    std::vector<int32_t> per_level_limit; // [max_depth + 1]

    // 是否使用 GPU CSR kernel（由 use_gpu_csr_kernel 字段传入）
    bool use_gpu_kernel = false;

    // state_id -> 原始 prefix key（如 "225_64000"）
    // 用于 getStatus() 返回可读前缀路径
    std::unordered_map<int32_t, std::string> id_to_key;
};

struct StreamTreeInfo {
    bool                                       in_tree_mode;
    int32_t                                    input_length;
    int32_t                                    current_output_length;
    bool                                       is_beam_search;
    std::shared_ptr<TreeDFA<std::string, int>> dfa_ptr;
    bool                                       soft_constraint_mode;

    // 个性化树路径：CSR GPU 矩阵（多个 beam 共享同一份，只读）
    std::shared_ptr<TreeDFAGPUData>            gpu_dfa_data;
    // 每个 beam 独立维护自己的当前状态（CPU 侧）
    int32_t                                    current_state_id = -1;
    // GPU 路径下：单个 beam 当前状态的 GPU buffer（[1] int32）
    // 由 process/updateStatus 的 GPU kernel 直接读写
    rtp_llm::BufferPtr                         gpu_current_state_buf;
    // 预分配 pinned HOST buffer，用于 updateStatus() 里 H2D token 写入
    // 避免每个 decode step 都 malloc（热路径优化）
    rtp_llm::BufferPtr                         pinned_token_buf;   // [1] int32, HOST pinned
    // 预分配 pinned HOST buffer，用于 D2H 状态回读
    rtp_llm::BufferPtr                         pinned_state_buf;   // [1] int32, HOST pinned

    StreamTreeInfo() = default;
    StreamTreeInfo(bool                                       in_tree_mode,
                   int32_t                                    input_length,
                   int32_t                                    output_length,
                   bool                                       is_beam_search,
                   std::shared_ptr<TreeDFA<std::string, int>> dfa_ptr,
                   bool                                       soft_constraint_mode = true):
        in_tree_mode(in_tree_mode),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        dfa_ptr(dfa_ptr),
        soft_constraint_mode(soft_constraint_mode),
        gpu_dfa_data(nullptr),
        current_state_id(-1) {}

    // 个性化树构造函数（直接传入 GPU 矩阵）
    StreamTreeInfo(bool                             in_tree_mode,
                   int32_t                          input_length,
                   int32_t                          output_length,
                   bool                             is_beam_search,
                   std::shared_ptr<TreeDFAGPUData>  gpu_dfa_data,
                   int32_t                          start_state_id,
                   bool                             soft_constraint_mode = true):
        in_tree_mode(in_tree_mode),
        input_length(input_length),
        current_output_length(output_length),
        is_beam_search(is_beam_search),
        dfa_ptr(nullptr),
        soft_constraint_mode(soft_constraint_mode),
        gpu_dfa_data(gpu_dfa_data),
        current_state_id(start_state_id) {}

    StreamTreeInfo copy() {
        StreamTreeInfo tree_info;
        tree_info.in_tree_mode          = in_tree_mode;
        tree_info.input_length          = input_length;
        tree_info.current_output_length = current_output_length;
        tree_info.is_beam_search        = is_beam_search;
        tree_info.soft_constraint_mode  = soft_constraint_mode;
        if (dfa_ptr) {
            // 全局树路径：深拷贝 DFA 对象（各 beam 独立状态）
            tree_info.dfa_ptr = std::make_shared<TreeDFA<std::string, int>>(*dfa_ptr);
        }
        if (gpu_dfa_data) {
            // 个性化树路径：GPU 矩阵只读，shared_ptr 引用计数 +1，不做深拷贝
            tree_info.gpu_dfa_data     = gpu_dfa_data;
            // CPU 状态直接复制当前値（每个 beam 独立进展）
            tree_info.current_state_id = current_state_id;
            // gpu_current_state_buf / pinned_token_buf / pinned_state_buf 不拷贝：
            // 每个 beam 需要独立的 GPU buffer，会在 fromGenerateInput 里预分配
            tree_info.gpu_current_state_buf = nullptr;
            tree_info.pinned_token_buf      = nullptr;
            tree_info.pinned_state_buf      = nullptr;
        }
        return tree_info;
    }
};

class TreeLogitsProcessor: public BaseLogitsProcessor {
public:
    TreeLogitsProcessor(rtp_llm::DeviceBase* device);
    TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos);
    virtual ~TreeLogitsProcessor() {}

public:
    static std::shared_ptr<TreeLogitsProcessor>
    fromGenerateInput(rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int32_t num);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) override;

public:
    std::vector<std::string> getStatus();
    size_t                   size() {
        return tree_infos_.size();
    }
    void insert(std::shared_ptr<TreeLogitsProcessor> others, size_t num) {
        if (others != nullptr) {
            tree_infos_.insert(tree_infos_.end(), others->tree_infos_.begin(), others->tree_infos_.end());
        }
    }

private:
    std::vector<StreamTreeInfo> tree_infos_;
};
typedef std::shared_ptr<TreeLogitsProcessor> TreeLogitsProcessorPtr;

}  // namespace rtp_llm