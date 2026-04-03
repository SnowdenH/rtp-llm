#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace std;

namespace rtp_llm {

// 将输入的叶子路径列表解析为 CSR 状态转移矩阵并上传到 GPU
//
// 输入格式：每条 path 为 "tok1_tok2_tok3" 形式的完整路径（不含根节点）
// 根节点 = start_token_id，其 state_id 固定为 0
//
// 示例：start=225, paths=["1_2_3","1_2_4","2_3_8"]
//   转换后的树：225->1->2->{3,4}; 225->2->3->8
//   CSR 根节点 key = "225"，中间节点 key = "225_1","225_1_2" 等自动推导
static std::shared_ptr<TreeDFAGPUData> buildCSRAndUpload(
    const std::vector<std::string>&      paths,
    int32_t                              start_token_id,
    int32_t                              end_token_id,
    rtp_llm::DeviceBase*                 device,
    bool                                 use_gpu_kernel)
{
    auto data            = std::make_shared<TreeDFAGPUData>();
    data->end_token_id   = end_token_id;
    data->use_gpu_kernel = use_gpu_kernel;

    const char SEP    = '_';
    std::string root_key = std::to_string(start_token_id);

    // Step 1: 从叶子路径推导所有节点，构建
    //   prefix_map[parent_key] -> sorted set of child token ids
    // 使用 std::map 保证遍历顺序确定，set 保证 col_idx 行内升序
    std::map<std::string, std::set<int32_t>> prefix_map;

    for (const auto& path : paths) {
        if (path.empty()) continue;
        // 解析 path -> token id 序列
        std::vector<int32_t> tokens;
        std::string seg;
        for (char c : path) {
            if (c == SEP) {
                if (!seg.empty()) { tokens.push_back(std::stoi(seg)); seg.clear(); }
            } else {
                seg += c;
            }
        }
        if (!seg.empty()) tokens.push_back(std::stoi(seg));
        if (tokens.empty()) continue;

        // 逐层建立 prefix_map
        std::string parent = root_key;
        for (int32_t tok : tokens) {
            prefix_map[parent].insert(tok);
            parent = parent + SEP + std::to_string(tok);
        }
        // 叶子节点本身不作为任何边的 parent，无需加入 prefix_map
    }

    if (prefix_map.empty()) {
        RTP_LLM_LOG_WARNING("buildCSRAndUpload: no valid paths provided");
        return nullptr;
    }

    // Step 2: 为所有出现过的 key 分配整数 state_id
    std::unordered_map<std::string, int32_t> key_to_id;
    key_to_id[root_key] = 0;  // 根节点固定为 state 0
    int32_t next_id = 1;
    for (const auto& kv : prefix_map) {
        if (key_to_id.find(kv.first) == key_to_id.end()) {
            key_to_id[kv.first] = next_id++;
        }
        for (int32_t tok : kv.second) {
            std::string child_key = kv.first + SEP + std::to_string(tok);
            if (key_to_id.find(child_key) == key_to_id.end()) {
                key_to_id[child_key] = next_id++;
            }
        }
    }
    int32_t num_states   = next_id;
    data->start_state_id = 0;

    // Step 3: 构建 CSR
    data->row_ptr.assign(num_states + 1, 0);

    // Step 3-1: 统计每个 state 的出边数
    for (const auto& kv : prefix_map) {
        int32_t state_id            = key_to_id.at(kv.first);
        data->row_ptr[state_id + 1] = static_cast<int32_t>(kv.second.size());
    }

    // Step 3-2: 前缀和
    for (int32_t i = 1; i <= num_states; ++i) {
        data->row_ptr[i] += data->row_ptr[i - 1];
    }
    int32_t nnz = data->row_ptr[num_states];
    data->col_idx.resize(nnz);
    data->next_state.resize(nnz);

    // Step 3-3: 填充 col_idx 和 next_state
    // col_idx 已通过 set 保证升序；同时记录层深用于计算 per_level_limit
    std::unordered_map<int32_t, int32_t> state_depth;  // state_id -> 层深
    state_depth[0] = 0;  // 根节点层深为 0
    int32_t max_depth = 0;

    std::vector<int32_t> fill_pos(data->row_ptr.begin(), data->row_ptr.begin() + num_states);
    for (const auto& kv : prefix_map) {
        int32_t state_id = key_to_id.at(kv.first);
        int32_t depth = static_cast<int32_t>(std::count(kv.first.begin(), kv.first.end(), SEP));
        state_depth[state_id] = depth;
        if (depth > max_depth) max_depth = depth;

        for (int32_t tok : kv.second) {
            int32_t pos           = fill_pos[state_id]++;
            data->col_idx[pos]    = tok;
            std::string child_key = kv.first + SEP + std::to_string(tok);
            auto nit              = key_to_id.find(child_key);
            data->next_state[pos] = (nit != key_to_id.end()) ? nit->second : -1;
        }
    }

    // Step 3-4: 计算 per_level_limit：每层最大分支数
    data->per_level_limit.assign(max_depth + 1, 0);
    for (int32_t s = 0; s < num_states; ++s) {
        int32_t row_width = data->row_ptr[s + 1] - data->row_ptr[s];
        if (row_width == 0) continue;
        auto dit = state_depth.find(s);
        if (dit == state_depth.end()) continue;
        int32_t d = dit->second;
        if (d <= max_depth && row_width > data->per_level_limit[d]) {
            data->per_level_limit[d] = row_width;
        }
    }

    // Step 3-5: 构建反向映射 id_to_key（用于 getStatus() 返回前缀字符串）
    for (const auto& kv : key_to_id) {
        data->id_to_key[kv.second] = kv.first;
    }

    // Step 4: H2D 上传
    auto upload = [&](const std::vector<int32_t>& vec) -> rtp_llm::BufferPtr {
        auto cpu_buf = rtp_llm::vector2Buffer(vec);
        return device->clone({*cpu_buf, rtp_llm::AllocationType::DEVICE});
    };
    data->gpu_row_ptr    = upload(data->row_ptr);
    data->gpu_col_idx    = upload(data->col_idx);
    data->gpu_next_state = upload(data->next_state);

    RTP_LLM_LOG_INFO("buildCSRAndUpload: num_paths=%zu, num_states=%d, nnz=%d, max_depth=%d, "
                     "use_gpu_kernel=%d, start_token_id=%d, end_token_id=%d",
                     paths.size(), num_states, nnz, max_depth,
                     (int)use_gpu_kernel, start_token_id, end_token_id);
    return data;
}

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {};

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos):
    BaseLogitsProcessor(device), tree_infos_(tree_infos) {}

void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    bool                             need_process  = false;
    bool                             need_cpu_mask = false;  // 是否有 beam 需要走 CPU maskLogits 路径
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    // GPU kernel 路径：收集需要处理的 beam 索引，批量调用
    // beam 索引在 tree_infos_ 中的位置（相对 start_idx）
    struct GPUBeamEntry {
        size_t local_idx;   // tree_infos_ 下标
        int32_t limit;      // 当前层 per_level_limit
    };
    std::vector<GPUBeamEntry> gpu_beams;

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }

        if (info.gpu_dfa_data != nullptr) {
            if (info.gpu_dfa_data->use_gpu_kernel) {
                // ── GPU kernel 路径 ────────────────────────────────────────────
                // 状态无效，退出树模式
                if (info.current_state_id < 0) {
                    info.in_tree_mode = false;
                    continue;
                }
                if (!info.gpu_current_state_buf) {
                    // 懒创建（理论上已在 fromGenerateInput 预分配，这里是 copy() 后 beam 的兜底）
                    auto init_buf = device_->allocateBuffer(
                        {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
                    init_buf->data<int32_t>()[0] = info.current_state_id;
                    info.gpu_current_state_buf = device_->clone(
                        {*init_buf, rtp_llm::AllocationType::DEVICE});
                }
                // 检查当前层的 limit
                const auto& d     = *info.gpu_dfa_data;
                int32_t     depth = info.current_output_length;
                int32_t     limit = (depth < (int32_t)d.per_level_limit.size())
                                        ? d.per_level_limit[depth] : 0;
                if (limit <= 0) {
                    info.in_tree_mode = false;
                    continue;
                }
                need_process = true;
                gpu_beams.push_back({i, limit});
            } else {
                // ── CPU 路径：从 CSR CPU 侧读取当前状态的候选 token ────────────
                const auto& d      = *info.gpu_dfa_data;
                int32_t     s      = info.current_state_id;
                if (s < 0 || s >= static_cast<int32_t>(d.row_ptr.size()) - 1) {
                    info.in_tree_mode = false;
                    continue;
                }
                std::vector<size_t> candidates;
                for (int32_t k = d.row_ptr[s]; k < d.row_ptr[s + 1]; ++k) {
                    candidates.push_back(static_cast<size_t>(d.col_idx[k]));
                }
                // 无候选 token 时（叶子节点）退出树模式
                if (candidates.empty()) {
                    info.in_tree_mode = false;
                    continue;
                }
                batch_candidate_token_ids[i] = std::move(candidates);
                need_process  = true;
                need_cpu_mask = true;
            }
        } else {
            // 全局树路径：现有逐字符串 DFA 逻辑不变
            // 》新增》如果即将到达终止状态，提前退出树模式
            if (info.dfa_ptr->isAboutToFinish()) {
                info.in_tree_mode = false;
                continue;
            }
            const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
            batch_candidate_token_ids[i]    = candidate_token_ids;
            if (candidate_token_ids.size() > 0) {
                need_process  = true;
                need_cpu_mask = true;
            }
        }
    }

    // 所有 beam 均无需处理，提前返回
    if (!need_process) {
        return;
    }

    // ── GPU kernel 批量调用 ────────────────────────────────────────────────────
    // 按 limit 分组，相同 limit 的 beam 合并成一次 kernel launch
    // （同一请求内各 beam 层相同，limit 必然一致；跨请求 limit 可能不同）
    if (!gpu_beams.empty()) {
        // 按 limit 分组
        std::sort(gpu_beams.begin(), gpu_beams.end(),
                  [](const GPUBeamEntry& a, const GPUBeamEntry& b) { return a.limit < b.limit; });

        size_t g = 0;
        while (g < gpu_beams.size()) {
            int32_t cur_limit = gpu_beams[g].limit;
            // 收集同 limit 的 beam 区间 [g, ge)
            size_t ge = g;
            while (ge < gpu_beams.size() && gpu_beams[ge].limit == cur_limit) ++ge;

            int32_t group_sz = static_cast<int32_t>(ge - g);

            if (group_sz == 1) {
                // 单 beam：直接调用，无需额外 buffer
                size_t li = gpu_beams[g].local_idx;
                auto&  info = tree_infos_[li];
                auto beam_logits = inputs.logits->slice(start_idx + li, 1);
                device_->csrMaskLogits(
                    *beam_logits,
                    *info.gpu_current_state_buf,
                    *info.gpu_dfa_data->gpu_row_ptr,
                    *info.gpu_dfa_data->gpu_col_idx,
                    cur_limit);
            } else {
                // 多 beam 同 limit：
                //   1. 将各 beam 的 gpu_current_state_buf 拼成临时 states[group_sz]
                //   2. 一次 batch kernel 处理所有 beam
                //   3. 结果直接写回各 beam 的 logits slice（kernel 按 blockIdx.x 寻址）
                //
                // 注：csrMaskLogits 的 logits 参数需要是连续内存中的多行
                //     inputs.logits 本身是连续的 [total_batch, vocab]，
                //     只需要确保 slice 起始地址和步长正确即可
                //
                // 方案：把 group_sz 个 beam 的 state 拷贝到一个临时 [group_sz] GPU buffer，
                //       然后对 logits 的对应行范围做一次 batch kernel。
                //       前提：group_sz 个 beam 在 logits 里的行号连续（同一请求 beam search 时成立）。
                //       若不连续则退化为逐 beam 调用。

                // 检查是否连续
                bool contiguous = true;
                for (size_t k = g + 1; k < ge; ++k) {
                    if (gpu_beams[k].local_idx != gpu_beams[k-1].local_idx + 1) {
                        contiguous = false;
                        break;
                    }
                }

                if (!contiguous) {
                    // 退化：逐 beam 调用
                    for (size_t k = g; k < ge; ++k) {
                        size_t li    = gpu_beams[k].local_idx;
                        auto&  info  = tree_infos_[li];
                        auto beam_logits = inputs.logits->slice(start_idx + li, 1);
                        device_->csrMaskLogits(
                            *beam_logits,
                            *info.gpu_current_state_buf,
                            *info.gpu_dfa_data->gpu_row_ptr,
                            *info.gpu_dfa_data->gpu_col_idx,
                            cur_limit);
                    }
                } else {
                    // 连续：构建 states[group_sz] CPU buffer → 上传 → 单次 batch kernel
                    auto cpu_states = device_->allocateBuffer(
                        {rtp_llm::DataType::TYPE_INT32,
                         {static_cast<size_t>(group_sz)},
                         rtp_llm::AllocationType::HOST}, {});
                    for (int32_t k = 0; k < group_sz; ++k) {
                        cpu_states->data<int32_t>()[k] =
                            tree_infos_[gpu_beams[g + k].local_idx].current_state_id;
                    }
                    auto gpu_states = device_->clone(
                        {*cpu_states, rtp_llm::AllocationType::DEVICE});

                    size_t first_li  = gpu_beams[g].local_idx;
                    auto group_logits = inputs.logits->slice(start_idx + first_li, group_sz);
                    const auto& d    = *tree_infos_[first_li].gpu_dfa_data;
                    device_->csrMaskLogits(
                        *group_logits,
                        *gpu_states,
                        *d.gpu_row_ptr,
                        *d.gpu_col_idx,
                        cur_limit);
                }
            }
            g = ge;
        }
    }

    // 只有 CPU 路径有候选 token 时，才需要调用 maskLogits
    // GPU kernel 路径已内联处理 logits，无需再次 mask
    if (!need_cpu_mask) {
        return;
    }

    auto   batch_logits     = inputs.logits->slice(start_idx, batch_size);
    size_t vocab_size       = batch_logits->shape()[1];
    auto   batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
}

void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        StreamTreeInfo info = tree_infos_[src_batch_idx].copy();
        // copy() 将三个 buffer 置 nullptr（每个 beam 需要独立 GPU buffer）
        // 在这里立即补分配，避免 decode 热路径里懒创建触发 malloc
        if (info.gpu_dfa_data && info.gpu_dfa_data->use_gpu_kernel) {
            // GPU state buffer：初始化为当前 CPU 侧状态 id
            auto init_buf = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
            init_buf->data<int32_t>()[0] = info.current_state_id;
            info.gpu_current_state_buf = device_->clone(
                {*init_buf, rtp_llm::AllocationType::DEVICE});
            // pinned buffers：用于热路径 H2D/D2H 复用
            info.pinned_token_buf = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
            info.pinned_state_buf = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
        }
        new_tree_infos.push_back(std::move(info));
    }
    tree_infos_ = std::move(new_tree_infos);
}

void TreeLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens->shape()[1]);
        }

        if (info.gpu_dfa_data != nullptr) {
            if (info.gpu_dfa_data->use_gpu_kernel) {
                // ── GPU kernel 路径：调用 csrUpdateStates kernel 推进状态 ────────────────
                if (!info.gpu_current_state_buf || info.current_state_id < 0) {
                    info.in_tree_mode = false;
                    info.current_output_length += num_new_tokens;
                    continue;
                }
                const auto& d = *info.gpu_dfa_data;

                // 确保 pinned buffer 已分配（首次走到这里时将创建）
                if (!info.pinned_token_buf) {
                    info.pinned_token_buf = device_->allocateBuffer(
                        {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
                }
                if (!info.pinned_state_buf) {
                    info.pinned_state_buf = device_->allocateBuffer(
                        {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
                }

                // 复用预分配的 pinned token buffer，避免每步 malloc
                auto& tok_buf = info.pinned_token_buf;
                for (size_t j = 0; j < num_new_tokens; ++j) {
                    auto token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
                    tok_buf->data<int32_t>()[0] = token_id;
                    // H2D: clone pinned → DEVICE
                    auto gpu_tok_buf = device_->clone({*tok_buf, rtp_llm::AllocationType::DEVICE});
                    // 调用 GPU kernel 更新状态
                    device_->csrUpdateStates(
                        *info.gpu_current_state_buf,
                        *gpu_tok_buf,
                        *d.gpu_row_ptr,
                        *d.gpu_col_idx,
                        *d.gpu_next_state,
                        d.end_token_id);
                    // D2H: 复用 pinned state buffer 回读状态
                    device_->copy({*info.pinned_state_buf, *info.gpu_current_state_buf});
                    info.current_state_id = info.pinned_state_buf->data<int32_t>()[0];
                    if (info.current_state_id < 0) {
                        // -1: 违反约束 / -2: 正常结束
                        info.in_tree_mode = false;
                        break;
                    }
                }
                info.current_output_length += num_new_tokens;
            } else {
                // ── CPU 路径：CSR 二分查找推进状态 ────────────────────────────────
                const auto& d = *info.gpu_dfa_data;
                for (size_t j = 0; j < num_new_tokens; ++j) {
                    auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
                    int32_t s = info.current_state_id;
                    if (s < 0 || s >= static_cast<int32_t>(d.row_ptr.size()) - 1) {
                        info.in_tree_mode = false;
                        break;
                    }
                    // 遇到结束 token，退出树模式
                    if (current_token_id == d.end_token_id) {
                        info.in_tree_mode = false;
                        break;
                    }
                    // 在 col_idx[row_ptr[s] .. row_ptr[s+1]) 范围内二分查找
                    auto begin_it = d.col_idx.begin() + d.row_ptr[s];
                    auto end_it   = d.col_idx.begin() + d.row_ptr[s + 1];
                    auto found    = std::lower_bound(begin_it, end_it, current_token_id);
                    if (found == end_it || *found != current_token_id) {
                        // token 不在候选集中：软约束退出，严格约束抛异常
                        if (info.soft_constraint_mode) {
                            RTP_LLM_LOG_WARNING("Personalized tree soft constraint: token %d not in candidates at state %d, "
                                               "beam %zu, exiting constraint decoding",
                                               current_token_id, s, i);
                            info.in_tree_mode = false;
                            break;
                        } else {
                            RTP_LLM_LOG_ERROR("Personalized tree strict constraint: token %d not in candidates at state %d, "
                                             "beam %zu",
                                             current_token_id, s, i);
                            throw std::runtime_error("Personalized tree: invalid token in strict mode");
                        }
                    }
                    int32_t edge_pos       = static_cast<int32_t>(found - d.col_idx.begin());
                    int32_t next_state     = d.next_state[edge_pos];
                    // -1 表示叶子节点，接受该 token 后退出树模式
                    if (next_state < 0) {
                        info.current_state_id = next_state;
                        info.in_tree_mode     = false;
                        break;
                    }
                    info.current_state_id = next_state;
                }
                info.current_output_length += num_new_tokens;
            }
        } else {
            // 全局单例 DFA 路径（原有逻辑不变）
            for (size_t j = 0; j < num_new_tokens; ++j) {
                auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
                if (info.soft_constraint_mode) {
                    if (!info.dfa_ptr->isValidNext(current_token_id)) {
                        RTP_LLM_LOG_WARNING("Soft constraint mode: Invalid token %d detected for beam %zu, "
                                          "exiting constraint decoding gracefully. Current status: %s",
                                          current_token_id, i, info.dfa_ptr->status().c_str());
                        info.in_tree_mode = false;
                        break;
                    }
                }
                try {
                    info.dfa_ptr->next(current_token_id);
                } catch (const std::runtime_error& e) {
                    if (info.soft_constraint_mode) {
                        RTP_LLM_LOG_WARNING("Soft constraint mode: Unexpected error processing token %d for beam %zu: %s",
                                          current_token_id, i, e.what());
                        info.in_tree_mode = false;
                        break;
                    } else {
                        throw;
                    }
                }
            }
            info.current_output_length += num_new_tokens;
            if (info.dfa_ptr->isFinished()) {
                info.in_tree_mode = false;
            }
        }
    }
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                                              std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    const auto& cfg = generate_input->generate_config;
    bool soft_constraint_enabled = cfg->soft_constraint_mode;
    bool is_beam_search = cfg->hasNumBeams() || cfg->num_return_sequences > 1;

    // --- 个性化树路径：请求中携带 custom_tree_paths ---
    if (cfg->custom_tree_paths.has_value() && !cfg->custom_tree_paths->empty()) {
        int32_t start_token_id = cfg->custom_tree_start_token_id.value_or(-1);
        int32_t end_token_id   = cfg->custom_tree_end_token_id.value_or(-1);
        if (start_token_id < 0 || end_token_id < 0) {
            RTP_LLM_LOG_WARNING("custom_tree_paths provided but start/end token ids are missing, skipping tree processor");
            return nullptr;
        }

        auto gpu_data = buildCSRAndUpload(*cfg->custom_tree_paths, start_token_id, end_token_id, device,
                                          cfg->use_gpu_csr_kernel);
        if (!gpu_data) {
            return nullptr;
        }

        auto processor_ptr = std::make_shared<TreeLogitsProcessor>(device);
        for (int32_t i = 0; i < num; i++) {
            StreamTreeInfo tree_info(/*in_tree_mode=*/true,
                                     generate_input->inputLength(),
                                     /*output_length=*/0,
                                     is_beam_search,
                                     gpu_data,
                                     gpu_data->start_state_id,
                                     soft_constraint_enabled);
            // GPU 路径下：预分配每个 beam 的 GPU/pinned buffer，避免 decode 热路径里 malloc
            if (gpu_data->use_gpu_kernel) {
                // GPU state buffer [1] int32，初始化为 start_state_id
                auto init_buf = device->allocateBuffer(
                    {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
                init_buf->data<int32_t>()[0] = gpu_data->start_state_id;
                tree_info.gpu_current_state_buf = device->clone(
                    {*init_buf, rtp_llm::AllocationType::DEVICE});
                // pinned token buffer [1] int32，用于 updateStatus() H2D token 写入
                tree_info.pinned_token_buf = device->allocateBuffer(
                    {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
                // pinned state buffer [1] int32，用于 D2H 状态回读
                tree_info.pinned_state_buf = device->allocateBuffer(
                    {rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
            }
            std::vector<StreamTreeInfo> tree_infos = {tree_info};
            auto single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);
            processor_ptr->insert(single_processor, 1);
        }
        RTP_LLM_LOG_INFO("TreeLogitsProcessor: personalized tree, num_streams=%d, soft_constraint=%d",
                         num, (int)soft_constraint_enabled);
        return processor_ptr;
    }

    // --- 全局单例路径：回退到 PrefixToCandidateTokens ---
    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    if (soft_constraint_enabled) {
        RTP_LLM_LOG_INFO("TreeLogitsProcessor: global tree, soft constraint enabled, num_streams=%d", num);
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 is_beam_search,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()),
                                 soft_constraint_enabled);
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);
        processor_ptr->insert(single_processor, 1);
    }
    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        if (tree_info.gpu_dfa_data != nullptr) {
            // 个性化树：尝试返回当前状态对应的前缀字符串
            int32_t sid = tree_info.current_state_id;
            const auto& id_map = tree_info.gpu_dfa_data->id_to_key;
            auto it = id_map.find(sid);
            if (it != id_map.end()) {
                status_list.push_back(it->second);
            } else {
                // 状态 id 无法映射到前缀时（如 -1/-2 返回已结束），返回带状态 id 的占位字符串
                status_list.push_back("csr_state:" + std::to_string(sid));
            }
        } else if (tree_info.dfa_ptr != nullptr) {
            status_list.push_back(tree_info.dfa_ptr->status());
        } else {
            status_list.push_back("no_tree");
        }
    }
    return status_list;
}

}  // namespace rtp_llm
