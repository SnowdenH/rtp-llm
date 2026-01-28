#pragma once

#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <shared_mutex>  // 改用 shared_mutex 支持读写分离
#include <list>  // 用于 LRU 缓存
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class TreeDecodeConfig: public autil::legacy::Jsonizable {
public:
    int32_t                                     start_token_id;
    int32_t                                     end_token_id;
    std::string                                 sep;
    std::map<std::string, std::vector<int32_t>> prefix_dict;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("start_token_id", start_token_id, 225);
        json.Jsonize("end_token_id", end_token_id, 2);
        json.Jsonize("sep", sep, "_");
        json.Jsonize("prefix_dict", prefix_dict, prefix_dict);
    }
};

class PrefixToCandidateTokens {
public:
    // 添加 LRU 缓存结构
    struct LRUCacheEntry {
        std::string key;
        std::unordered_set<int32_t> value;
    };
    
    const std::unordered_set<int32_t>& getCandidateTokens(const std::string& key) {
        // 读操作使用共享锁，允许多个线程并发读取
        std::shared_lock<std::shared_mutex> lock(mutex_);
        static std::unordered_set<int32_t> EMPTY;
        if (!init_success_) {
            static std::unordered_set<int32_t> EMPTY;
            RTP_LLM_LOG_WARNING("PrefixToCandidateTokens is not initialized yet");
            return EMPTY;
        }
        
        // 先从 LRU 缓存查找
        auto cache_iter = lru_cache_map_.find(key);
        if (cache_iter != lru_cache_map_.end()) {
            // 命中缓存，更新 LRU 顺序
            lru_list_.splice(lru_list_.begin(), lru_list_, cache_iter->second);
            return cache_iter->second->value;
        }
        
        // 缓存未命中，从主数据结构查找
        auto iter = prefix_to_cadicates_.find(key);
        if (prefix_to_cadicates_.end() == iter) {
            return EMPTY;
        }
        
        // 添加到缓存
        addToCache(key, iter->second);
        
        return iter->second;
    }
    
    bool isValidStatus(const std::string& key) {
        // 读操作使用共享锁
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto iter = prefix_to_cadicates_.find(key);
        if (prefix_to_cadicates_.end() == iter) {
            return false;
        } else {
            return true;
        }
    }

    bool initSuccess() {
        return init_success_;
    }
    int32_t startTokenId() {
        return config.start_token_id;
    }
    int32_t endTokenId() {
        return config.end_token_id;
    }
    
    // 减少字符串分配，使用 reserve 预分配
    std::string generateNextKey(std::string old_key, int next) {
        // 读操作使用共享锁
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (!old_key.empty()) {
            // 预分配足够的空间，避免多次重分配
            size_t required_size = old_key.size() + config.sep.size() + 16; // 16 足够存储数字
            old_key.reserve(required_size);
            old_key += config.sep;
        }
        old_key += std::to_string(next);
        return old_key;
    }
    void reloadPrefixDictWithPrefix(std::string dir_path, std::string tree_decode_config) {
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load filepath : %s", tree_decode_config.c_str());
        if (tree_decode_config.size() > 0) {
            std::string prefix_dict_path = dir_path + "/" + tree_decode_config;
            reloadPrefixDict(prefix_dict_path);
        }
    }
    void reloadPrefixDict(std::string file_path) {
        loadPrefixDict(file_path);
    }

public:
    static std::shared_ptr<PrefixToCandidateTokens> instance() {
        static std::shared_ptr<PrefixToCandidateTokens> t(new PrefixToCandidateTokens());
        return t;
    }

private:
    PrefixToCandidateTokens() {}
    PrefixToCandidateTokens(PrefixToCandidateTokens&)                  = delete;
    PrefixToCandidateTokens(PrefixToCandidateTokens&&)                 = delete;
    PrefixToCandidateTokens& operator=(const PrefixToCandidateTokens&) = delete;
    
    // 添加到 LRU 缓存
    void addToCache(const std::string& key, const std::unordered_set<int32_t>& value) {
        // 如果缓存已满，移除最久未使用的项
        if (lru_cache_map_.size() >= MAX_CACHE_SIZE) {
            auto last = lru_list_.back();
            lru_cache_map_.erase(last.key);
            lru_list_.pop_back();
        }
        
        // 添加新项到缓存头部
        lru_list_.emplace_front(LRUCacheEntry{key, value});
        lru_cache_map_[key] = lru_list_.begin();
    }
    
    void loadPrefixDict(std::string file_path) {
        // 写操作使用独占锁
        std::unique_lock<std::shared_mutex> lock(mutex_);
        init_success_ = false;
        prefix_to_cadicates_.clear();
        
        // 清空缓存
        lru_list_.clear();
        lru_cache_map_.clear();
        
        std::ifstream file(file_path);
        if (!file) {
            std::stringstream ss;
            ss << "Unable to open file[" << file_path << "]" << std::endl;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: %s", ss.str().c_str());
            return;
        }

        try {
            std::ostringstream ss;
            ss << file.rdbuf();
            autil::legacy::FromJsonString(config, ss.str());
        } catch (autil::legacy::ExceptionBase& e) {
            std::stringstream ss;
            ss << "file[" << file_path << "]'s format is not json" << std::endl;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: %s", ss.str().c_str());
            return;
        }
        for (auto kv : config.prefix_dict) {
            std::unordered_set<int32_t> tmp_set;
            for (auto token_id : kv.second) {
                tmp_set.insert(token_id);
            }
            prefix_to_cadicates_[kv.first] = tmp_set;
        }
        file.close();
        init_success_ = true;
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully", file_path.c_str());
    }

private:
    // 使用 shared_mutex 替代 mutex
    std::shared_mutex                                            mutex_;
    TreeDecodeConfig                                             config;
    std::unordered_map<std::string, std::unordered_set<int32_t>> prefix_to_cadicates_;
    bool                                                         init_success_ = false;
    
    // LRU 缓存相关
    static constexpr size_t MAX_CACHE_SIZE = 1024;  // 缓存大小
    std::list<LRUCacheEntry>                                lru_list_;
    std::unordered_map<std::string, typename std::list<LRUCacheEntry>::iterator> lru_cache_map_;
};

typedef std::shared_ptr<PrefixToCandidateTokens> PrefixToCandidateTokensPtr;
}  // namespace rtp_llm