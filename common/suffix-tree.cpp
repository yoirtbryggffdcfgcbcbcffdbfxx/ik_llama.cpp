#include "suffix-tree.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

common_suffix_tree::common_suffix_tree(int max_depth)
    : _max_depth(max_depth)
    , _root(std::make_unique<common_suffix_node>())
{}

common_suffix_tree::~common_suffix_tree() = default;

void common_suffix_tree::clear() {
    _root = std::make_unique<common_suffix_node>();
    _tokens.clear();
    _n_inserted = 0;
}

void common_suffix_tree::extend(const llama_token * tokens, int n_tokens) {
    if (n_tokens <= 0) return;

    const int old_size = (int)_tokens.size();
    _tokens.insert(_tokens.end(), tokens, tokens + n_tokens);
    const int new_size = (int)_tokens.size();

    // Insert/update suffixes that are affected by the new tokens.
    // For any position i, the suffix covers tokens[i .. min(i+max_depth, end)].
    // Positions within max_depth of the old end had truncated suffixes that
    // can now be extended with new tokens.
    const int reinsert_from = std::max(0, old_size - _max_depth);

    for (int i = reinsert_from; i < new_size; ++i) {
        if (i < _n_inserted) {
            const int old_len = std::min(old_size - i, _max_depth);
            const int new_len = std::min(new_size - i, _max_depth);
            if (new_len > old_len) {
                _extend_suffix(i, old_len, new_len);
            }
        } else {
            _insert_suffix(i);
        }
    }

    _n_inserted = new_size;
}

void common_suffix_tree::_insert_suffix(int start_pos) {
    const int total = (int)_tokens.size();
    const int len = std::min(total - start_pos, _max_depth);
    if (len <= 0) return;

    common_suffix_node * node = _root.get();

    for (int i = 0; i < len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            auto child = std::make_unique<common_suffix_node>();
            auto * child_ptr = child.get();
            child_ptr->count = 1;
            node->children[tok] = std::move(child);
            node = child_ptr;
        } else {
            node = it->second.get();
            node->count++;
        }
    }
}

void common_suffix_tree::_extend_suffix(int start_pos, int old_len, int new_len) {
    common_suffix_node * node = _root.get();

    for (int i = 0; i < old_len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            return;
        }
        node = it->second.get();
    }

    for (int i = old_len; i < new_len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            auto child = std::make_unique<common_suffix_node>();
            auto * child_ptr = child.get();
            child_ptr->count = 1;
            node->children[tok] = std::move(child);
            node = child_ptr;
        } else {
            node = it->second.get();
            node->count++;
        }
    }
}

const common_suffix_node * common_suffix_tree::find_best_match_node(
        const llama_token * context, int n_context,
        int max_spec_tokens, float min_token_prob,
        int min_match_count, int min_match_len,
        std::vector<llama_token> * out_draft) const {

    if (!_root || n_context <= 0 || max_spec_tokens <= 0) {
        return nullptr;
    }

    if (n_context > _max_depth) {
        context += (n_context - _max_depth);
        n_context = _max_depth;
    }

    float best_score = 0.0f;
    const common_suffix_node * best_node = nullptr;

    for (int match_len = std::max(1, min_match_len); match_len <= n_context; ++match_len) {
        const llama_token * ctx = context + (n_context - match_len);

        const common_suffix_node * node = _root.get();
        bool matched = true;
        for (int i = 0; i < match_len; ++i) {
            auto it = node->children.find(ctx[i]);
            if (it == node->children.end()) {
                matched = false;
                break;
            }
            node = it->second.get();
        }

        if (!matched) break;
        if (node->count < min_match_count) continue;
        if (node->children.empty()) continue;

        // greedily follow the highest-count child; the probability decays multiplicatively
        const int draft_limit = std::min(max_spec_tokens, match_len + 8);

        std::vector<llama_token> draft;
        float score = 0.0f;
        float prob  = 1.0f;
        const common_suffix_node * cur = node;

        for (int i = 0; i < draft_limit; ++i) {
            if (cur->children.empty()) break;

            llama_token best_tok   = -1;
            int64_t     best_count = 0;
            for (const auto & [token, child] : cur->children) {
                if (child->count > best_count) {
                    best_count = child->count;
                    best_tok   = token;
                }
            }
            if (best_tok < 0 || cur->count <= 0) {
                break;
            }

            prob *= (float)best_count / (float)cur->count;
            if (prob < min_token_prob) break;

            score += prob;
            draft.push_back(best_tok);
            cur = cur->children.find(best_tok)->second.get();
        }

        if (score > best_score && !draft.empty()) {
            best_score = score;
            best_node  = node;
            if (out_draft != nullptr) {
                *out_draft = std::move(draft);
            }
        }
    }

    return best_node;
}

std::vector<llama_token> common_suffix_tree::speculate(
        const llama_token * context, int n_context,
        int max_spec_tokens,
        float min_token_prob,
        int   min_match_count,
        int   min_match_len) const {

    std::vector<llama_token> best_draft;
    find_best_match_node(context, n_context, max_spec_tokens, min_token_prob,
                         min_match_count, min_match_len, &best_draft);
    return best_draft;
}

std::vector<std::vector<llama_token>> common_suffix_tree::branch_candidates(
        const llama_token * context, int n_context,
        int max_branches, int depth,
        int max_spec_tokens,
        float min_token_prob,
        int   min_match_count,
        int   min_match_len,
        float competition_ratio) const {

    std::vector<std::vector<llama_token>> out;

    if (max_branches <= 0 || depth <= 0) {
        return out;
    }

    // same best-scoring match node that speculate() uses
    const common_suffix_node * best_node = find_best_match_node(
        context, n_context, max_spec_tokens, min_token_prob,
        min_match_count, min_match_len, nullptr);

    if (best_node == nullptr) {
        return out;
    }

    // rank the match node's children by frequency (desc)
    std::vector<std::pair<llama_token, int64_t>> ranked;
    ranked.reserve(best_node->children.size());
    for (const auto & [token, child] : best_node->children) {
        ranked.emplace_back(token, child->count);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    if (ranked.size() < 2 || ranked.front().second <= 0) {
        return out;
    }

    const int64_t top_count = ranked.front().second;

    // ranked[0] is the top child == the spine head; fork on the runners-up
    for (size_t r = 1; r < ranked.size() && (int) out.size() < max_branches; ++r) {
        const llama_token tok = ranked[r].first;
        const int64_t     cnt = ranked[r].second;

        // only branch on genuine competition with the top child
        if ((float) cnt < competition_ratio * (float) top_count) {
            break;
        }

        std::vector<llama_token> branch;
        branch.reserve(depth);
        branch.push_back(tok);

        // probability of the branch head relative to the match node
        float prob = best_node->count > 0 ? (float) cnt / (float) best_node->count : 0.0f;
        if (prob < min_token_prob) {
            break; // ranked is sorted descending: every remaining child is even less likely
        }

        const common_suffix_node * cur = best_node->children.at(tok).get();
        for (int i = 1; i < depth && !cur->children.empty(); ++i) {
            llama_token best_tok   = -1;
            int64_t     best_count = 0;
            for (const auto & [t, child] : cur->children) {
                if (child->count > best_count) {
                    best_count = child->count;
                    best_tok   = t;
                }
            }
            if (best_tok < 0) {
                break;
            }
            prob *= cur->count > 0 ? (float) best_count / (float) cur->count : 0.0f;
            if (prob < min_token_prob) {
                break;
            }
            branch.push_back(best_tok);
            const auto it = cur->children.find(best_tok);
            if (it == cur->children.end()) {
                break;
            }
            cur = it->second.get();
        }

        out.push_back(std::move(branch));
    }

    return out;
}

static void _extract_texts(const json & node, std::vector<std::string> & out) {
    if (node.is_string()) {
        const std::string s = node.get<std::string>();
        if (!s.empty()) out.push_back(s);
    } else if (node.is_array()) {
        for (const auto & item : node) {
            _extract_texts(item, out);
        }
    } else if (node.is_object()) {
        if (node.contains("content") && node["content"].is_string()) {
            const std::string s = node["content"].get<std::string>();
            if (!s.empty()) out.push_back(s);
        } else if (node.contains("messages")) {
            _extract_texts(node["messages"], out);
        }
    }
}

namespace {

constexpr size_t SUFFIX_CORPUS_BINARY_CHUNK_TOKENS = 1u << 15;
constexpr uint64_t SUFFIX_CORPUS_MAX_INSERT_WORK   = 256ull * 1024ull * 1024ull;

static uint64_t suffix_estimated_insert_work(size_t n_tokens, int max_depth) {
    return (uint64_t) n_tokens * (uint64_t) std::max(max_depth, 1);
}

static bool suffix_corpus_check_limit(const std::string & path, size_t n_tokens, int max_depth) {
    const uint64_t estimated_work = suffix_estimated_insert_work(n_tokens, max_depth);
    if (estimated_work <= SUFFIX_CORPUS_MAX_INSERT_WORK) {
        return true;
    }

    LOG_ERR("load_corpus: refusing suffix corpus '%s' - estimated insert work %llu exceeds limit %llu (tokens=%zu, depth=%d); reduce corpus size or lower suffix_max_depth inside --spec-type suffix:suffix_max_depth=...\n",
            path.c_str(),
            (unsigned long long) estimated_work,
            (unsigned long long) SUFFIX_CORPUS_MAX_INSERT_WORK,
            n_tokens,
            max_depth);
    return false;
}

static double suffix_elapsed_ms(const std::chrono::steady_clock::time_point & started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

bool common_suffix_tree::load_corpus(
        const std::string & path,
        std::function<std::vector<llama_token>(const std::string &)> tokenize_fn) {

    const auto load_started = std::chrono::steady_clock::now();

    bool is_json = path.size() >= 5 &&
                   path.compare(path.size() - 5, 5, ".json") == 0;

    if (is_json) {
        if (!tokenize_fn) {
            LOG_ERR("%s: JSON corpus requires a tokenizer but none was provided (path: '%s')\n",
                    __func__, path.c_str());
            return false;
        }
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERR("%s: failed to open corpus file '%s'\n", __func__, path.c_str());
            return false;
        }
        json root;
        try {
            f >> root;
        } catch (const json::exception & e) {
            LOG_ERR("%s: JSON parse error in '%s': %s\n", __func__, path.c_str(), e.what());
            return false;
        }
        std::vector<std::string> texts;
        _extract_texts(root, texts);
        if (texts.empty()) {
            LOG_WRN("%s: no text content found in corpus '%s'\n", __func__, path.c_str());
            return false;
        }

        LOG_INF("load_corpus: loading suffix JSON corpus '%s' (%zu texts, depth=%d)\n",
            path.c_str(), texts.size(), _max_depth);

        size_t total_tokens = 0;

        for (size_t i = 0; i < texts.size(); ++i) {
            const auto & text = texts[i];
            auto tokens = tokenize_fn(text);
            if (!tokens.empty()) {
                const size_t projected_tokens = total_tokens + tokens.size();
                if (!suffix_corpus_check_limit(path, projected_tokens, _max_depth)) {
                    clear();
                    return false;
                }

                extend(tokens.data(), (int) tokens.size());
                total_tokens = projected_tokens;
            }
        }

        if (total_tokens == 0) {
            LOG_WRN("%s: no tokens were extracted from suffix corpus '%s'\n",
                    __func__, path.c_str());
            clear();
            return false;
        }

        LOG_INF("load_corpus: done loading suffix JSON corpus '%s' - %zu texts, %zu tokens in %.1f ms\n",
            path.c_str(), texts.size(), total_tokens, suffix_elapsed_ms(load_started));
        return true;
    }

    // Binary format: raw int32 token IDs
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        LOG_ERR("%s: failed to open corpus file '%s'\n", __func__, path.c_str());
        return false;
    }

    size_t total_tokens_est = 0;
    if (std::fseek(fp, 0, SEEK_END) == 0) {
        const long file_size = std::ftell(fp);
        if (file_size >= 0) {
            total_tokens_est = (size_t) file_size / sizeof(int32_t);
            if ((size_t) file_size % sizeof(int32_t) != 0) {
                LOG_WRN("%s: suffix corpus '%s' has %zu trailing bytes; ignoring the remainder\n",
                        __func__, path.c_str(), (size_t) file_size % sizeof(int32_t));
            }
        }
        std::rewind(fp);
    }

    if (total_tokens_est > 0 && !suffix_corpus_check_limit(path, total_tokens_est, _max_depth)) {
        std::fclose(fp);
        return false;
    }

    LOG_INF("load_corpus: loading suffix binary corpus '%s' (%zu tokens, depth=%d)\n",
            path.c_str(), total_tokens_est, _max_depth);

    std::vector<int32_t> raw_tokens(SUFFIX_CORPUS_BINARY_CHUNK_TOKENS);
    std::vector<llama_token> tokens(SUFFIX_CORPUS_BINARY_CHUNK_TOKENS);

    size_t total_tokens = 0;

    while (true) {
        const size_t n_read = std::fread(raw_tokens.data(), sizeof(int32_t), raw_tokens.size(), fp);
        if (n_read == 0) {
            break;
        }

        const size_t projected_tokens = total_tokens + n_read;
        if (!suffix_corpus_check_limit(path, projected_tokens, _max_depth)) {
            std::fclose(fp);
            clear();
            return false;
        }

        for (size_t i = 0; i < n_read; ++i) {
            tokens[i] = raw_tokens[i];
        }

        extend(tokens.data(), (int) n_read);
        total_tokens = projected_tokens;
    }

    const bool read_error = std::ferror(fp) != 0;
    std::fclose(fp);

    if (read_error) {
        LOG_ERR("%s: read error while loading suffix corpus '%s'\n", __func__, path.c_str());
        clear();
        return false;
    }

    if (total_tokens == 0) {
        LOG_WRN("%s: suffix corpus file '%s' is empty\n", __func__, path.c_str());
        return false;
    }

    LOG_INF("load_corpus: done loading suffix binary corpus '%s' - %zu tokens in %.1f ms\n",
            path.c_str(), total_tokens, suffix_elapsed_ms(load_started));
    return true;
}
