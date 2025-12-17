#include <unistd.h>     // pipe, fork, dup2, execl, _exit, close, STDIN_FILENO, STDOUT_FILENO
#include <signal.h>     // kill, SIGTERM
#include <sys/types.h>  // pid_t

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#error "This agent is intended for the Linux Docker environment."
#endif

namespace {

// --------------------------- Config ----------------------------

// Base parameters (can be overridden by dynamic tuning)
double BASE_RAVE_K = 200.0;
double BASE_C_PUCT = 0.8;
int BASE_VALUE_SWITCH_EMPTY = 45;

// Use dynamic parameter tuning based on game phase
bool USE_DYNAMIC_PARAMS = true;

// Use policy priors for expansion (recommended)
bool USE_POLICY_PRIORS = true;

// Max NN cache entries (simple eviction)
size_t NN_CACHE_MAX = 20000;

// Safety: if NN server fails, fall back to uniform policy + 0 value
bool NN_FAIL_OPEN = true;

// Opening book depth (first N moves use book)
int BOOK_DEPTH = 3;

// --------------------------- Types ----------------------------

struct Move {
    int row = -1;
    int col = -1;
    bool is_swap() const { return row == -1 && col == -1; }
    bool operator==(const Move& other) const { return row == other.row && col == other.col; }
};

enum class CellState { Empty, Red, Blue };
enum class Player { Red, Blue };

inline Player opposite(Player p) { return p == Player::Red ? Player::Blue : Player::Red; }
inline CellState cell_for(Player p) { return p == Player::Red ? CellState::Red : CellState::Blue; }

// --------------------------- Board ----------------------------

class HexBoard {
  public:
    HexBoard() = default;
    explicit HexBoard(int size) : size_(size), cells_(size * size, CellState::Empty) {}

    static HexBoard FromSerialized(const std::string& serialized) {
        std::vector<std::string> rows;
        std::string token;
        std::stringstream ss(serialized);
        while (std::getline(ss, token, ',')) rows.push_back(token);

        int board_size = static_cast<int>(rows.size());
        if (board_size <= 0) board_size = 11;

        HexBoard b(board_size);
        for (int r = 0; r < board_size; ++r) {
            if (r >= (int)rows.size()) break;
            const std::string& row_str = rows[r];
            for (int c = 0; c < board_size && c < (int)row_str.size(); ++c) {
                b.set(r, c, char_to_cell(row_str[c]));
            }
        }
        return b;
    }

    int size() const { return size_; }
    int n_cells() const { return size_ * size_; }

    CellState get(int r, int c) const { return cells_[idx(r, c)]; }
    void set(int r, int c, CellState s) { cells_[idx(r, c)] = s; }

    void apply_move(const Move& m, Player p) {
        if (!on_board(m.row, m.col)) return;
        set(m.row, m.col, cell_for(p));
    }

    std::vector<Move> legal_moves() const {
        std::vector<Move> ms;
        ms.reserve(size_ * size_);
        for (int r = 0; r < size_; ++r)
            for (int c = 0; c < size_; ++c)
                if (get(r, c) == CellState::Empty) ms.push_back({r, c});
        return ms;
    }

    int empty_count() const {
        int cnt = 0;
        for (auto cs : cells_) if (cs == CellState::Empty) ++cnt;
        return cnt;
    }

    bool has_winner(Player player) const {
        CellState target = cell_for(player);
        std::vector<char> visited(size_ * size_, 0);
        std::vector<int> st;
        st.reserve(size_ * size_);

        if (player == Player::Red) {
            for (int c = 0; c < size_; ++c) {
                if (get(0, c) == target) {
                    int id = idx(0, c);
                    visited[id] = 1;
                    st.push_back(id);
                }
            }
            while (!st.empty()) {
                int id = st.back();
                st.pop_back();
                int r = id / size_;
                if (r == size_ - 1) return true;
                for (int nb : neighbours(id)) {
                    if (!visited[nb] && cells_[nb] == target) {
                        visited[nb] = 1;
                        st.push_back(nb);
                    }
                }
            }
            return false;
        } else {
            for (int r = 0; r < size_; ++r) {
                if (get(r, 0) == target) {
                    int id = idx(r, 0);
                    visited[id] = 1;
                    st.push_back(id);
                }
            }
            while (!st.empty()) {
                int id = st.back();
                st.pop_back();
                int c = id % size_;
                if (c == size_ - 1) return true;
                for (int nb : neighbours(id)) {
                    if (!visited[nb] && cells_[nb] == target) {
                        visited[nb] = 1;
                        st.push_back(nb);
                    }
                }
            }
            return false;
        }
    }

    bool is_terminal() const { return has_winner(Player::Red) || has_winner(Player::Blue); }

    std::optional<Player> winner() const {
        if (has_winner(Player::Red)) return Player::Red;
        if (has_winner(Player::Blue)) return Player::Blue;
        return std::nullopt;
    }

    std::string serialize() const {
        // Same format as engine: rows separated by commas, each cell 0/R/B
        std::string out;
        out.reserve(n_cells() + size_);
        for (int r = 0; r < size_; ++r) {
            if (r) out.push_back(',');
            for (int c = 0; c < size_; ++c) {
                CellState s = get(r, c);
                out.push_back(s == CellState::Red ? 'R' : (s == CellState::Blue ? 'B' : '0'));
            }
        }
        return out;
    }

  private:
    static CellState char_to_cell(char c) {
        if (c == 'R') return CellState::Red;
        if (c == 'B') return CellState::Blue;
        return CellState::Empty;
    }

    bool on_board(int r, int c) const { return r >= 0 && c >= 0 && r < size_ && c < size_; }
    int idx(int r, int c) const { return r * size_ + c; }

    const std::vector<int>& neighbours(int id) const {
        static thread_local std::vector<int> res;
        res.clear();
        int r = id / size_;
        int c = id % size_;
        static constexpr int DR[6] = {0, 0, -1, -1, 1, 1};
        static constexpr int DC[6] = {-1, 1, 0, 1, -1, 0};
        for (int i = 0; i < 6; ++i) {
            int nr = r + DR[i], nc = c + DC[i];
            if (on_board(nr, nc)) res.push_back(idx(nr, nc));
        }
        return res;
    }

    int size_ = 11;
    std::vector<CellState> cells_;
};

// --------------------------- Opening Book ----------------------------

std::unordered_map<std::string, Move> opening_book;

void init_opening_book() {
    // Static initialization - only called once
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    
    const int N = 11;
    
    // Helper to create board string in serialize() format: "000,000,000,..."
    auto make_board_str = [N](const std::vector<std::string>& rows) {
        std::string result;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i > 0) result += ",";
            result += rows[i];
        }
        // Pad remaining rows with zeros
        while ((int)result.size() < (N * (N + 1) - 1)) {
            if (!result.empty()) result += ",";
            result += std::string(N, '0');
        }
        return result;
    };
    
    // First move: always center (empty board)
    std::vector<std::string> empty_rows(N, std::string(N, '0'));
    std::string empty_board = make_board_str(empty_rows);
    std::string key1 = empty_board;
    key1.erase(std::remove(key1.begin(), key1.end(), ','), key1.end());
    opening_book[key1 + "_R"] = {N/2, N/2};  // Red's first move
    
    // Second move: Blue's response after Red plays center (5,5)
    std::vector<std::string> after_center_rows = empty_rows;
    after_center_rows[5][5] = 'R';
    std::string after_center = make_board_str(after_center_rows);
    std::string key2 = after_center;
    key2.erase(std::remove(key2.begin(), key2.end(), ','), key2.end());
    opening_book[key2 + "_B"] = {5, 4};   // Adjacent to center
    
    // Third move: Red's response after Red(5,5) -> Blue(5,4)
    std::vector<std::string> after_second_rows = after_center_rows;
    after_second_rows[5][4] = 'B';
    std::string after_second = make_board_str(after_second_rows);
    std::string key3 = after_second;
    key3.erase(std::remove(key3.begin(), key3.end(), ','), key3.end());
    opening_book[key3 + "_R"] = {5, 6};  // Continue connection
    
    // Alternative: After Red(5,5) -> Blue(4,5)
    std::vector<std::string> alt_second_rows = after_center_rows;
    alt_second_rows[4][5] = 'B';
    std::string alt_second = make_board_str(alt_second_rows);
    std::string key4 = alt_second;
    key4.erase(std::remove(key4.begin(), key4.end(), ','), key4.end());
    opening_book[key4 + "_R"] = {6, 5};
}

std::optional<Move> get_book_move(const HexBoard& board, Player to_move, int empty_count) {
    // Only use book for first few moves
    const int N = board.size();
    const int total_cells = N * N;
    
    if (empty_count < total_cells - BOOK_DEPTH) {
        return std::nullopt;  // Too deep, don't use book
    }
    
    init_opening_book();
    
    // Create key: board state + player
    std::string key = board.serialize();
    // Remove commas for key (more compact)
    key.erase(std::remove(key.begin(), key.end(), ','), key.end());
    key += "_";
    key += (to_move == Player::Red ? "R" : "B");
    
    auto it = opening_book.find(key);
    if (it != opening_book.end()) {
        // Verify the move is still legal
        Move m = it->second;
        if (board.get(m.row, m.col) == CellState::Empty) {
            return m;
        }
    }
    
    return std::nullopt;
}

// -------------------- Engine stdin protocol --------------------

struct CommandData {
    std::string command;
    bool has_move = false;
    Move move;
    std::string board_string;
    int turn = 1;
};

CommandData parse_command(const std::string& line) {
    CommandData data;
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start < line.size()) {
        size_t pos = line.find(';', start);
        if (pos == std::string::npos) {
            if (start < line.size()) tokens.emplace_back(line.substr(start));
            break;
        }
        tokens.emplace_back(line.substr(start, pos - start));
        start = pos + 1;
    }

    if (tokens.size() >= 3) {
        data.command = tokens[0];
        data.board_string = tokens[2];

        if (!tokens[1].empty()) {
            size_t comma = tokens[1].find(',');
            if (comma != std::string::npos) {
                try {
                    data.has_move = true;
                    data.move.row = std::stoi(tokens[1].substr(0, comma));
                    data.move.col = std::stoi(tokens[1].substr(comma + 1));
                } catch (...) { data.has_move = false; }
            }
        }
        if (tokens.size() > 3 && !tokens[3].empty()) {
            try { data.turn = std::stoi(tokens[3]); } catch (...) {}
        }
    }
    return data;
}

// ------------------------- NN Server IPC ------------------------

struct NNResult {
    // policy over N*N (not including swap)
    std::vector<float> policy;
    float value = 0.0f; // in [-1,+1], from perspective of `to_move`
};

class NNPipe {
  public:
    NNPipe() = default;
    ~NNPipe() { close(); }

    bool start(const std::string& python_exe,
               const std::string& nn_server_path,
               const std::string& model_path,
               int board_size) {
        close();
        board_size_ = board_size;
        n_ = board_size * board_size;

        // Command: python nn_server.py --board 11 --model weights.pt
        std::string cmd = python_exe + " " + nn_server_path +
                          " --board " + std::to_string(board_size) +
                          " --model " + model_path;

        // popen gives one-way; we need bi-directional -> use two pipes via fork/exec.
        // For simplicity and robustness in Docker, we implement a minimal fork/exec pipe.

        int in_pipe[2];   // parent writes -> child reads (stdin)
        int out_pipe[2];  // child writes -> parent reads (stdout)
        if (pipe(in_pipe) != 0) return false;
        if (pipe(out_pipe) != 0) return false;

        pid_ = fork();
        if (pid_ == -1) return false;

        if (pid_ == 0) {
            // child
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            // close fds
            close_fd(in_pipe[1]); close_fd(out_pipe[0]);
            close_fd(in_pipe[0]); close_fd(out_pipe[1]);

            // exec via /bin/sh -lc "..."
            execl("/bin/sh", "sh", "-lc", cmd.c_str(), (char*)nullptr);
            _exit(127);
        }

        // parent
        close_fd(in_pipe[0]);
        close_fd(out_pipe[1]);
        child_stdin_ = fdopen(in_pipe[1], "w");
        child_stdout_ = fdopen(out_pipe[0], "r");
        if (!child_stdin_ || !child_stdout_) return false;

        setvbuf(child_stdin_, nullptr, _IONBF, 0);
        setvbuf(child_stdout_, nullptr, _IONBF, 0);

        // simple "PING" handshake
        if (!send_line_("PING")) return false;
        std::string resp;
        if (!read_line_(resp)) return false;
        return resp == "PONG";
    }

    bool ok() const { return pid_ > 0 && child_stdin_ && child_stdout_; }

    NNResult eval(const HexBoard& state, Player to_move) {
        NNResult r;
        r.policy.assign(n_, 1.0f / std::max(1, n_));
        r.value = 0.0f;

        if (!ok()) return r;

        // Protocol:
        // EVAL;<to_move_char>;<board_string>
        // Response:
        // OK;<value>;<p0>,<p1>,...,<pN-1>
        // or ERR;<message>
        char tm = (to_move == Player::Red) ? 'R' : 'B';
        std::string msg = "EVAL;";
        msg.push_back(tm);
        msg.push_back(';');
        msg += state.serialize();

        if (!send_line_(msg)) return r;

        std::string line;
        if (!read_line_(line)) return r;

        if (line.rfind("OK;", 0) != 0) {
            if (!NN_FAIL_OPEN) {
                // hard-fail: no response -> likely lose anyway
            }
            return r;
        }

        // Parse OK;value;csv...
        // Minimal parsing for speed.
        size_t p1 = line.find(';');          // after OK
        size_t p2 = line.find(';', p1 + 1);  // after value
        if (p2 == std::string::npos) return r;

        try {
            r.value = std::stof(line.substr(p1 + 1, p2 - (p1 + 1)));
        } catch (...) {}

        // Parse policy CSV
        r.policy.assign(n_, 0.0f);
        const char* s = line.c_str() + p2 + 1;
        int i = 0;
        while (*s && i < n_) {
            char* endp = nullptr;
            float v = std::strtof(s, &endp);
            if (endp == s) break;
            r.policy[i++] = v;
            s = endp;
            if (*s == ',') ++s;
        }
        // Normalize (legal masking happens elsewhere)
        return r;
    }

    void close() {
        if (child_stdin_) { fclose(child_stdin_); child_stdin_ = nullptr; }
        if (child_stdout_) { fclose(child_stdout_); child_stdout_ = nullptr; }
        if (pid_ > 0) {
            // best effort terminate
            kill(pid_, SIGTERM);
            pid_ = -1;
        }
    }

  private:
    static void close_fd(int fd) { if (fd >= 0) ::close(fd); }

    bool send_line_(const std::string& s) {
        if (!child_stdin_) return false;
        std::fwrite(s.c_str(), 1, s.size(), child_stdin_);
        std::fwrite("\n", 1, 1, child_stdin_);
        std::fflush(child_stdin_);
        return true;
    }

    bool read_line_(std::string& out) {
        out.clear();
        if (!child_stdout_) return false;
        char* lineptr = nullptr;
        size_t n = 0;
        ssize_t len = getline(&lineptr, &n, child_stdout_);
        if (len <= 0) {
            if (lineptr) free(lineptr);
            return false;
        }
        // trim newline
        while (len > 0 && (lineptr[len - 1] == '\n' || lineptr[len - 1] == '\r')) {
            lineptr[len - 1] = '\0';
            --len;
        }
        out.assign(lineptr, (size_t)len);
        free(lineptr);
        return true;
    }

    pid_t pid_ = -1;
    FILE* child_stdin_ = nullptr;
    FILE* child_stdout_ = nullptr;
    int board_size_ = 11;
    int n_ = 121;
};

// ------------------------- MCTS Node ---------------------------

struct ChildEdge {
    std::unique_ptr<struct Node> node;
    float prior = 0.0f; // P(s,a)
};

struct Node {
    Move move;
    Player player_just_moved;
    HexBoard state;
    Node* parent = nullptr;

    int visits = 0;
    double value_sum = 0.0;  // value in [-1,+1] from ROOT player's perspective

    // Optional RAVE stats (AMAF): also from ROOT player's perspective
    int amaf_visits = 0;
    double amaf_value_sum = 0.0;

    std::vector<Move> untried_moves;   // moves not expanded yet
    std::vector<float> untried_priors; // same length, priors aligned
    std::vector<ChildEdge> children;

    Node(const HexBoard& s, Player pjm, Node* par, const Move& m)
        : move(m), player_just_moved(pjm), state(s), parent(par) {}
};

// -------------------------- Dynamic Parameter Tuning --------------------------

struct DynamicParams {
    double rave_k;
    double c_puct;
    int value_switch_empty;
};

DynamicParams compute_dynamic_params(int empty_count, int board_size) {
    DynamicParams params;
    
    if (!USE_DYNAMIC_PARAMS) {
        // Use base parameters
        params.rave_k = BASE_RAVE_K;
        params.c_puct = BASE_C_PUCT;
        params.value_switch_empty = BASE_VALUE_SWITCH_EMPTY;
        return params;
    }
    
    const int total_cells = board_size * board_size;
    const double game_progress = 1.0 - (double)empty_count / total_cells;  // 0.0 (start) to 1.0 (end)
    
    // RAVE_K: Higher in early game (trust RAVE), lower in late game (trust UCT)
    // Early (0-30%): 300, Mid (30-70%): 200, Late (70-100%): 100
    if (game_progress < 0.3) {
        params.rave_k = 300.0;  // Early: use RAVE more
    } else if (game_progress < 0.7) {
        params.rave_k = 200.0;  // Mid: default value
    } else {
        params.rave_k = 100.0;  // Late: trust UCT more
    }
    
    // C_PUCT: Higher in early game (exploration), lower in late game (exploitation)
    // Early (0-40%): 1.0, Mid (40-80%): 0.8, Late (80-100%): 0.6
    if (game_progress < 0.4) {
        params.c_puct = 1.0;   // Early: more exploration
    } else if (game_progress < 0.8) {
        params.c_puct = 0.8;   // Mid: default value
    } else {
        params.c_puct = 0.6;   // Late: stronger exploitation
    }
    
    // VALUE_SWITCH_EMPTY: Adjust based on game phase
    // Use rollout more in early game, NN value more in late game
    if (game_progress < 0.3) {
        params.value_switch_empty = 40;  // Early: use rollout more
    } else if (game_progress < 0.6) {
        params.value_switch_empty = 45;  // Mid: default value
    } else {
        params.value_switch_empty = 50;  // Late: use NN value more
    }
    
    return params;
}

// -------------------------- MCTS Core --------------------------

class MCTS {
  public:
    MCTS(std::mt19937& rng, NNPipe* nn) : rng_(rng), nn_(nn) {}

    Move search(const HexBoard& root_state, Player to_move, int time_limit_ms) {
        const int N = root_state.size();
        auto legal = root_state.legal_moves();
        if (legal.empty()) return {-1, -1};

        // Try opening book first
        int empty_count = root_state.empty_count();
        auto book_move = get_book_move(root_state, to_move, empty_count);
        if (book_move.has_value()) {
            Move m = book_move.value();
            // Verify move is legal (safety check)
            if (root_state.get(m.row, m.col) == CellState::Empty) {
                return m;
            }
        }
        
        // Fallback: quick opening center (if book doesn't have it)
        if (empty_count == N * N) return {N/2, N/2};

        // Compute dynamic parameters based on game phase
        current_params_ = compute_dynamic_params(empty_count, N);

        root_player_ = to_move;
        Node root(root_state, opposite(to_move), nullptr, Move{});

        // init root priors
        init_untried_with_priors(root, to_move);

        auto t0 = std::chrono::steady_clock::now();
        auto budget = std::chrono::milliseconds(time_limit_ms);

        int iters = 0;
        while (true) {
            run_iteration(root, to_move);
            ++iters;
            if ((iters & 255) == 0) {
                if (std::chrono::steady_clock::now() - t0 >= budget) break;
            }
        }

        // pick child with most visits
        Node* best = nullptr;
        int best_v = -1;
        for (auto& ch : root.children) {
            if (ch.node && ch.node->visits > best_v) {
                best_v = ch.node->visits;
                best = ch.node.get();
            }
        }
        if (!best) return legal.front();
        return best->move;
    }

  private:
    // Cache NN results by serialized board + to_move char
    struct CacheEntry {
        NNResult res;
        uint64_t tick = 0;
    };
    std::unordered_map<std::string, CacheEntry> nn_cache_;
    uint64_t tick_ = 0;

    NNResult nn_eval_cached(const HexBoard& s, Player to_move) {
        ++tick_;
        std::string key;
        key.reserve(s.n_cells() + s.size() + 4);
        key.push_back(to_move == Player::Red ? 'R' : 'B');
        key.push_back('|');
        key += s.serialize();

        auto it = nn_cache_.find(key);
        if (it != nn_cache_.end()) {
            it->second.tick = tick_;
            return it->second.res;
        }

        NNResult res = nn_ ? nn_->eval(s, to_move) : NNResult{};
        nn_cache_[key] = CacheEntry{res, tick_};

        // crude eviction
        if (nn_cache_.size() > NN_CACHE_MAX) {
            // remove ~10% oldest
            std::vector<std::pair<uint64_t, std::string>> age;
            age.reserve(nn_cache_.size());
            for (auto& kv : nn_cache_) age.push_back({kv.second.tick, kv.first});
            std::nth_element(age.begin(), age.begin() + age.size()/10, age.end(),
                             [](auto& a, auto& b){ return a.first < b.first; });
            size_t cut = age.size()/10;
            for (size_t i = 0; i < cut; ++i) nn_cache_.erase(age[i].second);
        }

        return res;
    }

    void init_untried_with_priors(Node& node, Player to_move) {
        node.untried_moves = node.state.legal_moves();
        const int N = node.state.size();
        const int n = N*N;

        node.untried_priors.assign(node.untried_moves.size(), 1.0f);

        if (!USE_POLICY_PRIORS || !nn_) {
            // uniform
            float uni = 1.0f / std::max(1, (int)node.untried_moves.size());
            std::fill(node.untried_priors.begin(), node.untried_priors.end(), uni);
            return;
        }

        NNResult r = nn_eval_cached(node.state, to_move);
        // mask + renorm
        float sum = 0.0f;
        for (size_t i = 0; i < node.untried_moves.size(); ++i) {
            const Move& m = node.untried_moves[i];
            int idx = m.row * N + m.col;
            float p = (idx >= 0 && idx < n) ? r.policy[idx] : 0.0f;
            if (!(p > 0)) p = 0.0f;
            node.untried_priors[i] = p;
            sum += p;
        }
        if (sum <= 1e-12f) {
            float uni = 1.0f / std::max(1, (int)node.untried_moves.size());
            std::fill(node.untried_priors.begin(), node.untried_priors.end(), uni);
        } else {
            for (auto& p : node.untried_priors) p /= sum;
        }

        // sort moves by prior descending to expand best first
        std::vector<size_t> order(node.untried_moves.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b){
            return node.untried_priors[a] > node.untried_priors[b];
        });

        auto moves = node.untried_moves;
        auto priors = node.untried_priors;
        for (size_t i = 0; i < order.size(); ++i) {
            node.untried_moves[i] = moves[order[i]];
            node.untried_priors[i] = priors[order[i]];
        }
    }

    Node* select_child(Node& node) {
        // PUCT + optional RAVE blend
        Node* best = nullptr;
        double best_score = -1e100;

        double sqrt_parent = std::sqrt(std::max(1, node.visits));
        const double log_parent = std::log(std::max(1, node.visits));

        for (auto& edge : node.children) {
            Node* ch = edge.node.get();
            if (!ch) continue;

            // Q from root perspective
            double Q = (ch->visits > 0) ? (ch->value_sum / ch->visits) : 0.0;

            // U (PUCT) - use dynamic parameter
            double c_puct = current_params_.c_puct;
            double U = c_puct * (double)edge.prior * (sqrt_parent / (1.0 + ch->visits));

            double score = Q + U;

            // Optional: blend with AMAF in early game - use dynamic parameter
            double rave_k = current_params_.rave_k;
            if (rave_k > 0.0) {
                double amaf = (ch->amaf_visits > 0) ? (ch->amaf_value_sum / ch->amaf_visits) : 0.0;
                double beta = std::sqrt(rave_k / (3.0 * std::max(1, ch->visits) + rave_k));
                // Blend only the Q part (not the U term)
                score = (1.0 - beta) * (Q + U) + beta * (amaf + U);
            }

            // tiny UCT bias if you want (optional)
            // score += 0.01 * std::sqrt(log_parent / (1.0 + ch->visits));

            if (score > best_score) {
                best_score = score;
                best = ch;
            }
        }
        return best;
    }

    void run_iteration(Node& root, Player to_move) {
        Node* node = &root;
        Player current = to_move;

        const int N = root.state.size();
        const int n = N*N;

        // AMAF bookkeeping (for RAVE)
        std::vector<char> played_by_red(n, 0), played_by_blue(n, 0);

        // Selection
        while (!node->state.is_terminal() && node->untried_moves.empty() && !node->children.empty()) {
            Node* next = select_child(*node);
            if (!next) break;
            node = next;

            if (!node->move.is_swap()) {
                int idx = node->move.row * N + node->move.col;
                if (idx >= 0 && idx < n) {
                    Player who = node->player_just_moved;
                    if (who == Player::Red) played_by_red[idx] = 1;
                    else played_by_blue[idx] = 1;
                }
            }
            current = opposite(node->player_just_moved);
        }

        // Expansion
        if (!node->state.is_terminal() && !node->untried_moves.empty()) {
            Move m = node->untried_moves.front();
            float prior = node->untried_priors.front();

            node->untried_moves.erase(node->untried_moves.begin());
            node->untried_priors.erase(node->untried_priors.begin());

            HexBoard ns = node->state;
            ns.apply_move(m, current);

            auto child = std::make_unique<Node>(ns, current, node, m);
            // init priors for child (need next player's turn)
            init_untried_with_priors(*child, opposite(current));

            node->children.push_back(ChildEdge{std::move(child), prior});
            node = node->children.back().node.get();

            int idx = m.row * N + m.col;
            if (idx >= 0 && idx < n) {
                if (current == Player::Red) played_by_red[idx] = 1;
                else played_by_blue[idx] = 1;
            }

            current = opposite(current);
        }

        // Evaluation: either rollout or NN value depending on phase
        double v_root = 0.0; // value from ROOT player's perspective

        if (node->state.is_terminal()) {
            auto w = node->state.winner();
            if (!w.has_value()) v_root = 0.0;
            else v_root = (w.value() == root_player_) ? 1.0 : -1.0;
        } else {
            int empty = node->state.empty_count();
            // Use dynamic parameter for value switch threshold
            int value_switch_empty = current_params_.value_switch_empty;
            if (empty <= value_switch_empty && nn_) {
                // value net: returned from perspective of current player-to-move at this leaf
                NNResult r = nn_eval_cached(node->state, current);
                double v_leaf_pov = (double)r.value; // leaf player's perspective
                // Convert to root perspective:
                // if current == root_player -> same sign, else flip
                v_root = (current == root_player_) ? v_leaf_pov : -v_leaf_pov;
            } else {
                // rollout
                v_root = rollout_value(node->state, current, played_by_red, played_by_blue);
            }
        }

        // Backprop (value from root perspective)
        Node* bp = node;
        while (bp) {
            bp->visits += 1;
            bp->value_sum += v_root;

            // RAVE update: update children edges based on whether their move appeared in playout
            // Use dynamic parameter
            double rave_k = current_params_.rave_k;
            if (rave_k > 0.0) {
                for (auto& edge : bp->children) {
                    Node* ch = edge.node.get();
                    if (!ch || ch->move.is_swap()) continue;
                    int idx = ch->move.row * N + ch->move.col;
                    if (idx < 0 || idx >= n) continue;

                    // AMAF: move counted for the player who played that child move
                    Player who = ch->player_just_moved;
                    bool played = (who == Player::Red) ? played_by_red[idx] : played_by_blue[idx];
                    if (played) {
                        ch->amaf_visits += 1;
                        ch->amaf_value_sum += v_root;
                    }
                }
            }

            bp = bp->parent;
        }
    }

    double rollout_value(HexBoard state, Player current,
                         std::vector<char>& played_by_red,
                         std::vector<char>& played_by_blue) {
        // Random rollout; return value from ROOT player's perspective
        auto moves = state.legal_moves();
        while (!state.is_terminal() && !moves.empty()) {
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            size_t k = dist(rng_);
            Move m = moves[k];
            moves[k] = moves.back();
            moves.pop_back();

            int N = state.size();
            int idx = m.row * N + m.col;
            if (idx >= 0 && idx < (int)played_by_red.size()) {
                if (current == Player::Red) played_by_red[idx] = 1;
                else played_by_blue[idx] = 1;
            }

            state.apply_move(m, current);
            current = opposite(current);
        }

        auto w = state.winner();
        if (!w.has_value()) return 0.0;
        return (w.value() == root_player_) ? 1.0 : -1.0;
    }

    std::mt19937& rng_;
    NNPipe* nn_ = nullptr;
    Player root_player_ = Player::Red;
    DynamicParams current_params_;  // Current dynamic parameters for this search
};

// ------------------------- Agent wrapper ------------------------

class HexHybridAgent {
  public:
    HexHybridAgent(Player colour, int board_size,
                   const std::string& python_exe,
                   const std::string& nn_server_path,
                   const std::string& model_path)
        : my_colour_(colour), initial_colour_(colour), board_size_(board_size),
          swap_available_(colour == Player::Blue) {
        unsigned seed = (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng_ = std::mt19937(seed);

        // Start NN server
        if (!python_exe.empty() && !nn_server_path.empty() && !model_path.empty()) {
            nnpipe_ = std::make_unique<NNPipe>();
            bool ok = nnpipe_->start(python_exe, nn_server_path, model_path, board_size_);
            if (!ok) {
                std::cerr << "[WARN] Failed to start NN server. Falling back.\n";
                nnpipe_.reset();
            }
        }

        mcts_ = std::make_unique<MCTS>(rng_, nnpipe_.get());
    }

    Move handle(const CommandData& data) {
        if (data.command == "SWAP") my_colour_ = opposite(my_colour_);

        HexBoard board = HexBoard::FromSerialized(data.board_string);
        board_size_ = board.size();

        if (can_swap_now(data)) {
            if (data.has_move && should_swap(data.move, board_size_)) {
                my_colour_ = opposite(my_colour_);
                swap_available_ = false;
                return {-1, -1};
            }
            swap_available_ = false;
        }

        int budget = compute_time_budget(data.turn, board.empty_count());
        Move m = mcts_->search(board, my_colour_, budget);
        swap_available_ = false;
        return m;
    }

  private:
    bool can_swap_now(const CommandData& d) const {
        return swap_available_ && initial_colour_ == Player::Blue && d.turn == 2 && d.command == "CHANGE";
    }

    bool should_swap(const Move& opp_opening, int size) const {
        int centre = size / 2;
        int dr = opp_opening.row - centre;
        int dc = opp_opening.col - centre;
        int dist2 = dr*dr + dc*dc;
        // return dist2 <= std::max(2, size);
        return dist2 <= 4;
    }

    int compute_time_budget(int /*turn*/, int empty) const {
        // You should time-manage for 5 minutes total. This is a simple heuristic.
        if (empty > 90) return 2500;
        if (empty > 70) return 2000;
        if (empty > 50) return 1600;
        if (empty > 30) return 1200;
        return 900;
    }

    Player my_colour_;
    Player initial_colour_;
    int board_size_;
    bool swap_available_ = false;

    std::mt19937 rng_;
    std::unique_ptr<NNPipe> nnpipe_;
    std::unique_ptr<MCTS> mcts_;
};

} // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 3) return 1;
    Player colour = (argv[1][0] == 'R') ? Player::Red : Player::Blue;
    int board_size = std::stoi(argv[2]);

    // Args:
    // 1: R|B
    // 2: board_size
    // 3: RAVE_K (optional)
    // 4: python exe (optional)
    // 5: nn_server.py path (optional)
    // 6: weights/model path (optional)
    // 7: VALUE_SWITCH_EMPTY (optional)
    // 8: C_PUCT (optional)

    if (argc >= 4) { try { BASE_RAVE_K = std::stod(argv[3]); } catch (...) {} }
    std::string pyexe, server_path, model_path;
    if (argc >= 7) {
        pyexe = argv[4];
        server_path = argv[5];
        model_path = argv[6];
    }
    if (argc >= 8) { try { BASE_VALUE_SWITCH_EMPTY = std::stoi(argv[7]); } catch (...) {} }
    if (argc >= 9) { try { BASE_C_PUCT = std::stod(argv[8]); } catch (...) {} }

    std::cerr << ">>> Hybrid MCTS (PUCT+optional RAVE, rollout early, value late) <<<\n";
    if (!pyexe.empty()) std::cerr << "NN enabled via: " << pyexe << " " << server_path << "\n";

    HexHybridAgent agent(colour, board_size, pyexe, server_path, model_path);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line.back() != ';') continue;
        CommandData d = parse_command(line);
        Move m = agent.handle(d);
        std::cout << m.row << "," << m.col << "\n";
        std::cout.flush();
    }
    return 0;
}
