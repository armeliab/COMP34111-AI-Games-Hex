#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

// [IMPORTANT] Use row, col instead of x, y to avoid confusion
struct Move {
    int row = -1;
    int col = -1;

    bool is_swap() const { return row == -1 && col == -1; }
};

enum class CellState { Empty, Red, Blue };
enum class Player { Red, Blue };

inline Player opposite(Player p) {
    return p == Player::Red ? Player::Blue : Player::Red;
}

inline CellState cell_for(Player p) {
    return p == Player::Red ? CellState::Red : CellState::Blue;
}

class HexBoard {
  public:
    HexBoard() = default;
    explicit HexBoard(int size) : size_(size), cells_(size * size, CellState::Empty) {}

    static HexBoard FromSerialized(const std::string &serialized) {
        std::vector<std::string> rows;
        std::string token;
        std::stringstream ss(serialized);
        while (std::getline(ss, token, ',')) {
            rows.push_back(token);
        }

        int board_size = static_cast<int>(rows.size());
        if (board_size == 0) {
            board_size = 11; 
            rows.resize(board_size);
        }

        HexBoard board(board_size);
        // Python engine data is in "Row0, Row1, ..." order
        for (int r = 0; r < board_size; ++r) {
            if (r >= static_cast<int>(rows.size())) break;
            const std::string &row_str = rows[r];
            for (int c = 0; c < board_size && c < static_cast<int>(row_str.size()); ++c) {
                // [KEY] Store in (row, col) order
                board.set(r, c, char_to_cell(row_str[c]));
            }
        }
        return board;
    }

    int size() const { return size_; }

    // Internal storage uses Row-Major (Row * Size + Col) indexing
    CellState get(int row, int col) const { return cells_[index(row, col)]; }
    void set(int row, int col, CellState state) { cells_[index(row, col)] = state; }

    void apply_move(const Move &move, Player player) {
        if (!is_on_board(move.row, move.col)) return;
        set(move.row, move.col, cell_for(player));
    }

    std::vector<Move> legal_moves() const {
        std::vector<Move> moves;
        moves.reserve(size_ * size_);
        for (int r = 0; r < size_; ++r) {
            for (int c = 0; c < size_; ++c) {
                if (get(r, c) == CellState::Empty) {
                    moves.push_back({r, c});
                }
            }
        }
        return moves;
    }

    bool has_winner(Player player) const {
        CellState target = cell_for(player);
        std::vector<char> visited(size_ * size_, 0);
        std::vector<int> stack;

        if (player == Player::Red) {
            // Red connects from top (Row 0) to bottom (Row 10)
            for (int c = 0; c < size_; ++c) {
                if (get(0, c) == target) {
                    int idx = index(0, c);
                    stack.push_back(idx);
                    visited[idx] = 1;
                }
            }
            while (!stack.empty()) {
                int idx = stack.back();
                stack.pop_back();
                int r = idx / size_;
                if (r == size_ - 1) return true; // Reached bottom

                for (const auto &nbr : neighbours(idx)) {
                    if (!visited[nbr] && cells_[nbr] == target) {
                        visited[nbr] = 1;
                        stack.push_back(nbr);
                    }
                }
            }
            return false;
        } else {
            // Blue connects from left (Col 0) to right (Col 10)
            for (int r = 0; r < size_; ++r) {
                if (get(r, 0) == target) {
                    int idx = index(r, 0);
                    stack.push_back(idx);
                    visited[idx] = 1;
                }
            }
            while (!stack.empty()) {
                int idx = stack.back();
                stack.pop_back();
                int c = idx % size_;
                if (c == size_ - 1) return true; // Reached right edge

                for (const auto &nbr : neighbours(idx)) {
                    if (!visited[nbr] && cells_[nbr] == target) {
                        visited[nbr] = 1;
                        stack.push_back(nbr);
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

    int empty_count() const {
        int count = 0;
        for (auto cell : cells_) {
            if (cell == CellState::Empty) ++count;
        }
        return count;
    }

  private:
    static CellState char_to_cell(char c) {
        if (c == 'R') return CellState::Red;
        if (c == 'B') return CellState::Blue;
        return CellState::Empty;
    }

    bool is_on_board(int r, int c) const {
        return r >= 0 && c >= 0 && r < size_ && c < size_;
    }

    int index(int r, int c) const { return r * size_ + c; }

    std::vector<int> neighbours(int idx) const {
        int r = idx / size_;
        int c = idx % size_;
        
        // Hex grid neighbor offsets in (Row, Col) coordinates
        // (r, c-1), (r, c+1) -> left, right
        // (r-1, c), (r-1, c+1) -> upper two neighbors
        // (r+1, c-1), (r+1, c) -> lower two neighbors
        static constexpr int DR[6] = {0, 0, -1, -1, 1, 1};
        static constexpr int DC[6] = {-1, 1, 0, 1, -1, 0};

        std::vector<int> result;
        result.reserve(6);
        for (int i = 0; i < 6; ++i) {
            int nr = r + DR[i];
            int nc = c + DC[i];
            if (is_on_board(nr, nc)) {
                 result.push_back(index(nr, nc));
            }
        }
        return result;
    }

    int size_ = 11;
    std::vector<CellState> cells_;
};

struct CommandData {
    std::string command;
    bool has_move = false;
    Move move;
    std::string board_string;
    int turn = 1;
};

CommandData parse_command(const std::string &line) {
    CommandData data;
    std::vector<std::string> tokens;
    tokens.reserve(5);
    std::size_t start = 0;
    
    while (start < line.size()) {
        std::size_t pos = line.find(';', start);
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
        
        // Parse MOVE (Python sends x,y in Row,Col order)
        if (!tokens[1].empty()) {
            std::size_t comma = tokens[1].find(',');
            if (comma != std::string::npos) {
                try {
                    data.has_move = true;
                    data.move.row = std::stoi(tokens[1].substr(0, comma));
                    data.move.col = std::stoi(tokens[1].substr(comma + 1));
                } catch (...) { data.has_move = false; }
            }
        }

        if (tokens.size() > 3 && !tokens[3].empty()) {
            try { data.turn = std::stoi(tokens[3]); } catch (...) { }
        }
    }
    return data;
}

class MCTS {
  private:
    struct Node {
        Move move;
        Player player_just_moved;
        HexBoard state;
        std::vector<Move> untried_moves;
        std::vector<std::unique_ptr<Node>> children;
        Node *parent = nullptr;
        int visits = 0;
        double wins = 0.0;

        Node(const HexBoard &state, Player player, Node *parent, const Move &move)
            : move(move), player_just_moved(player), state(state), parent(parent) {
            untried_moves = state.legal_moves();
        }

        bool fully_expanded() const { return untried_moves.empty(); }

        Node *best_child(double exploration) const {
            Node *best = nullptr;
            double best_value = -std::numeric_limits<double>::infinity();
            for (const auto &child : children) {
                if (child->visits == 0) return child.get(); 
                double win_rate = child->wins / static_cast<double>(child->visits);
                double uct = win_rate + exploration * std::sqrt(std::log(static_cast<double>(visits)) / static_cast<double>(child->visits));
                if (uct > best_value) {
                    best_value = uct;
                    best = child.get();
                }
            }
            return best;
        }

        Move pop_untried_move(std::mt19937 &rng) {
            std::uniform_int_distribution<std::size_t> dist(0, untried_moves.size() - 1);
            std::size_t idx = dist(rng);
            Move m = untried_moves[idx];
            untried_moves[idx] = untried_moves.back();
            untried_moves.pop_back();
            return m;
        }
    };

  public:
    explicit MCTS(std::mt19937 &rng) : rng_(rng) {}

    Move search(const HexBoard &root_state, Player to_move, int time_limit_ms) {
        std::vector<Move> legal = root_state.legal_moves();
        if (legal.empty()) return {-1, -1};
        
        // First move at center (Row 5, Col 5)
        if (root_state.empty_count() == 11*11) {
            return {5, 5};
        }

        Node root(root_state, opposite(to_move), nullptr, Move{});
        auto start_time = std::chrono::steady_clock::now();
        auto time_budget = std::chrono::milliseconds(time_limit_ms);
        
        int iterations = 0;
        while (true) {
            run_iteration(root, to_move);
            iterations++;
            if (iterations % 200 == 0) {
                if (std::chrono::steady_clock::now() - start_time >= time_budget) break;
            }
        }

        Node *best = root.best_child(0.0);
        if (best == nullptr) return legal.front();
        return best->move;
    }

  private:
    void run_iteration(Node &root, Player to_move) {
        Node *node = &root;
        Player current_player = to_move;

        // Selection
        while (node->fully_expanded() && !node->state.is_terminal() && !node->children.empty()) {
            node = node->best_child(1.41421356237);
            current_player = opposite(node->player_just_moved);
        }

        // Expansion
        if (!node->state.is_terminal() && !node->untried_moves.empty()) {
            Move move = node->pop_untried_move(rng_);
            HexBoard next_state = node->state;
            next_state.apply_move(move, current_player);
            auto child = std::make_unique<Node>(next_state, current_player, node, move);
            node->children.push_back(std::move(child));
            node = node->children.back().get();
            current_player = opposite(current_player);
        }

        // Simulation
        HexBoard rollout_state = node->state;
        Player winner = simulate(rollout_state, current_player);

        // Backpropagation
        while (node != nullptr) {
            node->visits += 1;
            if (node->player_just_moved == winner) {
                node->wins += 1.0;
            }
            node = node->parent;
        }
    }

    Player simulate(HexBoard state, Player to_move) {
        Player current_player = to_move;
        
        while (!state.is_terminal()) {
            std::vector<Move> moves = state.legal_moves();
            if (moves.empty()) break;
            
            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            std::size_t rand_idx = dist(rng_);
            
            Move move = moves[rand_idx];
            state.apply_move(move, current_player);
            current_player = opposite(current_player);
        }
        return state.winner().value_or(opposite(to_move));
    }

    std::mt19937 &rng_;
};

class HexMCTSAgent {
  public:
    HexMCTSAgent(Player colour, int board_size)
        : my_colour_(colour), initial_colour_(colour), board_size_(board_size),
          swap_available_(colour == Player::Blue) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        rng_ = std::mt19937(seed);
        mcts_ = std::make_unique<MCTS>(rng_);
    }

    Move handle_command(const CommandData &data) {
        if (data.command == "SWAP") {
            my_colour_ = opposite(my_colour_);
        }
        
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
        Move move = mcts_->search(board, my_colour_, budget);
        swap_available_ = false;
        return move;
    }

  private:
    bool can_swap_now(const CommandData &data) const {
        return swap_available_ && initial_colour_ == Player::Blue && data.turn == 2 &&
               data.command == "CHANGE";
    }

    bool should_swap(const Move &opp_opening, int size) const {
        int centre = size / 2;
        // Calculate distance in Row, Col coordinates
        int dist_sq = (opp_opening.row - centre) * (opp_opening.row - centre) + 
                      (opp_opening.col - centre) * (opp_opening.col - centre);
        return dist_sq <= std::max(2, size);
    }

    int compute_time_budget(int turn, int empty_tiles) const {
        if (empty_tiles > 90) return 3000; 
        if (empty_tiles > 60) return 2000;
        if (empty_tiles > 30) return 1500;
        return 900; 
    }

    Player my_colour_;
    Player initial_colour_;
    int board_size_;
    bool swap_available_ = false;
    std::mt19937 rng_;
    std::unique_ptr<MCTS> mcts_; 
};

} // namespace

int main(int argc, char **argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc < 3) return 1;
    Player colour = (argv[1][0] == 'R') ? Player::Red : Player::Blue;
    int board_size = std::stoi(argv[2]);

    HexMCTSAgent agent(colour, board_size);
    std::string line;
    
    // Debug tag: Version 3.0 (Final Fix)
    std::cerr << ">>> FIXED VERSION 3.0 LOADED (Row/Col Strict) <<<" << std::endl;

    while (std::getline(std::cin, line)) {
        if (line.empty() || line.back() != ';') continue;
        CommandData data = parse_command(line);
        Move move = agent.handle_command(data);
        
        // [IMPORTANT] Output in Row, Col order as expected by Python engine
        std::cout << move.row << "," << move.col << '\n';
        std::cout.flush(); 
    }
    return 0;
}
