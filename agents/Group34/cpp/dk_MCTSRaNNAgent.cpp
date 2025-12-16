// Thin DK wrapper that reuses MCTSRaNNAgent.cpp but supplies default
// paths to dk_nn_server.py and dk_weights.pt when not provided.
#include <string>
#include <vector>

// Rename the included main to avoid collision, then include full agent.
#define main MCTS_RANN_MAIN
#include "MCTSRaNNAgent.cpp"
#undef main

// Helper to append default args if missing.
int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(9);
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);

    // If python/server/weights not provided, inject DK defaults relative to repo.
    if (args.size() < 7) {
        args.resize(7);
        if (args.size() >= 4) {
            args[4] = "python3";
            args[5] = "agents/Group34/bin/dk_nn_server.py";
            args[6] = "agents/Group34/bin/dk_weights.pt";
        }
    }
    // VALUE_SWITCH_EMPTY default
    if (args.size() < 8) args.push_back("60");
    // C_PUCT default
    if (args.size() < 9) args.push_back("1.2");

    // Build argv for the real main
    std::vector<char*> cargs;
    cargs.reserve(args.size());
    for (auto& s : args) cargs.push_back(s.data());
    cargs.push_back(nullptr);

    return MCTS_RANN_MAIN(static_cast<int>(args.size()), cargs.data());
}
