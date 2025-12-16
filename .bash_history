docker run -it --rm -v "$PWD":/home/hex hex bash -c "apt-get update && apt-get install -y g++ make"
docker run -it --rm -v "$PWD":/home/hex hex bash -c "apt-get update && apt-get install -y g++ make"
cd /home/hex/agents/Group34/cpp
g++ -std=c++17 -O3 -o MCTSRaveAgent MCTSRaveAgent.cpp
apt-get update && apt-get install -y g++ make
g++ -std=c++17 -O3 -o MCTSRaveAgent MCTSRaveAgent.cpp
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p1Name "MCTS+RAVE" -p2Name "Naive" -l test_game.log
cd /home/hex
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p1Name "MCTS+RAVE" -p2Name "Naive" -l test_game.log

python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "MCTS+RAVE+NN" -p2Name "Naive" -l test_game.log
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "MCTS+RAVE+NN" -p2Name "Naive" -l test_game.log
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "MCTS+RAVE+NN" -p2Name "Naive" -l test_game.log
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "MCTS+RAVE+NN" -p2Name "Naive" -l test_game.log
[INFO]-2025-12-15 17:51:51,208 - Player Naive; Move: Colour.RED(x=2, y=6)
[INFO]-2025-12-15 17:51:51,208 - Player Naive made an illegal move
[INFO]-2025-12-15 17:51:51,208 - Game over
[INFO]-2025-12-15 17:51:51,208 - Final Board:
   0 1 2 3 4 5 6 7 8 910
  0· · · · · · · · · · B  0
   1· · · · · · · · · B ·  1
    2· · · · · R B B B · ·  2
     3· · · · B B · · · · ·  3
      4· · · B R · · · · · R  4
       5B B · · · R · · · · ·  5
        6· · · · · R · R · · ·  6
         7· · · · R · · · · · ·  7
          8· · · · · · · · · · ·  8
           9· · · · · · · · · · ·  9
           10· · · · · R · R · · R 10
             0 1 2 3 4 5 6 7 8 910
[INFO]-2025-12-15 17:51:51,208 - Total time: 30.27s
[INFO]-2025-12-15 17:51:51,208 - Player Naive made an illegal move
[INFO]-2025-12-15 17:51:51,208 - Player MCTS+RAVE+NN has won
[INFO]-2025-12-15 17:51:51,208 - Total Game Time: 30.27s
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_rann.log
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRaveAgent" -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent"
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRaveAgent" -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent"
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRaveAgent" -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent"
python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRaveAgent" -p2Name "Naive" -l test.log

python3 Hex.py -p1 "agents.Group34.MCTSRaNNAgent MCTSRaveAgent" -p2Name "Naive" -l test.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_nn.log
python3 Hex.py -p1 "agents.Group34.MCTSRaveAgent MCTSRaveAgent" -p2 "agents.Group34.MCTSRaNNAgent MCTSRave300Agent" -p1Name "RAVE-NoNN" -p2Name "RAVE+NN" -l rave_vs_nn.log
python agents/Group34/bin/dk_train_hex.py   --data agents/Group34/bin/dk_selfplay_with_visits.npz   --epochs 80   --batch 256   --channels 128   --blocks 12   --out agents/Group34/bin/dk_weights.pt   --out-ts agents/Group34/bin/dk_hex_ts.pt
python3 agents/Group34/bin/dk_train_hex.py   --data agents/Group34/bin/dk_selfplay_with_visits.npz   --epochs 80   --batch 256   --channels 128   --blocks 12   --out agents/Group34/bin/dk_weights.pt   --out-ts agents/Group34/bin/dk_hex_ts.pt
python3 agents/Group34/bin/dk_logs_to_npz.py   --logs logs/game1.csv logs/game2.csv   --board 11   --out agents/Group34/bin/dk_selfplay_with_visits.npz
exit
cd /home/hex
g++ agents/Group34/cpp/dk_MCTSRaNNAgent.cpp -O3 -std=c++17 -o agents/Group34/bin/dk_mcts_rave_nn_agent
exit
