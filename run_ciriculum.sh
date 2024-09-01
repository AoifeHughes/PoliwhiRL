
### This goal setup should prioritise learning to get outside

# Learn 1 goal
echo "Learning 1 goal"
start_time=$(date +%s)
python main.py   --episode_length 100 --epochs 8 --num_episodes 2000 --N_goals_target 1 --sequence_length 4 --erase true
end_time=$(date +%s)
runtime=$((end_time-start_time))
echo "Runtime for Learning 1 goal: $runtime seconds"

# Learn 2
echo "Learning 2 goals"
start_time=$(date +%s)
python main.py  --episode_length 200 --epochs 8 --num_episodes 1000 --N_goals_target 2 --sequence_length 8 --erase false
end_time=$(date +%s)
runtime=$((end_time-start_time))
echo "Runtime for Learning 2 goals: $runtime seconds"

# Learn 3
echo "Learning 3 goals"
start_time=$(date +%s)
python main.py  --episode_length 200 --epochs 8 --num_episodes 1000 --N_goals_target 3 --sequence_length 8 --erase false
end_time=$(date +%s)
runtime=$((end_time-start_time))
echo "Runtime for Learning 3 goals: $runtime seconds"

# Learn 4
echo "Learning 4 goals"
start_time=$(date +%s)
python main.py  --episode_length 500 --epochs 8 --num_episodes 500 --N_goals_target 4 --sequence_length 32 --erase false
end_time=$(date +%s)
runtime=$((end_time-start_time))
echo "Runtime for Learning 4 goals: $runtime seconds"

# Learn 5
echo "Learning 5 goals"
start_time=$(date +%s)
python main.py  --episode_length 1000 --epochs 8 --num_episodes 1000 --N_goals_target 5 --sequence_length 64 --erase false
end_time=$(date +%s)
runtime=$((end_time-start_time))
echo "Runtime for Learning 5 goals: $runtime seconds"

# Total runtime
total_end_time=$(date +%s)
total_runtime=$((total_end_time-total_start_time))
echo "Total runtime: $total_runtime seconds"
