
for ((i=1; i<=100; i++))
do
    python main.py --device mps --model RainbowDQN --checkpoint_interval 50000 --epsilon_by_location --num_episodes 1000 --sight 
    mkdir run_$i
    mv results run_$i
    mv runs/Rainbow run_$i
    mv logs run_$i
done
