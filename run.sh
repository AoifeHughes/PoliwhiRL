for ((i=1; i<=2; i++))
do
    # Check if the directory already exists
    if [ -d "run_$i" ]; then
        echo "Directory run_$i already exists, skipping..."
        continue
    fi

    python main.py --device cpu --model RainbowDQN --checkpoint_interval 250 --epsilon_by_location --num_episodes 1000 --sight
    mkdir run_$i
    mv results run_$i
    mv runs/* run_$i
    mv logs run_$i
done
