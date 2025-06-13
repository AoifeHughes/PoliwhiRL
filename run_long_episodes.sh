#!/bin/bash
# Long episode training script without curriculum
# Each episode runs for extended periods to allow deep exploration

# Training parameters
num_episodes=50          # Fewer but much longer episodes
episode_length=50000     # 50k steps per episode (vs ~500 in curriculum)
update_frequency=256    # Less frequent updates for stability
ppo_iterations=15        # Moderate iterations to prevent overfitting

# Episode length options (uncomment one):
# episode_length=25000   # Short long episodes (25k steps)
# episode_length=50000   # Medium long episodes (50k steps) - DEFAULT
# episode_length=100000  # Very long episodes (100k steps)

# Training strategy
episodes_per_checkpoint=5  # Save checkpoint every 5 episodes
goal_targets=(1 3 5 7)     # Try different goal complexities

echo "🏃‍♂️ LONG EPISODE TRAINING"
echo "================================================"
echo "Episode length: $episode_length steps"
echo "Number of episodes: $num_episodes"
echo "Update frequency: $update_frequency"
echo "PPO iterations: $ppo_iterations"

# Device configuration - automatically detect best available
device="cpu"
if python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    device="mps"
    echo "🚀 Using Apple Silicon MPS acceleration"
elif python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    device="cuda"
    echo "🚀 Using CUDA GPU acceleration"
else
    echo "🖥️  Using CPU (no GPU acceleration available)"
fi
echo "================================================"

# Test different goal complexities
for goals in "${goal_targets[@]}"; do
    echo ""
    echo "🎯 Starting Long Episode Training: $goals goals"
    echo "================================================"
    
    output_dir="long_episodes_${goals}goals_${episode_length}steps"
    
    echo "📊 Configuration:"
    echo "   • Episodes: $num_episodes"
    echo "   • Episode length: $episode_length steps"
    echo "   • Update frequency: $update_frequency"
    echo "   • PPO iterations: $ppo_iterations"
    echo "   • Device: $device"
    echo "   • Target goals: $goals"
    echo "   • Output directory: $output_dir"
    
    echo "🚀 Starting long episode training for $goals goals..."
    
    # Run training with step-based entropy decay and long episodes
    python main.py \
        --model PPO \
        --device $device \
        --vision true \
        --episode_length $episode_length \
        --num_episodes $num_episodes \
        --ppo_update_frequency $update_frequency \
        --ppo_iterations $ppo_iterations \
        --ppo_entropy_decay_mode "step" \
        --ppo_entropy_coef 0.15 \
        --ppo_entropy_coef_min 0.08 \
        --N_goals_target $goals \
        --output_base_dir "$output_dir/" \
        --ppo_num_agents 1 \
        --punish_steps false \
        --report_episode true \
        --use_curriculum false \
        --break_on_goal false \
        --checkpoint_frequency $episodes_per_checkpoint \
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Long episode training for $goals goals completed successfully!"
        echo "📊 Results saved in: $output_dir"
        
        # Show checkpoint info
        checkpoint_dir="$output_dir/Checkpoints"
        if [ -d "$checkpoint_dir" ]; then
            echo "   • Final checkpoint: $checkpoint_dir"
            if [ -d "${checkpoint_dir}_best" ]; then
                echo "   • Best checkpoint: ${checkpoint_dir}_best"
            fi
        fi
        
        echo "📈 Performance summary:"
        results_file="$output_dir/Results/training_metrics.json"
        if [ -f "$results_file" ]; then
            echo "   • Training metrics: $results_file"
        fi
        
    else
        echo "❌ Long episode training for $goals goals failed (exit code: $exit_code)"
        echo "💡 Check logs in $output_dir/Results/ for debugging"
        
        # Ask if user wants to continue or abort
        read -p "Continue to next goal complexity? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "🛑 Long episode training aborted"
            exit $exit_code
        fi
    fi
    
    echo ""
done

echo "🎉 LONG EPISODE TRAINING COMPLETE!"
echo "================================================"
echo "📁 Results saved in long_episodes_*goals_*steps directories"
echo "📊 Training completed on device: $device"
echo "🎯 Goal complexities tested: ${goal_targets[*]}"
echo ""
echo "🔍 Next steps:"
echo "   • Compare performance across different goal complexities"
echo "   • Review step-based entropy decay effectiveness"
echo "   • Check exploration patterns in long episodes"
echo "   • Consider running evaluation to test final performance"
echo ""
echo "💡 To run evaluation on best model:"
echo "   python main.py --model evaluate --load_checkpoint long_episodes_Xgoals_${episode_length}steps/Checkpoints_best"