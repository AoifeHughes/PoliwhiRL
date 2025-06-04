#!/bin/bash
# Evaluation script for trained curriculum models
# Tests the final performance across different stages

echo "🧪 CURRICULUM MODEL EVALUATION"
echo "================================================"

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

# Evaluation parameters
eval_episodes=10
max_goals=7

echo "📊 Evaluation configuration:"
echo "   • Episodes per stage: $eval_episodes"
echo "   • Device: $device"
echo "   • Recording: enabled"
echo ""

# Test each stage
for goals in $(seq 1 $max_goals); do
    echo "🎯 Evaluating Stage $goals performance"
    echo "----------------------------------------"
    
    # Find the best checkpoint to use
    checkpoint_path=""
    
    # Priority: optimized > best > regular
    if [ -d "./stage_${goals}_optimized/Checkpoints_best" ]; then
        checkpoint_path="./stage_${goals}_optimized/Checkpoints_best"
        echo "📁 Using optimized best checkpoint"
    elif [ -d "./stage_${goals}_optimized/Checkpoints" ]; then
        checkpoint_path="./stage_${goals}_optimized/Checkpoints"
        echo "📁 Using optimized checkpoint"
    elif [ -d "./stage_${goals}/Checkpoints_best" ]; then
        checkpoint_path="./stage_${goals}/Checkpoints_best"
        echo "📁 Using best checkpoint"
    elif [ -d "./stage_${goals}/Checkpoints" ]; then
        checkpoint_path="./stage_${goals}/Checkpoints"
        echo "📁 Using regular checkpoint"
    else
        echo "❌ No checkpoint found for stage $goals, skipping..."
        continue
    fi
    
    echo "📊 Testing $goals-goal performance with $eval_episodes episodes"
    
    # Create evaluation output directory
    eval_dir="evaluation_stage_${goals}"
    mkdir -p "$eval_dir"
    
    # Run evaluation
    python main.py \
        --model PPO \
        --device $device \
        --vision true \
        --episode_length 10000 \
        --num_episodes $eval_episodes \
        --N_goals_target $goals \
        --output_base_dir "$eval_dir/" \
        --ppo_num_agents 1 \
        --punish_steps false \
        --report_episode true \
        --use_curriculum false \
        --break_on_goal true \
        --load_checkpoint "$checkpoint_path" \
        --save_checkpoint false \
        --record true \
        --record_path "$eval_dir/recordings"
    
    eval_exit_code=$?
    if [ $eval_exit_code -eq 0 ]; then
        echo "✅ Stage $goals evaluation complete"
        
        # Try to extract performance metrics
        if [ -f "$eval_dir/Results/training_metrics.json" ]; then
            echo "📈 Results saved to: $eval_dir/Results/"
        fi
        
        if [ -d "$eval_dir/recordings" ]; then
            echo "🎥 Recordings saved to: $eval_dir/recordings/"
        fi
    else
        echo "❌ Stage $goals evaluation failed"
    fi
    
    echo ""
done

echo "🏁 EVALUATION COMPLETE"
echo "================================================"
echo "📁 Results saved in evaluation_stage_X directories"
echo "🎥 Video recordings available for review"
echo "📊 Compare performance across different goal stages"
echo ""
echo "💡 Tips for analysis:"
echo "   • Check average episode length (lower = more efficient)"
echo "   • Review success rate (reaching all goals)"
echo "   • Watch recordings to see strategy quality"
echo "   • Compare optimized vs regular checkpoints"