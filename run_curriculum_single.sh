#!/bin/bash
# Single-agent curriculum learning script optimized for step minimization
# This script progressively trains a single agent with tighter step constraints

# Define the range for N_goals_target
start_goals=1
max_goals=7  # 7 total goals

# Optional: Run optimization passes with progressively tighter constraints
run_optimization_pass=true
optimization_reduction=0.8  # Reduce episode length by 20% in optimization pass

# Define a list of episode lengths for each goal level (aggressive step minimization)
# These are much tighter to force the agent to find optimal paths
episode_lengths=(
    75    # 1 goal - reduced from 150
    150    # 2 goals - reduced from 200
    600    # 3 goals - reduced from 300
    1000    # 4 goals - reduced from 400
    3000    # 5 goals - reduced from 600
    5000    # 6 goals - reduced from 1000
    10000    # 7 goals - reduced from 1500
)

# Single-agent training parameters
episodes_per_stage=500  
update_frequency=256
ppo_iterations=30       # Reduced from 50 for more stable learning

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

for goals in $(seq $start_goals $max_goals); do
    echo ""
    echo "🎯 Starting Goal Stage $goals of $max_goals"
    echo "================================================"

    # Get the episode length from the array (arrays are 0-indexed)
    episode_length=${episode_lengths[$((goals-1))]}

    # Copy checkpoint from previous stage (except for first stage)
    if [ $goals -eq 1 ]; then
        echo "🆕 Starting from scratch for stage 1"
        load_checkpoint=""
    else
        prev_goals=$((goals - 1))
        echo "📋 Copying checkpoint from stage ${prev_goals} to stage ${goals}"
        mkdir -p "./stage_${goals}"
        if [ -d "./stage_${prev_goals}/Checkpoints" ]; then
            cp -r "./stage_${prev_goals}/Checkpoints" "./stage_${goals}/Checkpoints"
            echo "✅ Checkpoint copied successfully"
        fi
        # Load from current stage's checkpoint directory
        load_checkpoint="./stage_${goals}/Checkpoints"
    fi

    echo "📊 Stage $goals configuration:"
    echo "   • Episodes: $episodes_per_stage"
    echo "   • Max steps per episode: $episode_length"
    echo "   • Update frequency: $update_frequency"
    echo "   • PPO iterations: $ppo_iterations"
    echo "   • Device: $device"
    echo "   • Target goals: $goals"

    # Run the single-agent training
    echo "🚀 Starting training for stage $goals..."
    python main.py \
        --model PPO \
        --device $device \
        --vision true \
        --episode_length $episode_length \
        --num_episodes $episodes_per_stage \
        --ppo_update_frequency $update_frequency \
        --ppo_iterations $ppo_iterations \
        --N_goals_target $goals \
        --output_base_dir "stage_${goals}/" \
        --ppo_num_agents 1 \
        --punish_steps true \
        --report_episode true \
        --use_curriculum false \
        --break_on_goal true \
        --load_checkpoint "$load_checkpoint"

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Stage $goals completed successfully!"
        echo "📈 Performance summary:"
        
        # Extract final statistics if results file exists
        results_file="./stage_${goals}/Results/training_metrics.json"
        if [ -f "$results_file" ]; then
            echo "   • Check results in: $results_file"
        fi
        
        # Show checkpoint info
        checkpoint_dir="./stage_${goals}/Checkpoints"
        if [ -d "$checkpoint_dir" ]; then
            echo "   • Model saved to: $checkpoint_dir"
            if [ -d "${checkpoint_dir}_best" ]; then
                echo "   • Best checkpoint: ${checkpoint_dir}_best"
            fi
        fi
        
    else
        echo "❌ Stage $goals failed with exit code $exit_code"
        echo "💡 Check logs in ./stage_${goals}/Results/ for debugging"
        
        # Ask if user wants to continue or abort
        read -p "Continue to next stage anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "🛑 Curriculum training aborted at stage $goals"
            exit $exit_code
        fi
    fi
    
    echo "⏱️  Stage $goals duration: completed"
    echo ""
done

echo "🎉 Base curriculum training complete!"
echo "Trained through $max_goals goal stages"

# Optional optimization pass: Further reduce episode lengths for already-trained goals
if [ "$run_optimization_pass" = true ]; then
    echo ""
    echo "🔧 STARTING OPTIMIZATION PASS"
    echo "================================================"
    echo "Running optimization pass to further minimize steps..."

    for goals in $(seq $start_goals $max_goals); do
        echo ""
        echo "⚡ Optimizing stage $goals with reduced episode length"

        # Calculate reduced episode length
        original_length=${episode_lengths[$((goals-1))]}
        optimized_length=$(echo "$original_length * $optimization_reduction" | bc | cut -d. -f1)

        # Copy checkpoint from regular training to optimized directory
        echo "📋 Copying checkpoint from stage ${goals} to stage ${goals}_optimized"
        mkdir -p "./stage_${goals}_optimized"
        if [ -d "./stage_${goals}/Checkpoints" ]; then
            cp -r "./stage_${goals}/Checkpoints" "./stage_${goals}_optimized/Checkpoints"
            echo "✅ Checkpoint copied for optimization"
        fi

        # Load checkpoint from optimized directory
        load_checkpoint="./stage_${goals}_optimized/Checkpoints"

        echo "📊 Optimization stage $goals configuration:"
        echo "   • Episodes: $((episodes_per_stage / 2))"  # Fewer episodes for optimization
        echo "   • Max steps: $optimized_length (reduced from $original_length)"
        echo "   • Device: $device"

        # Run optimization with tighter constraints
        echo "🚀 Starting optimization for stage $goals..."
        python main.py \
            --model PPO \
            --device $device \
            --vision true \
            --episode_length $optimized_length \
            --num_episodes $((episodes_per_stage / 2)) \
            --ppo_update_frequency $update_frequency \
            --ppo_iterations $((ppo_iterations + 10)) \
            --N_goals_target $goals \
            --output_base_dir "stage_${goals}_optimized/" \
            --ppo_num_agents 1 \
            --punish_steps true \
            --report_episode true \
            --use_curriculum false \
            --break_on_goal true \
            --enable_recovery true \
            --recovery_method auto \
            --load_checkpoint "$load_checkpoint" \
            --save_checkpoint true \
            --checkpoint_frequency 25

        opt_exit_code=$?
        if [ $opt_exit_code -eq 0 ]; then
            echo "✅ Optimization for $goals goals complete!"
            echo "📉 Reduced from $original_length to $optimized_length steps"
        else
            echo "⚠️  Optimization for $goals goals failed (exit code: $opt_exit_code)"
            echo "💡 Original stage $goals checkpoint still available"
        fi
    done

    echo ""
    echo "🔧 OPTIMIZATION PASS COMPLETE"
    echo "================================================"
fi

echo ""
echo "🏆 CURRICULUM TRAINING FINISHED"
echo "================================================"
echo "📁 Results saved in stage_X directories"
echo "📊 Training completed on device: $device"
echo "🎯 All $max_goals goal stages processed"

if [ "$run_optimization_pass" = true ]; then
    echo "⚡ Optimization pass included"
    echo "💾 Both regular and optimized checkpoints available"
fi

echo ""
echo "🔍 Next steps:"
echo "   • Review results in stage_X/Results/ directories"
echo "   • Best checkpoints saved as stage_X/Checkpoints_best/"
echo "   • Use best performing stage for production"
echo "   • Consider running ./run_evaluation.sh to test final performance"