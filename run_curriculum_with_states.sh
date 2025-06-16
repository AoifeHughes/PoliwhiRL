#!/bin/bash
# Enhanced single-agent curriculum learning script with state curriculum support
# This script demonstrates the integration of state curriculum learning with existing curriculum system

# Define the range for N_goals_target
start_goals=1
max_goals=8  # 8 total goals

# State curriculum configuration
use_state_curriculum=true
state_buffer_size=50  # Smaller buffer for demonstration

# Find the highest existing stage folder to continue from
find_latest_stage() {
    latest_stage=0
    for i in $(seq 1 $max_goals); do
        if [ -d "./stage_${i}" ] && [ -d "./stage_${i}/Checkpoints" ]; then
            latest_stage=$i
        fi
    done
    echo $latest_stage
}

# Check if we should continue from an existing stage
latest_existing_stage=$(find_latest_stage)
if [ $latest_existing_stage -gt 0 ]; then
    echo "🔍 Found existing stages up to stage $latest_existing_stage"
    if [ $latest_existing_stage -lt $max_goals ]; then
        start_goals=$((latest_existing_stage + 1))
        echo "📍 Continuing from stage $start_goals"
    else
        echo "✅ All stages already complete!"
        exit 0
    fi
fi

echo ""
echo "🧠 State Curriculum Learning Configuration:"
echo "   • State curriculum enabled: $use_state_curriculum"
echo "   • State buffer size: $state_buffer_size"
echo "   • States will be saved per stage in curriculum_states/ directories"
echo ""

# Define episode lengths for each goal level
episode_lengths=(
    200    # 1 goal - allow more exploration for state collection
    250    # 2 goals
    500    # 3 goals
    600    # 4 goals
    4000   # 5 goals
    5000   # 6 goals
    10000  # 7 goals
    12000  # 8 goals
)

# Training parameters optimized for state curriculum
episodes_per_stage=300  # Reduced episodes since state curriculum improves efficiency
update_frequency=256
ppo_iterations=25

# Device configuration
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
    echo "🎯 Starting Goal Stage $goals of $max_goals (with State Curriculum)"
    echo "=================================================================="

    # Get the episode length from the array
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
        
        # Also copy state curriculum data if it exists
        if [ -d "./stage_${prev_goals}/curriculum_states" ]; then
            echo "📁 State curriculum data found from previous stage"
            echo "   Note: New stage will start its own state buffer"
        fi
        
        load_checkpoint="./stage_${goals}/Checkpoints"
    fi

    echo "📊 Stage $goals configuration:"
    echo "   • Episodes: $episodes_per_stage"
    echo "   • Max steps per episode: $episode_length"
    echo "   • Update frequency: $update_frequency"
    echo "   • PPO iterations: $ppo_iterations"
    echo "   • Device: $device"
    echo "   • Target goals: $goals"
    echo "   • State curriculum: enabled"
    echo "   • State save location: ./stage_${goals}/curriculum_states/"

    # Run the training with state curriculum enabled
    echo "🚀 Starting training for stage $goals with state curriculum..."
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
        --load_checkpoint "$load_checkpoint" \
        --use_state_curriculum $use_state_curriculum \
        --state_buffer_size $state_buffer_size

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Stage $goals completed successfully!"
        echo "📈 Performance summary:"
        
        # Show state curriculum statistics
        state_dir="./stage_${goals}/curriculum_states"
        if [ -d "$state_dir" ]; then
            state_count=$(find "$state_dir" -name "state_*.pkl" | wc -l)
            echo "   • Saved state checkpoints: $state_count"
            echo "   • State curriculum directory: $state_dir"
            
            if [ -f "$state_dir/state_buffer.pkl" ]; then
                echo "   • State buffer saved for next stage"
            fi
        fi
        
        # Extract final statistics if results file exists
        results_file="./stage_${goals}/Results/training_metrics.json"
        if [ -f "$results_file" ]; then
            echo "   • Training results: $results_file"
        fi
        
        # Show checkpoint info
        checkpoint_dir="./stage_${goals}/Checkpoints"
        if [ -d "$checkpoint_dir" ]; then
            echo "   • Model checkpoint: $checkpoint_dir"
            if [ -d "${checkpoint_dir}_best" ]; then
                echo "   • Best checkpoint: ${checkpoint_dir}_best"
            fi
        fi
        
        echo ""
        echo "🔍 State Curriculum Summary for Stage $goals:"
        if [ -d "$state_dir" ]; then
            echo "   • Foundation states: $(find "$state_dir" -name "*foundation*" | wc -l || echo 0)"
            echo "   • Progression states: $(find "$state_dir" -name "*progression*" | wc -l || echo 0)"
            echo "   • Frontier states: $(find "$state_dir" -name "*frontier*" | wc -l || echo 0)"
            echo "   • Goal states: $(find "$state_dir" -name "*goal*" | wc -l || echo 0)"
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

echo "🎉 Curriculum training with state curriculum complete!"
echo "Trained through $max_goals goal stages"

echo ""
echo "🏆 ENHANCED CURRICULUM TRAINING FINISHED"
echo "================================================"
echo "📁 Results saved in stage_X directories"
echo "📊 Training completed on device: $device"
echo "🎯 Processed stages $start_goals through $max_goals"
echo "🧠 State curriculum data preserved per stage"

echo ""
echo "🔍 Next steps:"
echo "   • Review results in stage_X/Results/ directories"
echo "   • Analyze state curriculum data in stage_X/curriculum_states/"
echo "   • Best checkpoints saved as stage_X/Checkpoints_best/"
echo "   • State curriculum should have reduced training time and improved performance"
echo "   • Consider analyzing which states were most valuable for training"