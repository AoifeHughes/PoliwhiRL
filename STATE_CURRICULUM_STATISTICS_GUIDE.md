# State Curriculum Statistics Guide

## 📊 **How State Curriculum Affects Reporting Statistics**

The state curriculum system introduces significant changes to how training statistics should be interpreted. This guide explains the impacts and how to properly analyze your results.

## 🔍 **Key Statistical Changes**

### **1. Episode Length Distribution**
**What Changes:**
- **IMPORTANT**: All episode lengths include the **total steps from game beginning**, including checkpoint steps
- Episodes starting from curriculum states show the **complete game length** (checkpoint + additional steps)
- You'll still see efficiency differences, but they represent true total episode performance

**How to Interpret:**
```
Traditional Training:  All episodes ~400-500 steps (from start)
State Curriculum:      
  Scratch episodes:    ~400-500 steps (from start)
  Curriculum episodes: ~300-450 steps (150 checkpoint + 150-300 additional)
                      ↑ This shows true total game completion time
```

**✅ This is ACCURATE** - Episode lengths represent the true total steps needed to complete the game from beginning to end.

**Example:**
- Episode A (scratch): 400 steps total
- Episode B (curriculum): 320 steps total (200 checkpoint + 120 additional)
- **Episode B was actually more efficient** - it completed the full game in fewer total steps

### **2. Reward Patterns**
**What Changes:**
- **Higher apparent rewards** when starting from progress states
- **Reward curves may show "jumps"** when curriculum sampling changes
- **Different baseline expectations** for scratch vs curriculum episodes

**How to Interpret:**
```
Scratch Episode:     0.0 → 2.5 reward (started from beginning)
Curriculum Episode:  1.8 → 3.2 reward (started from mid-progress)
```

**✅ Focus on efficiency metrics** (reward/step) rather than absolute rewards.

### **3. Success Rate Interpretation**
**What Changes:**
- Success rates need to be **analyzed by starting type**
- Curriculum episodes should have **higher success rates** (they start closer to goals)
- Combined success rate may be **misleadingly high**

**Analysis Approach:**
- Track scratch vs curriculum success rates separately
- Use efficiency gains as the primary performance metric
- Monitor the "scratch episode success rate" for true capability assessment

## 📈 **New Statistics Available**

### **Real-Time Console Output**
Every 25 episodes, you'll see detailed breakdowns:

```
📊 STATE CURRICULUM STATISTICS - Episode 150
============================================================
📈 EPISODE START DISTRIBUTION (Total: 150):
        scratch:  90 episodes ( 60.0%)
    progression:  35 episodes ( 23.3%)
       frontier:  15 episodes ( 10.0%)
           goal:  10 episodes (  6.7%)

📊 RECENT 25 EPISODES:
   Scratch starts:      10 ( 40.0%)
   Curriculum starts:   15 ( 60.0%)
   Avg reward:           2.450
   Avg length:         245.3
   Avg efficiency:      0.0099 (reward/step)

🔍 PERFORMANCE COMPARISON:
   Scratch episodes:
      Avg reward:        1.850 (n=90)
      Avg total length:  425.6 (full episodes)
      Efficiency:        0.0043
   Curriculum episodes:
      Avg reward:        2.850 (n=60)
      Avg total length:  345.2 (including checkpoint)
         Checkpoint:     180.5 (avg steps from saved state)
         Additional:     164.7 (avg steps taken this episode)
      Efficiency:        0.0083
      Efficiency gain:   93.2% vs scratch

🏛️ STATE BUFFER STATUS:
   Total saved states: 45
      foundation:   8 states
     progression:  22 states
        frontier:  12 states
            goal:   3 states

🛡️ SAFETY MONITORING:
   Baseline perf:        1.750
   Recent perf:          2.450
   Performance ratio:    1.400
   Forgetting detected: False
   Safety override:    False
============================================================
```

### **Detailed JSON Reports**
Saved to `Results/state_curriculum_report_ep{episode}.json`:

```json
{
  "episode": 150,
  "stage": 3,
  "efficiency_metrics": {
    "scratch_efficiency": 0.00435,
    "curriculum_efficiency": 0.00826,
    "efficiency_improvement_percent": 89.9
  },
  "performance_comparison": {
    "scratch": {
      "avg_reward": 1.85,
      "avg_total_length": 425.6,
      "episode_count": 90
    },
    "curriculum": {
      "avg_reward": 2.85,
      "avg_total_length": 345.2,
      "avg_checkpoint_steps": 180.5,
      "avg_additional_steps": 164.7,
      "episode_count": 60
    }
  }
}
```

## 🎯 **Key Metrics to Focus On**

### **1. Efficiency Improvement Percentage**
```
Efficiency Gain: 235.2% vs scratch
```
**This is the most important metric** - it shows how much more efficient training becomes with state curriculum.

### **2. Scratch Episode Performance**
```
Scratch episodes: Avg reward: 1.850 (n=90)
```
**This shows true agent capability** - performance when starting from the beginning, unbiased by curriculum states.

### **3. Episode Start Distribution**
```
📈 EPISODE START DISTRIBUTION:
        scratch:  60.0%
    progression:  23.3%
       frontier:  10.0%
```
**This shows curriculum adaptation** - as training progresses, more episodes should start from advanced states.

### **4. Safety Monitoring**
```
🛡️ SAFETY MONITORING:
   Performance ratio:    1.400
   Forgetting detected: False
```
**This ensures no catastrophic forgetting** - ratio should stay above 0.8.

## ⚠️ **Potential Misleading Statistics**

### **❌ Don't Focus On:**

1. **Overall Average Reward** - Will be inflated by curriculum episodes
2. **Overall Average Episode Length** - Will be reduced by curriculum episodes  
3. **Combined Success Rate** - Will be inflated by easier curriculum starts
4. **Training Curves** - May show apparent "jumps" due to sampling changes

### **✅ Focus On Instead:**

1. **Efficiency Metrics** (reward per step)
2. **Scratch Episode Performance** (true capability)
3. **Efficiency Improvement Percentage** (training acceleration)
4. **State Distribution Evolution** (curriculum adaptation)

## 📋 **Practical Analysis Workflow**

### **During Training:**
1. **Monitor console output** every 25 episodes
2. **Check efficiency improvement** - should be >100% for good curriculum
3. **Watch safety monitoring** - ensure no forgetting detected
4. **Verify state distribution** - should shift toward advanced states over time

### **Post-Training Analysis:**
1. **Load latest report**: `Results/state_curriculum_latest.json`
2. **Compare efficiency metrics** across different runs
3. **Analyze scratch episode performance** for true capability assessment
4. **Review state buffer utilization** for curriculum effectiveness

### **Performance Comparison:**
```python
# Example analysis code
import json

# Load state curriculum report
with open('Results/state_curriculum_latest.json', 'r') as f:
    report = json.load(f)

# Key metrics
efficiency_gain = report['efficiency_metrics']['efficiency_improvement_percent']
scratch_performance = report['performance_comparison']['scratch']['avg_reward']
curriculum_utilization = (report['total_episodes'] - report['performance_comparison']['scratch']['episode_count']) / report['total_episodes']

print(f"Efficiency Gain: {efficiency_gain:.1f}%")
print(f"True Agent Capability: {scratch_performance:.3f}")
print(f"Curriculum Utilization: {curriculum_utilization:.1f}%")
```

## 🚨 **Warning Signs to Watch For**

### **Catastrophic Forgetting:**
```
⚠️ CATASTROPHIC FORGETTING DETECTED!
🚨 SAFETY OVERRIDE ACTIVATED:
```
**Action:** System automatically increases scratch training to 90%. Monitor for recovery.

### **Poor Efficiency Gains:**
```
Efficiency gain: 15.2% vs scratch
```
**Action:** Check if curriculum is too aggressive. Consider increasing scratch probability or adjusting state classification thresholds.

### **Low State Utilization:**
```
📈 EPISODE START DISTRIBUTION:
        scratch:  95.0%
    progression:   5.0%
```
**Action:** Check if states are being saved properly. Verify `use_state_curriculum` is true and episodes are achieving sufficient progress.

## 🎉 **Expected Good Results**

### **Healthy State Curriculum Training:**
- **Efficiency gain: 150-300%** vs scratch episodes
- **State distribution**: Gradually shifts from 90% scratch to 40-60% scratch
- **Safety ratio**: Stays above 1.0 (no forgetting)
- **Scratch performance**: Continues improving throughout training
- **Episode length reduction**: 30-50% shorter average episodes

### **Success Indicators:**
1. Console shows increasing curriculum utilization over time
2. Efficiency gains consistently above 100%
3. No safety override activations
4. Scratch episode performance continues improving
5. State buffer accumulates 50+ states per stage

This enhanced statistics system ensures you can properly evaluate the effectiveness of state curriculum learning while avoiding common interpretation pitfalls.