# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict


class MetricsTracker:
    """Comprehensive metrics tracking for PPO training with curriculum learning"""

    def __init__(self, results_dir, experiment_name=None):
        self.results_dir = results_dir
        self.experiment_name = experiment_name or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.metrics_dir = os.path.join(results_dir, "metrics", self.experiment_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Metrics storage
        self.episode_metrics = []
        self.curriculum_stage_metrics = defaultdict(list)
        self.action_distribution = defaultdict(int)
        self.reward_components = defaultdict(list)

        # Curriculum tracking
        self.current_stage = 0
        self.stage_start_episode = 0
        self.stage_episode_lengths = []
        self.stage_success_rates = []

    def log_episode(
        self,
        episode,
        total_reward,
        episode_length,
        goals_reached,
        entropy,
        loss=None,
        icm_loss=None,
        extrinsic_reward=None,
        intrinsic_reward=None,
        exploration_bonus=None,
        success=False,
        curriculum_stage=None,
    ):
        """Log metrics for a single episode"""
        metrics = {
            "episode": episode,
            "total_reward": float(total_reward),
            "episode_length": int(episode_length),
            "goals_reached": int(goals_reached),
            "entropy": float(entropy),
            "success": bool(success),
            "timestamp": datetime.now().isoformat(),
        }

        if loss is not None:
            metrics["loss"] = float(loss)
        if icm_loss is not None:
            metrics["icm_loss"] = float(icm_loss)
        if extrinsic_reward is not None:
            metrics["extrinsic_reward"] = float(extrinsic_reward)
        if intrinsic_reward is not None:
            metrics["intrinsic_reward"] = float(intrinsic_reward)
        if exploration_bonus is not None:
            metrics["exploration_bonus"] = float(exploration_bonus)
        if curriculum_stage is not None:
            metrics["curriculum_stage"] = int(curriculum_stage)
            self.current_stage = int(curriculum_stage)

        self.episode_metrics.append(metrics)

        # Track curriculum stage metrics
        if curriculum_stage is not None:
            self.curriculum_stage_metrics[curriculum_stage].append(metrics)

    def log_action_distribution(self, action_counts):
        """Log action distribution for analysis"""
        for action, count in action_counts.items():
            self.action_distribution[action] += count

    def log_curriculum_stage_completion(
        self, stage, final_success_rate, final_episode_length, episodes_taken
    ):
        """Log completion of a curriculum stage"""
        stage_summary = {
            "stage": int(stage),
            "final_success_rate": float(final_success_rate),
            "final_episode_length": int(final_episode_length),
            "episodes_taken": int(episodes_taken),
            "timestamp": datetime.now().isoformat(),
        }

        # Save stage summary
        stage_file = os.path.join(
            self.metrics_dir, f"curriculum_stage_{stage}_summary.json"
        )
        with open(stage_file, "w") as f:
            json.dump(stage_summary, f, indent=2)

    def get_recent_metrics(self, window=20):
        """Get metrics for the last N episodes"""
        if len(self.episode_metrics) < window:
            return self.episode_metrics
        return self.episode_metrics[-window:]

    def get_stage_metrics(self, stage):
        """Get all metrics for a specific curriculum stage"""
        return self.curriculum_stage_metrics.get(stage, [])

    def compute_summary_statistics(self, window=100):
        """Compute summary statistics over a rolling window"""
        recent = self.get_recent_metrics(window)
        if not recent:
            return {}

        return {
            "avg_reward": np.mean([m["total_reward"] for m in recent]),
            "std_reward": np.std([m["total_reward"] for m in recent]),
            "avg_episode_length": np.mean([m["episode_length"] for m in recent]),
            "std_episode_length": np.std([m["episode_length"] for m in recent]),
            "avg_goals": np.mean([m["goals_reached"] for m in recent]),
            "success_rate": np.mean([m.get("success", False) for m in recent]),
            "avg_entropy": np.mean([m["entropy"] for m in recent]),
            "num_episodes": len(recent),
        }

    def export_to_csv(self):
        """Export all metrics to CSV for analysis"""
        if not self.episode_metrics:
            return

        df = pd.DataFrame(self.episode_metrics)
        csv_path = os.path.join(self.metrics_dir, "episode_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported metrics to {csv_path}")

        return csv_path

    def export_to_json(self):
        """Export all metrics to JSON"""
        export_data = {
            "experiment_name": self.experiment_name,
            "episode_metrics": self.episode_metrics,
            "action_distribution": dict(self.action_distribution),
            "summary_statistics": self.compute_summary_statistics(),
            "curriculum_stages": {
                str(k): v for k, v in self.curriculum_stage_metrics.items()
            },
        }

        json_path = os.path.join(self.metrics_dir, "metrics_export.json")
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported metrics to {json_path}")

        return json_path

    def generate_training_report(self):
        """Generate a comprehensive training report"""
        report = []
        report.append("=" * 80)
        report.append(f"Training Report: {self.experiment_name}")
        report.append("=" * 80)
        report.append("")

        # Overall statistics
        stats = self.compute_summary_statistics(len(self.episode_metrics))
        report.append("Overall Statistics:")
        report.append(f"  Total Episodes: {len(self.episode_metrics)}")
        report.append(f"  Average Reward: {stats.get('avg_reward', 0):.2f}")
        report.append(
            f"  Average Episode Length: {stats.get('avg_episode_length', 0):.1f}"
        )
        report.append(f"  Success Rate: {stats.get('success_rate', 0) * 100:.1f}%")
        report.append(f"  Average Goals Reached: {stats.get('avg_goals', 0):.2f}")
        report.append("")

        # Curriculum stage breakdown
        if self.curriculum_stage_metrics:
            report.append("Curriculum Stage Breakdown:")
            for stage in sorted(self.curriculum_stage_metrics.keys()):
                stage_data = self.curriculum_stage_metrics[stage]
                stage_success = np.mean([m.get("success", False) for m in stage_data])
                stage_reward = np.mean([m["total_reward"] for m in stage_data])
                stage_length = np.mean([m["episode_length"] for m in stage_data])

                report.append(f"  Stage {stage}:")
                report.append(f"    Episodes: {len(stage_data)}")
                report.append(f"    Success Rate: {stage_success * 100:.1f}%")
                report.append(f"    Avg Reward: {stage_reward:.2f}")
                report.append(f"    Avg Length: {stage_length:.1f}")
                report.append("")

        # Action distribution
        if self.action_distribution:
            report.append("Action Distribution:")
            total_actions = sum(self.action_distribution.values())
            action_names = {
                0: "None",
                1: "A",
                2: "B",
                3: "Left",
                4: "Right",
                5: "Up",
                6: "Down",
                7: "Start",
                8: "Select",
            }
            for action, count in sorted(self.action_distribution.items()):
                percentage = (count / total_actions) * 100
                action_name = action_names.get(action, f"Action {action}")
                report.append(f"  {action_name}: {percentage:.1f}% ({count} times)")
            report.append("")

        # Recent performance
        recent_stats = self.compute_summary_statistics(20)
        report.append("Recent Performance (last 20 episodes):")
        report.append(f"  Average Reward: {recent_stats.get('avg_reward', 0):.2f}")
        report.append(
            f"  Success Rate: {recent_stats.get('success_rate', 0) * 100:.1f}%"
        )
        report.append(
            f"  Average Episode Length: {recent_stats.get('avg_episode_length', 0):.1f}"
        )
        report.append("")

        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.metrics_dir, "training_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {report_path}")

        return report_text

    def compare_with_baseline(self, baseline_metrics_path):
        """Compare current run with a baseline metrics file"""
        try:
            with open(baseline_metrics_path, "r") as f:
                baseline = json.load(f)

            current_stats = self.compute_summary_statistics()
            baseline_stats = baseline.get("summary_statistics", {})

            comparison = {
                "reward_improvement": (
                    current_stats.get("avg_reward", 0)
                    - baseline_stats.get("avg_reward", 0)
                ),
                "success_rate_improvement": (
                    current_stats.get("success_rate", 0)
                    - baseline_stats.get("success_rate", 0)
                ),
                "episode_length_change": (
                    current_stats.get("avg_episode_length", 0)
                    - baseline_stats.get("avg_episode_length", 0)
                ),
            }

            print("\nComparison with Baseline:")
            print(f"  Reward Improvement: {comparison['reward_improvement']:+.2f}")
            print(
                f"  Success Rate Improvement: {comparison['success_rate_improvement'] * 100:+.1f}%"
            )
            print(
                f"  Episode Length Change: {comparison['episode_length_change']:+.1f}"
            )

            return comparison
        except Exception as e:
            print(f"Could not compare with baseline: {e}")
            return None
