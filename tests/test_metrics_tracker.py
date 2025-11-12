# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import os
from PoliwhiRL.utils.metrics_tracker import MetricsTracker


class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = MetricsTracker(self.temp_dir, "test_experiment")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_log_episode(self):
        """Test logging a single episode"""
        self.tracker.log_episode(
            episode=1,
            total_reward=150.5,
            episode_length=200,
            goals_reached=2,
            entropy=0.05,
            success=True,
            curriculum_stage=1,
        )

        self.assertEqual(len(self.tracker.episode_metrics), 1)
        self.assertEqual(self.tracker.episode_metrics[0]["episode"], 1)
        self.assertEqual(self.tracker.episode_metrics[0]["total_reward"], 150.5)
        self.assertEqual(self.tracker.episode_metrics[0]["success"], True)

    def test_log_multiple_episodes(self):
        """Test logging multiple episodes"""
        for i in range(10):
            self.tracker.log_episode(
                episode=i,
                total_reward=100 + i * 10,
                episode_length=200 - i * 5,
                goals_reached=min(i, 3),
                entropy=0.05,
                success=i > 5,
                curriculum_stage=1,
            )

        self.assertEqual(len(self.tracker.episode_metrics), 10)

    def test_get_recent_metrics(self):
        """Test getting recent metrics with window"""
        for i in range(30):
            self.tracker.log_episode(
                episode=i,
                total_reward=100,
                episode_length=200,
                goals_reached=1,
                entropy=0.05,
                success=False,
                curriculum_stage=1,
            )

        recent = self.tracker.get_recent_metrics(window=10)
        self.assertEqual(len(recent), 10)

        # Test with smaller dataset than window
        tracker2 = MetricsTracker(self.temp_dir, "test_2")
        for i in range(5):
            tracker2.log_episode(
                episode=i,
                total_reward=100,
                episode_length=200,
                goals_reached=1,
                entropy=0.05,
                success=False,
                curriculum_stage=1,
            )
        recent2 = tracker2.get_recent_metrics(window=10)
        self.assertEqual(len(recent2), 5)

    def test_compute_summary_statistics(self):
        """Test computing summary statistics"""
        # Log episodes with known values
        for i in range(20):
            self.tracker.log_episode(
                episode=i,
                total_reward=100.0,  # Fixed reward
                episode_length=200,  # Fixed length
                goals_reached=2,
                entropy=0.05,
                success=i >= 10,  # 50% success rate
                curriculum_stage=1,
            )

        stats = self.tracker.compute_summary_statistics(window=20)

        self.assertAlmostEqual(stats["avg_reward"], 100.0, delta=0.01)
        self.assertEqual(stats["avg_episode_length"], 200)
        self.assertAlmostEqual(stats["success_rate"], 0.5, delta=0.01)
        self.assertEqual(stats["num_episodes"], 20)

    def test_export_to_csv(self):
        """Test exporting metrics to CSV"""
        for i in range(5):
            self.tracker.log_episode(
                episode=i,
                total_reward=100,
                episode_length=200,
                goals_reached=1,
                entropy=0.05,
                success=True,
                curriculum_stage=1,
            )

        csv_path = self.tracker.export_to_csv()
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(csv_path.endswith(".csv"))

    def test_export_to_json(self):
        """Test exporting metrics to JSON"""
        for i in range(5):
            self.tracker.log_episode(
                episode=i,
                total_reward=100,
                episode_length=200,
                goals_reached=1,
                entropy=0.05,
                success=True,
                curriculum_stage=1,
            )

        json_path = self.tracker.export_to_json()
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(json_path.endswith(".json"))

    def test_log_curriculum_stage_completion(self):
        """Test logging curriculum stage completion"""
        self.tracker.log_curriculum_stage_completion(
            stage=1, final_success_rate=0.8, final_episode_length=150, episodes_taken=50
        )

        stage_file = os.path.join(
            self.tracker.metrics_dir, "curriculum_stage_1_summary.json"
        )
        self.assertTrue(os.path.exists(stage_file))

    def test_generate_training_report(self):
        """Test generating a training report"""
        # Add some episodes across multiple curriculum stages
        for stage in range(1, 3):
            for i in range(10):
                self.tracker.log_episode(
                    episode=stage * 10 + i,
                    total_reward=100 + stage * 50,
                    episode_length=200 - stage * 20,
                    goals_reached=stage,
                    entropy=0.05,
                    success=i >= 5,
                    curriculum_stage=stage,
                )

        report = self.tracker.generate_training_report()

        # Check that report was generated
        self.assertIsInstance(report, str)
        self.assertIn("Training Report", report)
        self.assertIn("Overall Statistics", report)

        # Check that report file was created
        report_path = os.path.join(self.tracker.metrics_dir, "training_report.txt")
        self.assertTrue(os.path.exists(report_path))

    def test_get_stage_metrics(self):
        """Test getting metrics for a specific curriculum stage"""
        # Log episodes for different stages
        for stage in range(1, 4):
            for i in range(5):
                self.tracker.log_episode(
                    episode=stage * 10 + i,
                    total_reward=100,
                    episode_length=200,
                    goals_reached=stage,
                    entropy=0.05,
                    success=True,
                    curriculum_stage=stage,
                )

        # Test getting stage 2 metrics
        stage_2_metrics = self.tracker.get_stage_metrics(2)
        self.assertEqual(len(stage_2_metrics), 5)
        for metric in stage_2_metrics:
            self.assertEqual(metric["curriculum_stage"], 2)

        # Test getting non-existent stage
        stage_10_metrics = self.tracker.get_stage_metrics(10)
        self.assertEqual(len(stage_10_metrics), 0)


if __name__ == "__main__":
    unittest.main()
