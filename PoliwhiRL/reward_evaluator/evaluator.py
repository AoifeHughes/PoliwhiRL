# -*- coding: utf-8 -*-
from PoliwhiRL.environment import PyBoyEnvironment as Env
import sqlite3


def get_longest_manual_sequence(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find the longest manual sequence
    cursor.execute(
        """
        SELECT manual_run_id, COUNT(*) as length
        FROM memory_data
        WHERE is_manual = 1
        GROUP BY manual_run_id
        ORDER BY length DESC
        LIMIT 1
    """
    )
    longest_run = cursor.fetchone()

    if longest_run is None:
        print("No manual entries found in the database.")
        return []

    longest_run_id, _ = longest_run

    # Get the actions for the longest manual sequence
    cursor.execute(
        """
        SELECT action
        FROM memory_data
        WHERE is_manual = 1 AND manual_run_id = ?
        ORDER BY id
    """,
        (longest_run_id,),
    )

    actions = [row[0] for row in cursor.fetchall()]
    conn.close()

    return actions


def evaluate_reward_system(config):
    env = Env(config)

    # Get the path to the database from the config
    db_path = config.get("explore_db_loc", "memory_data.db")
    # Get the longest sequence of button presses
    button_presses = get_longest_manual_sequence(db_path)

    if not button_presses:
        print("No button presses found. Unable to evaluate reward system.")
        return

    print(f"Evaluating reward system with {len(button_presses)} button presses.")

    # Reset the environment
    env.reset()

    # Apply each action and observe the result
    for action in button_presses:
        observation, reward, done, info = env.step(action)

        # Here you can add your reward system evaluation logic
        # For example, you might want to print the reward for each action:
        print(f"Action: {action}, Reward: {reward}")

        if done:
            print("Environment signaled completion before all actions were executed.")
            break

    print("Evaluation complete.")
