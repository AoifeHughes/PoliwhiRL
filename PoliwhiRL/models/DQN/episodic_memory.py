import os
import pickle
import sqlite3
import numpy as np


class EpisodicMemory:
    def __init__(self, memory_size, db_path="./database/episodic_memory.db"):
        # Check if the folder for the database exists, create it if it doesn't
        db_folder = os.path.dirname(self.db_path)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        self.memory_size = memory_size
        self.db_path = db_path
        self.current_episode = []
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS episodes
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          states BLOB,
                          actions BLOB,
                          rewards BLOB,
                          next_states BLOB,
                          dones BLOB,
                          total_reward REAL)"""
            )
            conn.commit()

    def add(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            states, actions, rewards, next_states, dones = zip(*self.current_episode)
            states = np.stack(states)
            next_states = np.stack(next_states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            total_reward = np.sum(rewards)

            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO episodes (states, actions, rewards, next_states, dones, total_reward)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        pickle.dumps(states),
                        pickle.dumps(actions),
                        pickle.dumps(rewards),
                        pickle.dumps(next_states),
                        pickle.dumps(dones),
                        total_reward,
                    ),
                )
                conn.commit()

            self.current_episode = []

            # Limit the number of stored episodes to memory_size
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    "DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
                    (self.memory_size,),
                )
                conn.commit()

    def sample(self, batch_size):
        half_batch_size = batch_size // 2

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # Retrieve half the batch from the best-rewarded episodes
            c.execute(
                """SELECT states, actions, rewards, next_states, dones
                         FROM episodes
                         ORDER BY total_reward DESC
                         LIMIT ?""",
                (half_batch_size,),
            )
            best_rows = c.fetchall()

            # Retrieve the other half randomly
            c.execute(
                """SELECT states, actions, rewards, next_states, dones
                         FROM episodes
                         ORDER BY RANDOM()
                         LIMIT ?""",
                (batch_size - half_batch_size,),
            )
            random_rows = c.fetchall()

        episodes = []
        for row in best_rows + random_rows:
            states = pickle.loads(row[0])
            actions = pickle.loads(row[1])
            rewards = pickle.loads(row[2])
            next_states = pickle.loads(row[3])
            dones = pickle.loads(row[4])
            episode = list(zip(states, actions, rewards, next_states, dones))
            episodes.append(episode)

        return episodes

    def __len__(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM episodes")
            count = c.fetchone()[0]
        return count
