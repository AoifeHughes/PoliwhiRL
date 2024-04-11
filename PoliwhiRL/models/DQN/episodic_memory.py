# -*- coding: utf-8 -*-
import os
import pickle
import sqlite3
import numpy as np
import multiprocessing as mp

class EpisodicMemory:
    def __init__(self, memory_size, db_path="./database/episodic_memory.db", parallel=False):
        self.memory_size = memory_size
        self.db_path = db_path
        self.current_episode = []
        self.parallel = parallel
        
        if parallel:
            self.lock = mp.Lock()
            self.partial_sequences = {}
        
        self._create_table()

    def _create_table(self):
        db_folder = os.path.dirname(self.db_path)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)

        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS episodes
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          states BLOB,
                          actions BLOB,
                          rewards BLOB,
                          next_states BLOB,
                          dones BLOB)""")
            conn.commit()

    def _get_connection(self):
        if self.parallel:
            return sqlite3.connect(self.db_path, check_same_thread=False)
        else:
            return sqlite3.connect(self.db_path)

    def add(self, state, action, reward, next_state, done, worker_id=None):
        if self.parallel:
            if worker_id not in self.partial_sequences:
                self.partial_sequences[worker_id] = []

            self.partial_sequences[worker_id].append((state, action, reward, next_state, done))

            if done:
                with self.lock:
                    self._add_episode(self.partial_sequences[worker_id])
                    self.partial_sequences[worker_id] = []
        else:
            self.current_episode.append((state, action, reward, next_state, done))
            if done:
                self._add_episode(self.current_episode)
                self.current_episode = []

    def _add_episode(self, episode):
        states, actions, rewards, next_states, dones = zip(*episode)
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO episodes (states, actions, rewards, next_states, dones)
                         VALUES (?, ?, ?, ?, ?)""",
                      (pickle.dumps(states), pickle.dumps(actions), pickle.dumps(rewards),
                       pickle.dumps(next_states), pickle.dumps(dones)))
            conn.commit()

        # Limit the number of stored episodes to memory_size
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
                      (self.memory_size,))
            conn.commit()

    def sample(self, batch_size):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT states, actions, rewards, next_states, dones FROM episodes ORDER BY RANDOM() LIMIT ?",
                      (batch_size,))
            rows = c.fetchall()

        episodes = []
        for row in rows:
            states = pickle.loads(row[0])
            actions = pickle.loads(row[1])
            rewards = pickle.loads(row[2])
            next_states = pickle.loads(row[3])
            dones = pickle.loads(row[4])
            episode = (states, actions, rewards, next_states, dones)
            episodes.append(episode)

        return episodes

    def __len__(self):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM episodes")
            count = c.fetchone()[0]
        return count