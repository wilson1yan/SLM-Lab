from os.path import join
import numpy as np

class DummySequReplay(object):

    def __init__(self, game, clip_reward=True, seq_len=10):
        self.total_reward = 0
        self.clip_reward = clip_reward
        self.seq_len = seq_len
        self.episode_intervals, self.data = self.load_episodes(data_path, game)

        valid_seq_idx_ranges = list()
        for start, end in self.episode_intervals:
            if end - start + 1 < self.seq_len:
                continue
            valid_seq_idx_ranges.append((0, end - self.seq_len + 2))
        self.valid_seq_idx_ranges = valid_seq_idx_ranges

        total = sum([end - start + 1 for start, end in self.valid_seq_idx_ranges])
        self.valid_seq_idx_weights = [(end - start + 1) / total
                                      for start, end in self.valid_seq_idx_ranges]

    def load_data(self, data_path, game):
        states = np.load(join(data_path, "{}_states.npy".format(game)))
        actions = np.load(join(data_path, "{}_actions.npy".format(game)))
        rewards = np.load(join(data_path, "{}_rewards.npy".format(game)))
        dones = np.load(join(data_path, "{}_dones.npy".format(game)))

        dones[-1] = True

        return states, actions, rewards, dones

    def load_episodes(self, data_path, game):
        states, actions, rewards, dones = self.load_data(data_path, game)
        done_idxs = np.argwhere(dones).reshape(-1)
        episode_intervals = [(0, done_idxs[0])]
        for i in range(len(done_idxs) - 1):
            episode_intervals.append((done_idxs[i]+1, done_idxs[i+1]))

        if done_idxs[-1] != len(dones) - 1:
            episode_intervals.append((done_idxs[-1], len(dones)-1))
        return episode_intervals, (states, actions, rewards, dones)

    def sample(self):
        batch_size = self.batch_size
        states, actions, rewards, dones = self.data
        valid_seq_idx_ranges = self.valid_seq_idx_ranges
        valid_seq_idx_weights = self.valid_seq_idx_weights

        range_idx = np.random.choice(np.arange(len(valid_seq_idx_ranges)), size=batch_size,
                                     p=valid_seq_idx_weights)
        range_idx = [valid_seq_idx_ranges[i] for i in range_idx]
        batch_idx = [np.random.randint(start, end + 1)
                     for start, end in range_idx]

        state_batch = np.concatenate([states[i:i + self.seq_len]
                                      for i in batch_idx])
        action_batch = np.concatenate([actions[i:i + self.seq_len]
                                       for i in batch_idx]).astype(np.uint8)
        reward_batch = np.concatenate([rewards[i:i + self.seq_len]
                                       for i in batch_idx]).astype(np.float32)
        if self.clip_reward:
            reward_batch = np.sign(reward_batch)

        dones_batch = np.concatenate([dones[i:i + self.seq_len]
                                      for i in batch_idx])

        next_idxs = np.array([idx + 1 if not dones[idx] else idx for idx in batch_idx])
        next_state_batch = np.concatenate([states[i:i + self.seq_len]
                                           for i in next_idxs])

        return {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'dones': dones_batch,
            'next_states': next_state_batch
        }

    def reset(self):
        pass

    def epi_reset(sel,f state):
        pass

    @lab_api
    def update(self, action, reward, state, done):
        pass

    def add_experience(self, state, action, reward, next_state, done):
        pass

    @lab_api
    def sample(self):
        pass

    def sample_idxs(self, batch_size):
        pass
