from os.path import join
import numpy as np
import torch

from slm_lab.lib import util
from slm_lab.lib.decorator import lab_api

class DummySeqReplay(object):

    def __init__(self, memory_spec, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'seq_len',
            'game'
        ])

        self.total_reward = 0
        self.clip_reward = clip_reward
        self.seq_len = seq_len

        data_folder = join('data', 'experience', self.game)
        self.episode_intervals, self.data = self.load_episodes(data_folder, self.game)

        valid_seq_idx_ranges = list()
        for start, end in self.episode_intervals:
            if end - start + 1 < self.seq_len:
                continue
            valid_seq_idx_ranges.append((0, end - self.seq_len + 2))
        self.valid_seq_idx_ranges = valid_seq_idx_ranges

        total = sum([end - start + 1 for start, end in self.valid_seq_idx_ranges])
        self.valid_seq_idx_weights = [(end - start + 1) / total
                                      for start, end in self.valid_seq_idx_ranges]

        self.total_reward = 0
        self.is_episodic = False

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

    @lab_api
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

    @lab_api
    def update(self, action, reward, state, done):
        pass

class DummyReplay(object):

    def __init__(self, memory_spec, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'include_time_dim',
            'stack_len',
            'game'
        ])
        data_folder = join('data', 'experience', self.game)
        self.data = self.load_data(data_folder, self.game)

        states, _, _, dones = self.data
        frame_valid = np.zeros((len(states), self.stack_len))
        for i in range(len(states)):
            frame_valid[i, self.stack_len - 1] = 1
            for j in range(1, self.stack_len):
                if i - j < 0 or dones[i - j]:
                    break
                frame_valid[i, self.stack_len - 1 - j] = 1
        frame_valid = frame_valid.astype(np.uint8)
        self.frame_valid = frame_valid

        self.total_reward = 0
        self.is_episodic = False

    def load_data(self, data_path, game):
        states = np.load(join(data_path, "{}_states.npy".format(game)))
        actions = np.load(join(data_path, "{}_actions.npy".format(game)))
        rewards = np.load(join(data_path, "{}_rewards.npy".format(game)))
        dones = np.load(join(data_path, "{}_dones.npy".format(game)))

        dones[-1] = True

        return states, actions, rewards, dones

    @lab_api
    def sample(self):
        batch_size = self.batch_size
        states, actions, rewards, dones = self.data
        frame_valid = self.frame_valid

        batch_idx = np.random.randint(0, len(states), size=batch_size)

        state_batch = np.concatenate([states[batch_idx - i]
                                      for i in range(self.stack_len - 1, -1, -1)], axis=1)
        state_blanks = frame_valid[batch_idx]
        state_batch *= state_blanks.reshape(state_blanks.shape + (1, 1))

        if self.include_time_dim:
            action_batch = np.stack([actions[batch_idx - i]
                                     for i in range(self.stack_len - 1, -1, -1)], axis=1)
            reward_batch = np.stack([rewards[batch_idx - i]
                                     for i in range(self.stack_len - 1, -1, -1)], axis=1)
            dones_batch = np.stack([dones[batch_idx - i]
                                    for i in range(self.stack_len - 1, -1, -1)], axis=1)

            action_batch = action_batch.reshape(-1).astype(np.uint8)
            reward_batch = reward_batch.reshape(-1).astype(np.float32)
            dones_batch = dones_batch.reshape(-1)
        else:
            action_batch = actions[batch_idx].astype(np.uint8)
            reward_batch = rewards[batch_idx].astype(np.float32)
            dones_batch = dones[batch_idx]

        reward_batch = np.sign(reward_batch)

        next_idxs = np.array([idx + 1 if not dones[idx] else idx for idx in batch_idx])
        next_state_batch = np.concatenate([states[next_idxs - i]
                                           for i in range(self.stack_len - 1, -1, -1)], axis=1)
        next_state_blanks = frame_valid[next_idxs]
        next_state_batch *= next_state_blanks.reshape(next_state_blanks.shape + (1, 1))

        if self.include_time_dim:
            state_batch = state_batch.reshape((batch_size * self.stack_len, 1,
                                               state_batch.shape[2], state_batch.shape[3]))
            next_state_batch = next_state_batch.reshape((batch_size * self.stack_len, 1,
                                                         next_state_batch.shape[2],
                                                         next_state_batch.shape[3]))

        state_batch = np.random.rand(self.batch_size, 4, 84, 84)
        next_state_batch = np.random.rand(self.batch_size, 4, 84, 84)

        return {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'dones': dones_batch,
            'next_states': next_state_batch
        }

    @lab_api
    def update(self, action, reward, state, done):
        pass
