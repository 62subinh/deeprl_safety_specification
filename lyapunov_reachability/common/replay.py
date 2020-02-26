import numpy as np
import random
from lyapunov_reachability.common.sumtree import SumTree


class Replay(object):

    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.buffer = list()

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs_b, act_b, next_obs_b, reward_b, done_b = list(), list(), list(), list(), list()
        for sample in samples:
            obs_b.append(sample[0])
            act_b.append(sample[1])
            next_obs_b.append(sample[2])
            reward_b.append(sample[3])
            done_b.append(sample[4])
        return np.array(obs_b), np.array(act_b), np.array(next_obs_b), np.array(reward_b), np.array(done_b)

    def get(self, start, end):
        assert start < end
        obs_b, act_b, next_obs_b, reward_b, done_b = list(), list(), list(), list(), list()
        for idx in range(start, end):
            obs, action, next_obs, reward, done = self.buffer[idx]
            obs_b.append(obs)
            act_b.append(action)
            next_obs_b.append(next_obs)
            reward_b.append(reward)
            done_b.append(done)
        return np.array(obs_b), np.array(act_b), np.array(next_obs_b), np.array(reward_b), np.array(done_b)

    def shuffle(self, start, end, batch_size):
        return

    def store(self, sample: tuple, *args):
        """
        Store a sample in the buffer.
        :param sample: A tuple of (obs, action, next_obs, reward, done)
        :return:
        """
        if len(self.buffer) == self.replay_size:
            self.buffer.pop(0)
        self.buffer.append(sample)

    def clear(self):
        self.buffer.clear()

    def set_size(self, new_size):
        if len(self.buffer) > new_size:
            del self.buffer[0:(len(self.buffer) - new_size)]
        self.replay_size = new_size

    def update(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplay(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.buffer = SumTree(replay_size)

    def sample(self, batch_size):
        obs_b, act_b, next_obs_b, reward_b, done_b = list(), list(), list(), list(), list()
        idx_b = []
        priorities = []
        segment = self.buffer.total() / batch_size

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, sample) = self.buffer.get(s)

            obs_b.append(sample[0])
            act_b.append(sample[1])
            next_obs_b.append(sample[2])
            reward_b.append(sample[3])
            done_b.append(sample[4])

            priorities.append(p)
            idx_b.append(idx)

        sampling_probabilities = np.array(priorities) / self.buffer.total()
        is_weight = np.power(self.buffer.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return np.array(obs_b), np.array(act_b), np.array(next_obs_b), np.array(reward_b), np.array(done_b),\
               np.array(idx_b), is_weight

    def store(self, sample: tuple, *args):
        """
        Store a sample in the buffer.
        :param sample:  A tuple of (obs, action, next_obs, reward, done)
        :param args:    "error"
        :return:
        """
        error = 1. if len(args) == 0 else args[0]
        p = self._get_priority(error)
        self.buffer.store(p, sample)

    def clear(self):
        self.buffer.clear()

    def set_size(self, new_size):
        self.buffer.set_size(new_size)
        self.replay_size = new_size

    def update(self, idx, error):
        p = self._get_priority(error)
        self.buffer.update(idx, p)

    def __len__(self):
        return len(self.buffer)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
