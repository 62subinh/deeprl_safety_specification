import os
import gym
import math
import random
import pickle
import abc
import numpy as np
from collections import namedtuple, deque
from itertools import count, product
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter

from lyapunov_reachability.common.utils import init_weights
from lyapunov_reachability.common.networks import Mlp, Cnn
from lyapunov_reachability.common.replay import Replay, PrioritizedReplay

EPS = 1e-8

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousBase(object):

    def __init__(
            self, env, confidence, extractor=None, extractor_params=None, decoder=None, decoder_params=None,
            seed=None, lr=1e-3, batch_size=128, gamma=0.999, ob_side=32, grid_points=None, strict_done=False,
            replay_size=int(1e5), replay_prioritized=False, baseline_dir=None, baseline_step=None,
            re_init=False, save_dir='../../spec-ddpg-base'):
        # Neural networks & optimizers should be declared before. For example,
        # self.q = None
        # self.q_optimizer = None

        self.env = env

        if len(env.observation_space.shape) == 1:
            self.ob_space = env.observation_space.shape
            self.transform = torch.FloatTensor
        elif len(env.observation_space.shape) == 3:
            self.ob_space = (env.observation_space.shape[-1], ob_side, ob_side)
            self.transform = T.Compose([
                T.ToPILImage(), T.Resize(self.ob_space[1], interpolation=Image.CUBIC), T.ToTensor()])
        else:
            raise ValueError("Observation space should be 1D or 3D.")

        # self.ob_space = [defined later]           # Tuple, shape of the observation space.
        self.ac_space = len(env.action_space.low)   # Int, dimension of the action space.

        # Used for action (de)normalization
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

        # Learning hyperparameters
        self.seed = seed
        if seed is not None:
            env.seed(seed)
            torch.manual_seed(seed)

        self.lr = lr                        # learning rate,
        self.batch_size = batch_size        # batch size,
        self.confidence = confidence        # safety confidence,
        self.gamma = gamma                  # discount factor,
        self.steps = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = os.path.abspath(save_dir)
        self.grid_points = grid_points
        self.strict_done = strict_done

        # Replay
        self.replay = None
        self.replay_size = replay_size
        self.replay_prioritized = replay_prioritized

        # Record episodic information.
        self.record_step = [0]
        self.record_safety = [0.]

        # Create neural networks & optimizers, then initialize.
        self.setup_model(extractor, extractor_params, decoder, decoder_params)
        if not re_init:
            self.setup_optimizer(baseline_dir=baseline_dir, baseline_step=baseline_step)
            self.setup_replay(baseline_dir=baseline_dir, baseline_step=baseline_step)
            self.save_setup(extractor, extractor_params, decoder, decoder_params)

    @abc.abstractmethod
    def setup_model(self, extractor=None, extractor_params=None, decoder=None, decoder_params=None):
        """
        Construct neural networks here.
        e.g.
            self.actor = Actor(self.ob_space, self.ac_space, decoder=decoder, decoder_params=decoder_params)
            self.critic = Critic(self.ob_space, self.ac_space, extractor, extractor_params,
                                 decoder=decoder, decoder_params=decoder_params)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def setup_optimizer(self, reload=False, baseline_dir=None, baseline_step=None):
        """
        Initialize weights, load them on GPU, and construct optimizers.
        If baseline is provided, initialize with that baseline.
        e.g.
            init_weights(self.actor)
            init_weights(self.critic)
            --
            self.actor.cuda(device)
            self.critic.cuda(device)
            --
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_network(self, *args):
        """
        From the batches (provided as arguments), update the network.
        e.g.
            loss = F.mse_loss(q_t, q_t_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        :param obs_t:   Observation at time t.
        :param act_t:   Action taken at time t, drawn from exploratory policy.
        :param obs_tp1: Observation at time t+1.
        :param reached: 1. (True) if the agent ever reached target at time interval (0, t].
        :param done:    1. (True) if obs_t is the final state of the trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def target_update(self, **kwargs):
        """
        Update the "target" networks if necessary.
        e.g. hard_target_update:
            for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
                target_param.data.copy_(param.data)

        e.g. soft_target_update:
            for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
                target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        """
        raise NotImplementedError

    def setup_replay(self, reload=False, baseline_dir=None, baseline_step=None):
        if self.replay_prioritized:
            self.replay = PrioritizedReplay(self.replay_size)
        else:
            self.replay = Replay(self.replay_size)
        self.replay_prioritized = self.replay_prioritized

    def store_sample(self, obs, action, next_obs, reach, done):
        """
        Store the sample in the replay.
        :param obs:     Observation at time t.
        :param action:  Action taken at time t, drawn from exploratory policy.
        :param next_obs:Observation at time t+1.
        :param reach:   1. (True) if the agent ever reach target at time interval (0, t].
        :param done:    1. (True) if obs_t is the final state of the trajectory.
        :return:        batches (obs_b, act_b, next_obs_b, reward_b, reach_b, done_b)
        """
        # Normalize action first.
        action = (action - self.act_low) / (self.act_high - self.act_low) * 2. - 1.

        # Store the sample in both replay & trajectory buffer,
        self.replay.store((obs, action, next_obs, reach, done))
        return

    def close_episode(self):
        """
        Do episode-wise activity here if necessary.
        """
        return

    @abc.abstractmethod
    def step(self, obs, *args):
        """
        Fetch an action to operate the agent in environment.
        e.g.
            with torch.no_grad():
                act = self.actor.step(self.transform(obs).unsqueeze(0).to(device))
                return act.data.cpu().numpy()[0]

        :param obs: Observation, not batched.
        :return: Action, not batched.
        """
        raise NotImplementedError

    def run(self, steps, episode_length, log_interval=None, save_interval=None, **extra_args):
        extra_args['episode_length'] = episode_length

        # Set intervals
        if log_interval is None:
            log_interval = episode_length
        if save_interval is None:
            save_interval = episode_length

        # Set Tensorboard
        log_dir = os.path.abspath(self.save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        # Set variables for logging
        episode_step = 0
        episode_safety = 1.

        # First reset
        obs = self.env.reset()
        # extra_state = 1

        for t in range(1, steps+1):

            # [1] Fetch a sample state transition tuple.
            action = self.step(obs, t, steps)
            next_obs, _, done, info = self.env.step(action)

            # [2] Record episodic information.
            episode_step += 1
            episode_timeout = episode_step == episode_length
            episode_safety *= info['safety']# * (not episode_timeout)

            done = (done + (1. - self.strict_done) * episode_timeout > 0.) * 1.
            self.store_sample(obs, action, next_obs, 1. - episode_safety, done)

            # [3] Reset the environment if the agent visits the target set for the first time.
            if episode_safety == 0. or done or episode_timeout:
                obs = self.env.reset()
                # extra_state = 1
                self.record_step[0:0] = [episode_step]
                episode_step = 0
                self.record_step = self.record_step[:100]
                self.record_safety[0:0] = [episode_safety]
                episode_safety = 1.
                self.record_safety = self.record_safety[:100]
                self.close_episode()
            else:
                obs = next_obs
                # extra_state = extra_state * (safety > 0.)

            # [4] Update networks & target update
            self.update_network(t, episode_length, log_interval, save_interval)
            self.target_update(**extra_args)

            self.steps += 1

            # [5] Do Logging
            if t % log_interval == 0:
                self.print_log(summary_writer)
            if t % save_interval == 0:
                self.save_weights()
                if self.grid_points is not None:
                    np.savez(os.path.join(
                        self.save_dir, '{}-reachability-map.npz'.format(self.steps)), self.sweep_states())
                else:
                    np.savez(os.path.join(
                        self.save_dir, '{}-safe-samples.npz'.format(self.steps)), self.examine_samples())
        print('done.')
        self.save_weights()
        return

    # Methods about logging ----
    def print_log(self, summary_writer):
        """
        Add summary elements in Tensorboard log.
        :param summary_writer: A tensorboardX.SummaryWriter instance.
        """
        average_safety = np.mean(self.record_safety)
        average_step = np.mean(self.record_step)
        summary_writer.add_scalar('train/average_safety', average_safety, self.steps)
        summary_writer.add_scalar('train/average_step', average_step, self.steps)
        print('log\t\t\t:: episode_safety={}, episode_runtime={}'.format(average_safety, average_step))

    def verify_state(self, obs):
        """
        Fetch the value of reachability.
        e.g.
            with torch.no_grad():
                obs_b = self.transform(obs).unsqueeze(0).to(device)
                act = self.actor.step(obs_b)
                reachability = self.critic(obs_b, F.one_hot(act, self.ac_space).squeeze(1).float())
                return reachability.data.cpu().numpy()[0]
        :param obs: Observation, not batched.
        :return: Reachability, not batched.
        """
        raise NotImplementedError

    def sweep_states(self):
        """
        Discretize the state space with equal interval, and get reachability for each grid point.
        The parameter "grid_points" should be provided in self.__init__(..).
        There are 3 difference cases for sweep:
            (A) If env has "sample_states" method, get the list of observations to verify and compute reachability for
                each observation.
                e.g. LunarLanderEnv()
            (B) If env has "state_space" (i.e. state space is different from observation, and it's difficult to
                quantize the observation space), get all the grid points in the state space and so on.
                e.g. VanillaPendulumEnv()
            (C) If env does not have "state_space" (i.e. observation is state), get all the grid points in the
                observation space and so on.
        :return: The list of reachability at different states (1-D numpy array)
        """
        grid_points = self.grid_points
        if hasattr(self.env, 'sample_states'):
            sts = self.env.sample_states(grid_points=grid_points)
            reachability_chart = np.zeros((sts.shape[0],))
            for idx in range(sts.shape[0]):
                reachability_chart[idx] = self.verify_state(sts[idx, :])
            print('*** chart\t:: safe_set={}'.format(np.mean(reachability_chart <= 1. - self.confidence)))
            return reachability_chart
        elif hasattr(self.env, 'state_space'):
            st = self.env.state_space.shape[0]
            reachability_chart = np.zeros((int(np.power(grid_points, st)),))
            low = self.env.state_space.low
            high = self.env.state_space.high
            points = range(0, grid_points)
            numeral = np.array([int(grid_points ** n) for n in reversed(range(st))])
            for idx in product(points, repeat=st):
                state = (high - low) * np.array(idx) / grid_points + low
                reachability_chart[np.sum(numeral * np.array(idx))] = self.verify_state(self.env.convert(state))
            print('*** chart\t:: safe_set={}'.format(np.mean(reachability_chart <= 1. - self.confidence)))
            return reachability_chart
        elif len(self.ob_space) == 1:
            reachability_chart = np.zeros((int(np.power(grid_points, self.ob_space[0])),))
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            points = range(0, grid_points)
            numeral = np.array([int(grid_points ** n) for n in reversed(range(self.ob_space[0]))])
            for idx in product(points, repeat=self.ob_space[0]):
                obs = (high - low) * np.array(idx) / grid_points + low
                reachability_chart[np.sum(numeral * np.array(idx))] = self.verify_state(obs)
            print('*** chart\t:: safe_set={}'.format(np.mean(reachability_chart <= 1. - self.confidence)))
            return reachability_chart
        else:
            print('Not Implemented.')
            return None

    def examine_samples(self, size=10000):
        """
        Save the samples in replay buffer.
        :return: Mean & variance of collected samples.
        """
        obs_samples, safety_samples = list(), list()
        size = np.minimum(size, len(self.replay))
        if size < 1:
            print('*** chart\t:: insufficient samples.')
        for t in range(1, size+1):
            obs, _, _, safety, _ = self.replay.buffer[-t]
            obs_samples.append(obs)
            safety_samples.append(safety)
        obs_samples, safety_samples = np.array(obs_samples), np.array(safety_samples)
        valid_samples = obs_samples[safety_samples == 1.]
        print('*** chart\t:: std of safe states={}'.format(np.std(valid_samples)))
        return valid_samples

    @abc.abstractmethod
    def save_setup(self, extractor, extractor_params, decoder, decoder_params, save_env=False):
        """
        e.g.
            if save_env:
                with open(os.path.join(self.save_dir, 'env.pkl'), 'wb') as f:
                    pickle.dump(self.env, f)

            data = {
                # Arguments of __init__() and other necessary information.
            }

            with open(os.path.join(self.save_dir, "params.pkl".format(self.steps)), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        """
        raise NotImplementedError

    @classmethod
    def load(cls, load_dir, steps, env=None, **new_kwargs):
        data = None

        def network_finder(class_name):
            if class_name == "Mlp":
                return Mlp
            elif class_name == "Cnn":
                return Cnn
            else:
                return None

        try:
            with open(os.path.join(load_dir, 'params.pkl'), 'rb') as f:
                data = pickle.load(f)

            if env is None:
                with open(os.path.join(load_dir, 'env.pkl'), 'rb') as f:
                    env = pickle.load(f)

            print("******************** Loading the saved model ********************")
            for key in data.keys():
                print("{}: {}".format(key, data[key]))
            print("*****************************************************************")

            data['extractor'] = network_finder(data.get('extractor', None))
            data['decoder'] = network_finder(data.get('decoder', None))

        except FileNotFoundError or UnicodeDecodeError:
            print('Unable to restore parameters/environment.')

        confidence = data.pop('confidence')

        model = cls(env, confidence, **data, re_init=True, save_dir=load_dir)
        model.__dict__.update({'steps': steps})
        model.__dict__.update(**new_kwargs)
        model.load_weights(load_dir)
        model.setup_optimizer(reload=True)
        model.setup_replay(reload=True)
        return model

    @abc.abstractmethod
    def save_weights(self):
        """
        Save network weights.
        e.g.
            torch.save(self.actor.state_dict(), os.path.join(self.save_dir, "{}-actor".format(self.steps)))
            torch.save(self.critic.state_dict(), os.path.join(self.save_dir, "{}-critic".format(self.steps)))
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_weights(self, load_dir):
        """
        Load network weights. Don't forget to call eval().
        e.g.
            self.actor.load_state_dict(torch.load(os.path.join(load_dir, "{}-actor".format(self.steps))))
            self.critic.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic".format(self.steps))))

            self.actor.eval()
            self.critic.eval()
        :param load_dir: Directory where the weights are saved.
        """
        raise NotImplementedError

    # Hidden methods for drawing batches ----
    def _draw_batch(self, *args):
        """
        Draw a minibatch from the replay.
        :return:    batches (obs_b, act_b, next_obs_b, reach_b, done_b)
        """
        obs_b, act_b, next_obs_b, reached_b, done_b = self.replay.sample(self.batch_size)

        obs_b = self.transform(obs_b).to(device)
        act_b = torch.FloatTensor(act_b).to(device)
        next_obs_b = self.transform(next_obs_b).to(device)
        reached_b = torch.FloatTensor(reached_b).to(device)
        done_b = torch.FloatTensor(done_b).to(device)
        return obs_b, act_b, next_obs_b, reached_b, done_b

    def _draw_batch_prioritized(self, *args):
        """
        Draw a minibatch from the replay.
        :return:    batches (obs_b, act_b, next_obs_b, reach_b, done_b), and corresponding indices, and weight masks.
        """
        obs_b, act_b, next_obs_b, reached_b, done_b, idx_b, weight_mask_b = self.replay.sample(self.batch_size)

        obs_b = self.transform(obs_b).to(device)
        act_b = torch.FloatTensor(act_b).to(device)
        next_obs_b = self.transform(next_obs_b).to(device)
        reached_b = torch.FloatTensor(reached_b).to(device)
        done_b = torch.FloatTensor(done_b).to(device)
        weight_mask_b = torch.FloatTensor(weight_mask_b).to(device)
        return obs_b, act_b, next_obs_b, reached_b, done_b, idx_b, weight_mask_b
