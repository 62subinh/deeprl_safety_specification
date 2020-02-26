import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from itertools import product
from os import path


class DoubleIntegratorEnv(gym.Env):

    def __init__(self, seed):
        """
        Observation(=state):
            Two states x1, x2 (location, velocity)
        Action:
            Single input u, restricted to the interval [-1, 1].
        Dynamics:
            dx1/dt = x2, dx2/dt = u
        """
        super(DoubleIntegratorEnv, self).__init__()

        self.dt = .05
        self.state = None
        self.np_random = None
        self.last_u = None
        self.viewer = None

        bound = np.array([1., .5])
        self.action_space = spaces.Box(low=np.array([-.5]), high=np.array([.5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float32)

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, st):
        self.state = np.clip(np.array(st), self.observation_space.low, self.observation_space.high)
        self.last_u = None
        return self._get_obs()

    def step(self, u):
        x1, x2 = self.state
        dt = self.dt

        u = np.clip(u, -.5, .5)
        self.last_u = u  # for rendering

        new_x2 = dt * u *  + x2
        new_x1 = dt * x2 + x1

        cost = .5 * (x1 ** 2) + .5 * (x2 ** 2)
        info = {'safety': self._get_safety()}

        # Transition to next state
        self.state[:] = np.array([new_x1, new_x2])
        return self._get_obs(), -cost, self._get_done(), info

    def reset(self):
        r = self.np_random.uniform(low=0.40, high=0.60)
        theta = self.np_random.uniform(low=0., high=2*np.pi)
        self.state = np.array([r * np.cos(theta), r * np.sin(theta)]) * self.observation_space.high
        self.last_u = None
        return self._get_obs()

    def _get_safety(self):
        return 1. * np.prod(np.abs(self.state) < self.observation_space.high)

    def _get_done(self):
        return np.all(np.abs(self.state) <= self.observation_space.high * np.array([0.2, 0.075]))

    def _get_obs(self):
        return np.array(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 250)
            self.viewer.set_bounds(-2.0, 2.0, -1.0, 1.0)

            agent = rendering.make_capsule(.6, .8)
            agent.set_color(.8, .3, .3)
            self.agent_transform = rendering.Transform()
            agent.add_attr(self.agent_transform)

            self.viewer.add_geom(agent)

        self.agent_transform.set_translation(self.state[0], 0.)
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def convert(state):
        x, xdot = state
        return np.array([x, xdot,])


class DiscDoubleIntegratorEnv(DoubleIntegratorEnv):
    def __init__(self, n=21, grid_points=None, seed=None):
        super(DiscDoubleIntegratorEnv, self).__init__(seed)
        self.n = n
        try:
            self.grid_points = int(grid_points)
            self.numeral = np.array([int(grid_points ** n) for n in reversed(range(2))])
        except ValueError:
            self.grid_points = None
            self.numeral = None
        self.action_space = spaces.Discrete(n)

    def quantize(self, st):
        coordinate = np.round(
            (st - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
            * (self.grid_points - 1))
        return np.sum(self.numeral * coordinate.astype(np.int))

    def set_state(self, st_discrete):
        st = self.observation_space.sample()
        for idx, num in enumerate(self.numeral):
            st[idx] = st_discrete // num
            st_discrete = st_discrete % num
        st = st * (self.observation_space.high - self.observation_space.low) / (self.grid_points - 1.) +\
            self.observation_space.low
        return super(DiscDoubleIntegratorEnv, self).set_state(st)

    def step(self, u_idx):
        u = np.clip(u_idx, 0, self.action_space.n - 1) / (self.action_space.n - 1.) * 2. - 1.
        next_obs, reward, done, info = super(DiscDoubleIntegratorEnv, self).step(u)
        if self.grid_points is not None:
            info['state'] = self.quantize(self.state)
        return next_obs, reward, done, info

    def speculate(self, grid_points=21, confidence=0., gamma=1., episode_length=None):
        obs_dim = len(self.observation_space.low)
        nb_states = int(grid_points ** obs_dim)
        nb_actions = self.action_space.n

        low = self.observation_space.low
        high = self.observation_space.high
        numeral = np.array([int(grid_points ** n) for n in reversed(range(obs_dim))])

        def from_index(idx):
            state = list()
            for n in range(obs_dim - 1):
                state.append(idx // numeral[n])
                idx = idx % numeral[n]
            state.append(idx)
            return (high - low) * np.array(state) / (grid_points - 1.) + low

        def from_obs(obs):
            coordinate = np.round((obs - low) / (high - low) * (grid_points - 1))
            return np.sum(numeral * coordinate.astype(np.int))

        def interpolate_obs(obs):
            coordinate = (obs - low) / (high - low) * (grid_points - 1)
            floored = np.clip(np.floor(coordinate), 0, grid_points - 1)
            ceiled = np.clip(np.ceil(coordinate), 0, grid_points - 1)
            valid_grid_points = list()
            probabilities = list()
            for pt in product((False, True), repeat=obs_dim):
                pt = np.array(pt)
                valid_grid_points.append(
                    int(np.sum((numeral * floored)[pt]) + np.sum((numeral * ceiled)[~pt]))
                )
                probabilities.append(np.prod((coordinate - floored)[pt]) * np.prod((ceiled - coordinate)[~pt]))
            return np.array(valid_grid_points), np.array(probabilities)

        kernel = np.zeros((nb_states, nb_actions, nb_states), dtype=np.float)
        for s_t in range(nb_states):
            for a_t in range(nb_actions):
                # Copy & paste dynamics here. -- instead of observation, we discretize state.
                self.state[:] = from_index(s_t)
                _ = self.step(a_t)

                s_tp1s, probs = interpolate_obs(self.state)
                kernel[s_t, a_t, s_tp1s] = probs

        # Copy & paste safety criteria here.
        safety = np.ones((nb_states,))
        reward = np.ones((nb_states, nb_actions))
        is_terminal = np.ones((nb_states,), dtype=np.bool)
        for s_t in range(nb_states):
            self.state[:] = from_index(s_t)
            safety[s_t] = self._get_safety()
            # reward[s_t] = 1.
            is_terminal[s_t] = done = self._get_done()
            if done:
                # Kernel, reward, safety of terminal states
                kernel[s_t, ...] = 0.
                kernel[s_t, :, s_t] = 1.
                safety[s_t] = 1.
        terminal_states = np.where(is_terminal)[0]

        if episode_length is None:
            episode_length = 1000

        initial_policy = np.ones((nb_states, nb_actions)) * 1. / nb_actions

        return nb_states, nb_actions, kernel, reward, safety, initial_policy,\
               confidence, gamma, None, terminal_states, episode_length

    def speculate_light(self, grid_points=21):
        grid_points = self.grid_points if self.grid_points is not None else grid_points
        obs_dim = len(self.observation_space.low)
        nb_states = int(grid_points ** obs_dim)
        nb_actions = self.action_space.n

        low = self.observation_space.low
        high = self.observation_space.high
        numeral = np.array([int(grid_points ** n) for n in reversed(range(obs_dim))])

        def from_index(idx):
            state = list()
            for n in range(obs_dim - 1):
                state.append(idx // numeral[n])
                idx = idx % numeral[n]
            state.append(idx)
            return (high - low) * np.array(state) / grid_points + low

        # Copy & paste safety criteria here.
        is_terminal = np.ones((nb_states,), dtype=np.bool)
        for s_t in range(nb_states):
            self.state[:] = from_index(s_t)
            is_terminal[s_t] = self._get_done()
        terminal_states = np.where(is_terminal)[0]

        initial_policy = np.ones((nb_states, nb_actions)) * 1. / nb_actions

        return nb_states, nb_actions, initial_policy, terminal_states
