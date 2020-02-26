import numpy as np
import matplotlib.pyplot as plt

import gym
import gym.spaces as spaces


DEST = -2
AGENT = -1

COLORS = {
    0: (0, 0, 0),           # No obstacle
    1: (0.5, 0, 0),         # Obstacle
    AGENT: (1.0, 1.0, 0),   # Agent (idx: -1)
}

MOVE_REWARD = 1


class SimpleGridExplorerEnv(gym.Env):
    """
    ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/classic_envs/gridworld_env.py
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid=None, nrow=25, ncol=25, density=0.3, ctrl_error=0.05, episode_length=100, verbose=False):
        """
        Create a grid world instance.
            Initial state:  (nrow-1, ncol-1)
            Destination:    (0, alpha), where alpha is a uniform random variable.
            Reward:         -1 for each move; 1000 for reaching the destination.
            Constraint cost:1 if the current state is hazardous(has an obstacle); otherwise 0.
        :param grid: np.array, shape=(nrow, ncol,)
            If None, a new map is created randomly.
            Each element in the grid is equal to the constraint cost of each state. Destination is marked as -2.
        :param nrow: integer
        :param ncol: integer
        :param density: float in the range [0,1]
            The density of hazardous states in the map.
            If 0, the map is totally safe. As it becomes closer to 1, the map is more likely to be dangerous.
        :param ctrl_error: float in the range [0,1]
            The probability of control error for each action.
            With probability ctrl_error, the agent randomly moves to an adjacent state.
        :param episode_length: integer
            Maximum runtime (running steps).
        :param verbose: bool
            Display the observation if True.
        """
        if grid is not None:
            # Note: First row should be always safe.
            self.nrow, self.ncol = grid.shape
            grid[0, :] = 1.
        else:
            self.nrow, self.ncol = nrow, ncol

        self.grid = self._create_gridworld(density) if grid is None else np.copy(grid)
        self.density = density
        self.ctrl_error = ctrl_error
        self.episode_length = episode_length

        self.verbose = verbose
        if self.verbose:
            self.fig = plt.figure(self.fig_nb)
            plt.show(block=False)
            plt.axis('off')
            self.render()

        ##########################################################################################
        # Variables for actual gameplay
        self.state = self._initialize_state()
        self.runtime = 0
        self.online_obs = np.zeros((self.nrow, self.ncol, 3))
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.online_obs[i, j] = COLORS[self.grid[i, j]]
        self.online_obs[self.state] = COLORS[AGENT]
        self.fig_nb = 1
        ##########################################################################################

        # Set this in SOME subclasses
        self.reward_range = (MOVE_REWARD, 0)
        self.spec = None

        # Set these in ALL subclasses
        self.actions = [0, 1, 2, 3]
        self.inv_actions = [2, 3, 1, 0]
        self.action_space = spaces.Discrete(4)
        self.action_pos_dict = {
            0: (0, -1),     # Move left
            1: (1, 0),      # Move down
            2: (0, 1),      # Move right
            3: (-1, 0),     # Move up
        }
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.online_obs.shape)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = True if self.runtime > self.episode_length else False

        if done:
            move = (0, 0)
        elif np.random.random() < self.ctrl_error:
            move = self.action_pos_dict[np.floor(np.random.random() * 4)]
        else:
            move = self.action_pos_dict[action]
        next_state = self._move(self.state, move)

        self.runtime += 1

        reward = MOVE_REWARD
        info = {'safety': self.grid[self.state]}

        self._update_obs_and_state(next_state)

        return self.online_obs, reward, done, info

    def transfer(self, state):
        self._update_obs_and_state(state)
        self.render()
        return self.online_obs

    def reset(self):
        self._update_obs_and_state(self._initialize_state())
        self.runtime = 0
        self.fig_nb += 1
        self.render()
        return self.online_obs

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        """
        if mode is 'rgb_array':
            return np.copy(self.online_obs)
        elif mode is 'human':
            if not self.verbose:
                return
            if close is True:
                plt.close('all')
            img = np.copy(self.online_obs)
            fig = plt.figure(self.fig_nb)
            plt.clf()
            plt.imshow(img)
            fig.canvas.draw()
            plt.pause(0.00001)
        else:
            raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        plt.close('all')
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        try:
            if seed is not None:
                np.random.seed(seed)
        except TypeError:
            print("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def _create_gridworld(self, density):
        grid = np.ones((self.nrow, self.ncol)).astype(np.int)
        grid[1:, :] = (np.random.random((self.nrow-1, self.ncol)) > density).astype(np.int)
        return grid

    def _initialize_state(self):
        return 0, np.random.randint(0, self.ncol)
        # x, y = np.where(self.grid > 0)
        # index = np.random.randint(0, x.size)
        # return x[index], y[index]
    
    def _move(self, state, move):
        state = (state[0] + move[0], state[1] + move[1])
        return np.minimum(np.maximum(state[0], 0), self.nrow - 1), np.minimum(np.maximum(state[1], 0), self.ncol - 1)

    def _update_obs_and_state(self, next_state):
        self.online_obs[self.state] = COLORS[self.grid[self.state]]
        self.online_obs[next_state] = COLORS[AGENT]
        self.state = next_state
        return


class GridExplorerEnv(SimpleGridExplorerEnv):

    def __init__(self, grid=None, nrow=25, ncol=25, density=0.3, ctrl_error=0.05, episode_length=100, verbose=False):
        """
        Create an "explicit" grid world instance; State is given as a single scalar, transition kernel is provided.
        See the caption of GridWorldEnv for more information.
        """
        super(GridExplorerEnv, self).__init__(
            grid, nrow, ncol, density, ctrl_error, episode_length, verbose)
        self.nb_states = self.nrow * self.ncol
        self.nb_actions = 4

    def speculate(self, confidence=0., gamma=1.):
        kernel = np.zeros((self.nb_states, self.nb_actions, self.nb_states), dtype=np.float)
        for i in range(self.nrow):
            for j in range(self.ncol):
                current_state = self._coord_to_index((i, j))
                # Move left / down / right / up
                next_state_candidates =\
                    [self._coord_to_index(self._move((i, j), self.action_pos_dict[k])) for k in range(self.nb_actions)]
                for k in range(self.nb_actions):

                    # action given by controller
                    for l in range(self.nb_actions):
                        # action that actually happened
                        if k == l:
                            kernel[current_state, k, next_state_candidates[l]] += 1 - self.ctrl_error * 3 / 4
                        else:
                            kernel[current_state, k, next_state_candidates[l]] += self.ctrl_error / 4

        reward = np.ones((self.nb_states, self.nb_actions)) * MOVE_REWARD

        safety = np.ones((self.nb_states,))
        for i in range(self.nrow):
            for j in range(self.ncol):
                safety[self._coord_to_index((i, j))] = self.grid[i, j]

        initial_policy = self._construct_initial_policy()

        return self.nb_states, self.nb_actions, kernel, reward, safety, initial_policy,\
               confidence, gamma, self.episode_length

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        _, reward, done, info = super(GridExplorerEnv, self).step(action)
        info['state'] = self._coord_to_index(self.state)
        return self.online_obs, reward, done, info

    def reset(self):
        self._update_obs_and_state(self._initialize_state())
        self.runtime = 0
        self.fig_nb += 1
        self.render()
        return self.online_obs, self._coord_to_index(self.state)

    def transfer(self, state):
        super(GridExplorerEnv, self).transfer(self._index_to_coord(state))
        return self.online_obs, self._coord_to_index(self.state)

    def cheat(self):
        return self._coord_to_index(self.state)

    def _coord_to_index(self, coord):
        return coord[0] * self.ncol + coord[1]

    def _index_to_coord(self, index):
        return np.divmod(index, self.ncol)

    def _construct_initial_policy(self):
        initial_policy = np.zeros((self.nb_states, self.nb_actions))

        for i in range(self.nrow):
            for j in range(self.ncol):
                if i == 0:
                    initial_policy[self._coord_to_index((i, j)), 0] = 0.5
                    initial_policy[self._coord_to_index((i, j)), 2] = 0.5
                else:
                    initial_policy[self._coord_to_index((i, j)), 3] = 1.
                    # left:0 / down:1 / right:2 / up:3
        return initial_policy

