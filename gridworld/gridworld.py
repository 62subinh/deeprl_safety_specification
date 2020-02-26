import numpy as np
import matplotlib.pyplot as plt

import gym
import gym.spaces as spaces
from gym.utils import seeding


DEST = -2
AGENT = -1

COLORS = {
    0: (0, 0, 0),           # No obstacle
    1: (0.5, 0, 0),         # Obstacle
    DEST: (0, 0.5, 1.0),    # Destination (idx: -2)
    AGENT: (1.0, 1.0, 0),   # Agent (idx: -1)
}

MOVE_REWARD = -1


class SimpleGridWorldEnv(gym.Env):
    """
    ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/classic_envs/gridworld_env.py
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid=None, nrow=25, ncol=25, goal_length=1,
                 density=0.3, ctrl_error=0.05, episode_length=100, verbose=False):
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
            self.nrow, self.ncol = grid.shape
        else:
            self.nrow, self.ncol = nrow, ncol

        self.start = (self.nrow - 1, self.ncol - 1)

        goal_length = np.maximum(int(goal_length), 1)
        self.goals = list()
        for d in range(goal_length):
            self.goals.append((0, self.ncol - 1 - d))

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
        self.state = self.start
        self.runtime = 0
        self.display_background = np.zeros((self.nrow, self.ncol, 3))
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.display_background[i, j] = COLORS[self.grid[i, j]]
        for goal in self.goals:
            self.display_background[goal] = COLORS[DEST]
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
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,))
        
        self.seed()

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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
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
        if self.state in self.goals:
            move = (0, 0)
        elif np.random.random() < self.ctrl_error:
            move = self.action_pos_dict[np.floor(np.random.random() * 4)]
        else:
            move = self.action_pos_dict[action]
        next_state = self._move(self.state, move)

        self.runtime += 1

        if next_state in self.goals:
            reward = 0
            done = True
        else:
            reward = MOVE_REWARD
            done = False#True if self.runtime > self.episode_length else False
        info = {'safety': self.grid[self.state]}

        self.state = next_state
        return self._observe(), reward, done, info

    def forced_step(self, state):
        self.state = state
        return self._observe()

    def reset(self):
        self.state = self.start
        self.runtime = 0
        self.fig_nb += 1
        return self._observe()

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
        display = np.copy(self.display_background)
        display[self.state] = COLORS[AGENT]

        if mode is 'rgb_array':
            return (255. * display).astype(np.uint8)
        elif mode is 'human':
            if not self.verbose:
                return (255. * display).astype(np.uint8)
            if close is True:
                plt.close('all')
            fig = plt.figure(self.fig_nb)
            plt.clf()
            plt.imshow(display)
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
        grid = np.ones((self.nrow, self.ncol), dtype=np.int)
        _density = density * (self.nrow * self.ncol) / (self.nrow * self.ncol - 2 * self.ncol - self.nrow + 2)
        grid[1:(self.nrow-1), 1:] = (np.random.random((self.nrow-2, self.ncol-1)) > _density).astype(np.int)
        grid[self.start] = 1
        for goal in self.goals:
            grid[goal] = 1
        return grid

    def _move(self, state, move):
        next_row = state[0] + move[0]
        next_col = state[1] + move[1]
        if next_row < 0:
            next_row = 1
        elif next_row > self.nrow - 1:
            next_row = self.nrow - 2

        if next_col < 0:
            next_col = 1
        elif next_col > self.ncol - 1:
            next_col = self.ncol - 2

        return next_row, next_col
    
    def _observe(self):
        return np.array([1. * self.state[0] / self.nrow,
                         1. * self.state[1] / self.ncol])


class GridWorldEnv(SimpleGridWorldEnv):

    def __init__(self, grid=None, nrow=25, ncol=25, goal_length=1,
                 density=0.3, ctrl_error=0.05, episode_length=100, verbose=False):
        """
        Create an "explicit" grid world instance; State is given as a single scalar, transition kernel is provided.
        See the caption of GridWorldEnv for more information.
        """
        super(GridWorldEnv, self).__init__(
            grid, nrow, ncol, goal_length, density, ctrl_error, episode_length, verbose)
        self.nb_states = self.nrow * self.ncol
        self.nb_actions = 4

    def speculate(self, confidence=0., gamma=1., episode_length=-1):
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

        # Kernel, reward, safety of terminal states
        for goal in self.goals:
            for i in range(self.nb_actions):
                kernel[self._coord_to_index(goal), i, :] = 0.
                kernel[self._coord_to_index(goal), i, self._coord_to_index(goal)] = 1.

            reward[self._coord_to_index(goal)] = 0
            safety[self._coord_to_index(goal)] = 1

        initial_state = self._coord_to_index(self.start)
        terminal_states = np.array([self._coord_to_index(goal) for goal in self.goals])

        initial_policy = self._construct_initial_policy()

        if episode_length is not None and episode_length < 0:
            episode_length = self.episode_length
        else:
            episode_length = episode_length

        return self.nb_states, self.nb_actions, kernel, reward, safety, initial_policy,\
               confidence, gamma, initial_state, terminal_states, episode_length

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
        _, reward, done, info = super(GridWorldEnv, self).step(action)
        info['state'] = self._coord_to_index(self.state)
        return self._observe(), reward, done, info

    def forced_step(self, state):
        super(GridWorldEnv, self).forced_step(self._index_to_coord(state))
        return self._observe()

    def cheat(self):
        return self._coord_to_index(self.state)

    def _coord_to_index(self, coord):
        return coord[0] * self.ncol + coord[1]

    def _index_to_coord(self, index):
        return np.divmod(index, self.ncol)

    def _construct_initial_policy(self):
        initial_policy = np.zeros((self.nb_states, self.nb_actions))
        initial_policy[:, -1] = 1.

        for i in range(self.nrow):
            for j in range(self.ncol):
                if i == 0:
                    initial_policy[self._coord_to_index((i, j)), :] = np.array([0., 0., 1., 0.,])
                elif i == self.nrow - 1:
                    initial_policy[self._coord_to_index((i, j)), :] = np.array([1., 0., 0., 0.,])

        initial_policy[self._coord_to_index((self.nrow-1, 0)), :] = np.array([0., 0., 0., 1.])
        initial_policy[self._coord_to_index((0, self.ncol-1)), :] = np.array([0., 1., 0., 0.])
        # for i in range(self.nrow):
        #     for j in range(self.ncol):
        #         nb_safe_acts = 0
        #         for k in range(self.nb_actions):
        #             next_pos = self._move((i, j), self.action_pos_dict[k])
        #             if next_pos in self.goals:
        #                 nb_safe_acts = 1
        #                 for l in range(self.nb_actions):
        #                     initial_policy[self._coord_to_index((i, j)), l] = 0
        #                 initial_policy[self._coord_to_index((i, j)), k] = 1.
        #             elif self.grid[next_pos] == 1 and (i, j) != next_pos:
        #                 # left:0 / down:1 / right:2 / up:3
        #                 if k == 3 or np.abs(next_pos[1] - self.goals[-1][1]) < np.abs(j - self.goals[-1][1]):
        #                     nb_safe_acts += 3
        #                     initial_policy[self._coord_to_index((i, j)), k] = 3
        #                 else:
        #                     nb_safe_acts += 1
        #                     initial_policy[self._coord_to_index((i, j)), k] = 1
        #         if nb_safe_acts == 0:
        #             initial_policy[self._coord_to_index((i, j)), :] = [1. / 4, 1. / 4, 1. / 4, 1. / 4]
        #         else:
        #             initial_policy[self._coord_to_index((i, j)), :] /= nb_safe_acts
        return initial_policy

