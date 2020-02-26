import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PI = np.pi


class FixedReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Essential description:
        self.sim.data.qpos = [theta_1, theta_2, target_x, target_y]
        self.sim.data.qvel = [omega_1, omega_2, 0, 0]
        (Assuming that the target is fixed)

        theta_1: angle of first arm (fixed at the center)
        theta_2: angle of the second arm, w.r.t. the tip of the first arm

        Length of both arms are 0.1, thus its reach is 0.2
    """
    reach = 0.1 * np.sqrt(2)

    def __init__(self):
        utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self, os.path.join(FILE_DIR, 'frictionless_reacher.xml'), 2)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # Apply action
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        next_vec = self.get_body_com("fingertip") - self.get_body_com("target")
        next_dist = np.linalg.norm(next_vec)

        # done at time t+1
        done = next_dist <= 0.005

        # "safety" at time t
        safety = (np.abs(self.get_body_com("fingertip")[1]) <= 0.1)# and (reward_dist + next_dist < 0.)

        return ob, reward, done, dict(safety=safety * 1., reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.theta_goal = (np.random.rand() - 0.5) * np.pi / 2. + np.pi
        self.goal = self.reach * np.array([np.cos(self.theta_goal), np.sin(self.theta_goal)])

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def _get_safety(self):
        pass

    def _get_done(self):
        pass

    def sample_states(self, grid_points=21):
        """
        state = [
            np.cos(theta), np.sin(theta), target_pos, omega,
            self.get_body_com("fingertip") - self.get_body_com("target")
        ]
        """
        states = []
        for param_2 in range(grid_points):
            pos_2 = 2 * PI * param_2 / (grid_points - 1) - PI
            for param_1 in range(grid_points):
                pos_1 = 2 * PI * param_1 / (grid_points - 1) - PI

                dist_x = 0.1 * np.cos(pos_1) + 0.11 * np.cos(pos_1 + pos_2) - 0. + self.reach
                dist_y = 0.1 * np.sin(pos_1) + 0.11 * np.sin(pos_1 + pos_2) - 0.
                states.append(np.array([np.cos(pos_1), np.cos(pos_2), np.sin(pos_1), np.sin(pos_2),
                                        -self.reach, 0., 0., 0., dist_x, dist_y, 0.]))
        return np.array(states)
