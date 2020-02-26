import os
import pickle

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lyapunov_reachability.speculation_ddpg.base import ContinuousBase
from lyapunov_reachability.common.utils import init_weights
from lyapunov_reachability.common.models import *


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OUNoise:
    def __init__(self, action_size, theta, mu, sigma):
        self.action_size = action_size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X = np.zeros(self.action_size)

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx

        return self.X


class SimpleDDPG(ContinuousBase):

    def __init__(
            self, env, confidence, extractor=None, extractor_params=None, decoder=None, decoder_params=None,
            seed=None, lr=1e-3, batch_size=128, gamma=0.999, ob_resize=32, grid_points=21, strict_done=False,
            buffer_size=int(1e5), polyak=1e-3, noise_theta=0.15, noise_mu=0., noise_sigma=0.3, gradient_clip=(0.5, 1.0),
            lr_ratio=0.1, re_init=False, save_dir='../../simple-ddpg'):

        # Neural networks
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None

        # Optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None

        # Noise generator
        self.ou_noise = OUNoise(len(env.action_space.low), noise_theta, noise_mu, noise_sigma)

        # Parameters
        self.polyak = polyak                    # Soft target update
        self.gradient_clip = gradient_clip      # (clip_critic, clip_actor)
        self.lr_ratio = lr_ratio                # actor_lr = critic_lr * lr_ratio

        super(SimpleDDPG, self).__init__(
            env, confidence, extractor, extractor_params, decoder, decoder_params, seed, lr, batch_size, gamma,
            ob_resize, grid_points, strict_done, buffer_size, None, None, re_init, save_dir)

        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

    def setup_model(self, extractor=None, extractor_params=None, decoder=None, decoder_params=None):
        """
        actor:      Marginally safe policy.
        critic:     Reachability value function w.r.t. the actor.
        """
        # model_ = Cnn
        # params_ = {'channels_': [16, 32, 32], 'kernel_sizes_': [5, 5, 5], 'strides_': [2, 2, 2],}
        self.critic = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                 decoder=decoder, decoder_params=decoder_params)
        self.actor = DetActor(self.ob_space, self.ac_space, extractor, extractor_params,
                              decoder=decoder, decoder_params=decoder_params)
        self.target_critic = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                        decoder=decoder, decoder_params=decoder_params)
        self.target_actor = DetActor(self.ob_space, self.ac_space, extractor, extractor_params,
                                     decoder=decoder, decoder_params=decoder_params)

    def setup_optimizer(self, reload=False, baseline_dir=None, baseline_step=None):
        if reload:
            print("Check if you called agent.load_weights(load_dir)!")
        elif baseline_dir or baseline_step is None:
            init_weights(self.critic)
            init_weights(self.actor)
            self.hard_target_update()
        else:
            raise RuntimeError("Baseline training is not supported.")

        self.hard_target_update()

        self.critic.cuda(device)
        self.actor.cuda(device)
        self.target_actor.cuda(device)
        self.target_critic.cuda(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr*self.lr_ratio)

    def update_network(self, *args):
        """
        For more information, see
        (a) https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html.
        (b) https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/3_A2C_DDPG
        :param obs_t:   Observation at time t.
        :param act_t:   Action taken at time t, drawn from exploratory policy.
        :param obs_tp1: Observation at time t+1.
        :param reached: 1. (True) if the agent ever reached target at time interval (0, t].
        :param done:    1. (True) if obs_t is the final state of the trajectory.
        """
        try:
            obs_t, act_t, obs_tp1, reached, done = self._draw_batch()
        except ValueError:
            return

        # Get q-value.
        q_t = self.critic(obs_t, act_t).squeeze(1)

        # Get target q.
        act_tp1 = self.target_actor(obs_tp1)
        q_tp1 = self.target_critic(obs_tp1, act_tp1).squeeze(1)
        q_t_target = (1. - reached) * (1. - done) * (self.gamma * q_tp1) + reached

        critic_loss = F.mse_loss(q_t, q_t_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip[0])
        self.critic_optimizer.step()

        # Get actor
        actor_a_t = self.actor(obs_t)
        actor_loss = self.critic(obs_t, actor_a_t).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip[1])
        self.actor_optimizer.step()

    def step(self, obs, *args):
        pre_act = self.actor(self.transform(obs).unsqueeze(0).to(device)).detach()
        act = pre_act.data.cpu().numpy()[0] + self.ou_noise.sample()
        act = np.clip(act, -1., 1.)
        # Denormalize action before apply it to env.
        return (act + 1.) * 0.5 * (self.act_high - self.act_low) + self.act_low

    def verify_state(self, obs):
        with torch.no_grad():
            obs_b = self.transform(obs).unsqueeze(0).to(device)
            act_b = self.actor(obs_b)
            reachability = self.critic(obs_b, act_b)
            return reachability.data.cpu().numpy()[0]

    def target_update(self):
        self.soft_target_update()

    def hard_target_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def soft_target_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))

    def save_setup(self, extractor, extractor_params, decoder, decoder_params, save_env=False):

        if save_env:
            with open(os.path.join(self.save_dir, 'env.pkl'), 'wb') as f:
                pickle.dump(self.env, f)

        data = {
            "confidence": self.confidence,
            "extractor": extractor.__name__,
            "extractor_params": extractor_params,
            "decoder": None if decoder is None else decoder.__name__,
            "decoder_params": decoder_params,
            "seed": self.seed,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "ob_resize": self.ob_space[-1],
            "grid_points": self.grid_points,
            "strict_done": self.strict_done,
            "buffer_size": self.replay.buffer_size,
            "polyak": self.polyak,
            "noise_theta": self.ou_noise.theta,
            "noise_mu": self.ou_noise.mu,
            "noise_sigma": self.ou_noise.sigma,
            "gradient_clip": self.gradient_clip,
            "lr_ratio": self.lr_ratio
        }

        with open(os.path.join(self.save_dir, "params.pkl".format(self.steps)), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_weights(self):
        torch.save(self.critic.state_dict(), os.path.join(self.save_dir, "{}-critic".format(self.steps)))
        torch.save(self.actor.state_dict(), os.path.join(self.save_dir, "{}-actor".format(self.steps)))

    def load_weights(self, load_dir):
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic".format(self.steps))))
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "{}-actor".format(self.steps))))

        self.critic.eval()
        self.actor.eval()

        self.hard_target_update()


if __name__ == '__main__':
    from classic_envs.pendulum import VanillaPendulumEnv
    from lyapunov_reachability.common.networks import Mlp, Cnn

    grid_points = 31
    episode_length = 300
    confidence = 0.8
    gamma = 0.9999
    strict_done = True

    env = VanillaPendulumEnv()
    name = '{}-pendulum'.format(int(episode_length))

    steps = int(5e6)
    log_interval = int(1e4)
    save_interval = int(1e5)

    # Create & train
    ddpg = SimpleDDPG(
        env, confidence, extractor=Mlp, extractor_params={'channels_': [400, 300], 'activ': 'tanh'}, seed=1234,
        gamma=gamma, grid_points=grid_points, strict_done=strict_done, save_dir=os.path.join(name, 'ddpg-initial'))
    ddpg.run(steps, episode_length, log_interval=log_interval, save_interval=save_interval, )
