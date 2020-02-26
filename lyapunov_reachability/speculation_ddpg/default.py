import os
import pickle
from itertools import chain

import torch.optim as optim
import torch.nn.functional as F

from lyapunov_reachability.speculation_ddpg.ddpg import OUNoise
from lyapunov_reachability.speculation_ddpg.base import ContinuousBase
from lyapunov_reachability.common.models import *
from lyapunov_reachability.common.utils import init_weights
from lyapunov_reachability.common.replay import Replay, PrioritizedReplay

EPS = 1e-8

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefaultDDPG(ContinuousBase):

    def __init__(
            self, env, confidence, extractor=None, extractor_params=None, decoder=None, decoder_params=None,
            seed=None, lr=1e-3, batch_size=128, gamma=0.999, ob_side=32, grid_points=21, strict_done=False,
            replay_size=int(1e5), replay_prioritized=False, replay_double=False, polyak=1e-3,
            noise_theta=0.15, noise_mu=0., noise_sigma=0.3, gradient_clip=(0.5, 1.0), lr_ratio=0.1, double=False,
            baseline_dir=None, baseline_step=None, re_init=False, save_dir='../../spec-def-ddpg'):

        # Neural networks
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
        if double:
            self.critic0 = None
            self.target_critic0 = None

        # Optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None

        # Noise generator
        self.ou_noise = OUNoise(len(env.action_space.low), noise_theta, noise_mu, noise_sigma)

        # Parameters
        self.polyak = polyak                    # Soft target update
        self.gradient_clip = gradient_clip      # (clip_critic, clip_actor)
        self.lr_ratio = lr_ratio                # actor_lr = critic_lr * lr_ratio

        self.replay_reached = None
        self.replay_double = replay_double
        self.double = double                    # True if double Q-learning is used

        super(DefaultDDPG, self).__init__(
            env, confidence, extractor, extractor_params, decoder, decoder_params, seed, lr, batch_size, gamma, ob_side,
            grid_points, strict_done, replay_size, replay_prioritized, baseline_dir, baseline_step, re_init, save_dir)

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
        if self.double:
            self.critic0 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                      decoder=decoder, decoder_params=decoder_params)
            self.target_critic0 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                             decoder=decoder, decoder_params=decoder_params)

    def setup_optimizer(self, reload=False, baseline_dir=None, baseline_step=None):
        if reload:
            print("Check if you called agent.load_weights(load_dir)!")
        elif baseline_dir is None or baseline_step is None:
            print("Start from scratch.")
            init_weights(self.critic, conservative_init=True)
            if self.double:
                init_weights(self.critic0, conservative_init=True)
            init_weights(self.actor)
            self.hard_target_update()
        else:
            assert isinstance(baseline_step, int), "Provide train step. File name should be [steps]-critic/actor"
            print("Start from baseline.")
            critic_path = os.path.abspath(os.path.join(baseline_dir, '{}-critic'.format(baseline_step)))
            if not os.path.exists(critic_path):
                critic_path = os.path.abspath(os.path.join(baseline_dir, '{}-critic1'.format(baseline_step)))
            actor_path = os.path.abspath(os.path.join(baseline_dir, '{}-actor'.format(baseline_step)))
            self.critic.load_state_dict(torch.load(critic_path))
            # init_weights(self.critic, conservative_init=True)
            if self.double:
                self.critic0.load_state_dict(torch.load(critic_path))
                # init_weights(self.critic0, conservative_init=True)
            self.actor.load_state_dict(torch.load(actor_path))
            self.hard_target_update()

        self.actor.cuda(device)
        self.target_actor.cuda(device)
        self.critic.cuda(device)
        self.target_critic.cuda(device)
        if self.double:
            self.critic0.cuda(device)
            self.target_critic0.cuda(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr*self.lr_ratio, amsgrad=True)
        if self.double:
            critic_params = [self.critic.parameters(), self.critic0.parameters()]
            self.critic_optimizer = optim.Adam(chain(*critic_params), lr=self.lr, amsgrad=True)
        else:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, amsgrad=True)

    def setup_replay(self, reload=False, baseline_dir=None, baseline_step=None):
        create_replay = False

        if reload:
            print("Check if you called agent.load_weights(load_dir)!")
        elif baseline_dir is None or baseline_step is None:
            create_replay = True
        else:
            assert isinstance(baseline_step, int), "Provide train step. File name should be [steps]-replay[_*]"
            print("Trying to get the memory from the baseline..")
            replay_path = os.path.abspath(os.path.join(baseline_dir, '{}-replay'.format(baseline_step)))
            replay_reached_path = os.path.abspath(os.path.join(baseline_dir, '{}-replay_reached'.format(baseline_step)))
            try:
                with open(replay_path, 'rb') as f:
                    self.replay = pickle.load(f)
                    self.replay.set_size(self.replay_size)
                if self.replay_double:
                    with open(replay_reached_path, 'rb') as f:
                        self.replay_reached = pickle.load(f)
                        self.replay_reached.set_size(int(self.replay_size * (1. / self.confidence - 1)))
            except FileNotFoundError:
                create_replay = True

        if create_replay:
            print("Create a memory from scratch.")
            if self.replay_prioritized:
                self.replay = PrioritizedReplay(self.replay_size)
            elif self.replay_double:
                self.replay = Replay(self.replay_size)
                self.replay_reached = Replay(int(self.replay_size * (1. / self.confidence - 1)))
            else:
                self.replay = Replay(self.replay_size)

    def update_network(self, *args):
        """
        For more information, see
        (a) https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html.
        (b) https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/3_A2C_DDPG
        """
        try:
            if self.replay_prioritized:
                obs_t, act_t, obs_tp1, reached, done, idxs, weight_mask = self._draw_batch_prioritized(*args)
            elif self.replay_double:
                obs_t, act_t, obs_tp1, reached, done = self._draw_batch(*args)
            else:
                obs_t, act_t, obs_tp1, reached, done = super(DefaultDDPG, self)._draw_batch(*args)
        except ValueError:
            return

        if self.double:
            # Get q-value.
            q_t = self.critic(obs_t, act_t).squeeze(1)
            q0_t = self.critic0(obs_t, act_t).squeeze(1)

            # Get target q.
            act_tp1 = self.target_actor(obs_tp1)
            q_tp1 = torch.min(self.target_critic(obs_tp1, act_tp1).squeeze(1),
                              self.target_critic0(obs_tp1, act_tp1).squeeze(1))
            q_t_target = (1. - reached) * (1. - done) * (self.gamma * q_tp1) + reached

            if self.replay_prioritized:
                # Importance sampling mask is applied to all samples.
                critic_loss = (weight_mask * (F.mse_loss(q_t, q_t_target, reduction='none') +\
                                              F.mse_loss(q0_t, q_t_target, reduction='none'))).mean()
                errors = torch.abs(q_t - q_t_target).data.cpu().numpy()
                for idx, error in zip(idxs, errors):
                    self.replay.update(idx, error)
            else:
                critic_loss = F.mse_loss(q_t, q_t_target) + F.mse_loss(q0_t, q_t_target)
        else:
            # Get q-value.
            q_t = self.critic(obs_t, act_t).squeeze(1)

            # Get target q.
            act_tp1 = self.target_actor(obs_tp1)
            q_tp1 = self.target_critic(obs_tp1, act_tp1).squeeze(1)
            q_t_target = (1. - reached) * (1. - done) * (self.gamma * q_tp1) + reached

            if self.replay_prioritized:
                # Importance sampling mask is applied to all samples.
                critic_loss = (weight_mask * F.mse_loss(q_t, q_t_target, reduction='none')).mean()
                errors = torch.abs(q_t - q_t_target).data.cpu().numpy()
                for idx, error in zip(idxs, errors):
                    self.replay.update(idx, error)
            else:
                critic_loss = F.mse_loss(q_t, q_t_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip[0])
        self.critic_optimizer.step()

        # Do not improve actors until the critics are sufficiently trained.
        if self.steps < args[-1]:
            return
        # elif self.steps == args[-1]:#TODO: remove this if not helpful.
        #     try:
        #         self.replay_reached.clear()
        #     except ValueError:
        #         print("This message should not appear.")

        # Get actor
        actor_a_t = self.actor(obs_t)
        actor_loss = self.critic(obs_t, actor_a_t).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip[1])
        self.actor_optimizer.step()

    def store_sample(self, obs, action, next_obs, reach, done):
        # Normalize action first.
        action = (action - self.act_low) / (self.act_high - self.act_low) * 2. - 1.

        # Store the sample in both replay & trajectory buffer,
        if self.replay_double and reach == 1.:
            self.replay_reached.store((obs, action, next_obs, reach, done))
        else:
            self.replay.store((obs, action, next_obs, reach, done))
        return

    def _draw_batch(self, *args):
        """
        Draw a minibatch from the replay.
        :return:    batches (obs_b, act_b, next_obs_b, reach_b, done_b)
        """
        safe_batch_size = int(self.batch_size * self.confidence)
        obs_b0, act_b0, next_obs_b0, reached_b0, done_b0 = self.replay.sample(safe_batch_size)
        obs_b1, act_b1, next_obs_b1, reached_b1, done_b1 = self.replay_reached.sample(self.batch_size - safe_batch_size)

        obs_b = self.transform(np.concatenate((obs_b0, obs_b1), axis=0)).to(device)
        act_b = torch.FloatTensor(np.concatenate((act_b0, act_b1), axis=0)).to(device)
        next_obs_b = self.transform(np.concatenate((next_obs_b0, next_obs_b1), axis=0)).to(device)
        reached_b = torch.FloatTensor(np.concatenate((reached_b0, reached_b1), axis=0)).to(device)
        done_b = torch.FloatTensor(np.concatenate((done_b0, done_b1), axis=0)).to(device)
        return obs_b, act_b, next_obs_b, reached_b, done_b

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

    def target_update(self, **kwargs):
        self.soft_target_update()

    def hard_target_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        if self.double:
            for target_param, param in zip(self.target_critic0.parameters(), self.critic0.parameters()):
                target_param.data.copy_(param.data)

    def soft_target_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        if self.double:
            for target_param, param in zip(self.target_critic0.parameters(), self.critic0.parameters()):
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
            "ob_side": self.ob_space[-1],
            "grid_points": self.grid_points,
            "strict_done": self.strict_done,
            "replay_size": self.replay.replay_size,
            "replay_prioritized": self.replay_prioritized,
            "replay_double": self.replay_double,
            "polyak": self.polyak,
            "noise_theta": self.ou_noise.theta,
            "noise_mu": self.ou_noise.mu,
            "noise_sigma": self.ou_noise.sigma,
            "gradient_clip": self.gradient_clip,
            "lr_ratio": self.lr_ratio,
            "double": self.double,
        }

        with open(os.path.join(self.save_dir, "params.pkl".format(self.steps)), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_weights(self):
        torch.save(self.actor.state_dict(), os.path.join(self.save_dir, "{}-actor".format(self.steps)))
        torch.save(self.critic.state_dict(), os.path.join(self.save_dir, "{}-critic".format(self.steps)))
        if self.double:
            torch.save(self.critic0.state_dict(), os.path.join(self.save_dir, "{}-critic0".format(self.steps)))

        # TODO: save replay as well.
        with open(os.path.join(self.save_dir, "{}-replay".format(self.steps)), 'wb') as f:
            pickle.dump(self.replay, f, pickle.HIGHEST_PROTOCOL)
        if self.replay_double:
            with open(os.path.join(self.save_dir, "{}-replay_reached".format(self.steps)), 'wb') as f:
                pickle.dump(self.replay_reached, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "{}-actor".format(self.steps))))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic".format(self.steps))))
        if self.double:
            self.critic0.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic0".format(self.steps))))

        self.actor.eval()
        self.critic.eval()
        self.critic0.eval()

        self.hard_target_update()

        # TODO: save replay as well.
        try:
            with open(os.path.join(load_dir, "{}-replay".format(self.steps)), 'rb') as f:
                self.replay = pickle.load(f)
                self.replay.set_size(self.replay_size)
            if self.replay_double:
                with open(os.path.join(load_dir, "{}-replay_reached".format(self.steps)), 'rb') as f:
                    self.replay_reached = pickle.load(f)
                    self.replay_reached.set_size(int(self.replay_size * (1. / self.confidence - 1)))
        except FileNotFoundError:
            return


if __name__ == '__main__':
    from classic_envs.random_integrator import RandomIntegratorEnv
    from lyapunov_reachability.common.networks import Mlp, Cnn

    n = 10
    grid_points = 30
    episode_length = 400
    confidence = 0.8
    gamma = 0.99

    env = RandomIntegratorEnv()
    name = '{}-continuous-integrator'.format(episode_length)

    steps = int(5e6)
    log_interval = int(1e4)
    save_interval = int(1e5)

    # Create & train
    ddqn = DefaultDDPG(env, confidence, extractor=Mlp, extractor_params={'channels_': [128, 128]},
                       seed=1234, gamma=gamma, save_dir=os.path.join(name, 'baseline'))
    ddqn.run(steps, episode_length, log_interval=log_interval, save_interval=save_interval, )
