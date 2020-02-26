import os
import pickle
from itertools import chain

import torch.optim as optim
import torch.nn.functional as F
from scipy import stats

from lyapunov_reachability.speculation_ddpg.ddpg import OUNoise
from lyapunov_reachability.speculation_ddpg.base import ContinuousBase
from lyapunov_reachability.common.utils import init_weights
from lyapunov_reachability.common.replay import Replay, PrioritizedReplay
from lyapunov_reachability.common.models import *
from lyapunov_reachability.common.gradient import *


EPS = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def conservative_critic_gradient(critic, target_critic, critic_values, critic_loss, state, action, lr):
#     with torch.no_grad():
#         critic_ub = target_critic(state, action)
#     critic_grads = torch.autograd.grad(critic_loss, critic.parameters(), create_graph=True)
#     flat_critic_grads = flat_grad(critic_grads)
#
#     return ((critic_ub - critic_values).min() / (flat_critic_grads ** 2)).clamp_(min=0, max=lr)


class LyapunovDDPG(ContinuousBase):

    def __init__(
            self, env, confidence, extractor=None, extractor_params=None, decoder=None, decoder_params=None,
            seed=None, lr=1e-3, batch_size=128, gamma=0.999, ob_side=32, grid_points=21, strict_done=False,
            replay_size=int(1e5), replay_prioritized=False, replay_double=False, polyak=1e-3,
            noise_theta=0.15, noise_mu=0., noise_sigma=0.3, gradient_clip=(0.5, 1.0), lr_ratio=0.1,
            target_ratio=0., baseline_dir=None, baseline_step=None, re_init=False, save_dir='../../spec-lyap-ddpg'):

        # Neural networks
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.step_critic = None
        self.target_actor = None
        self.target_critic1 = None
        self.target_critic2 = None
        self.target_step_critic = None
        self.log_lambda = None

        # Optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None
        # self.target_optimizer = None
        self.lagrangian_optimizer = None

        # Noise generator
        self.ou_noise = OUNoise(len(env.action_space.low), noise_theta, noise_mu, noise_sigma)

        # Parameters
        self.polyak = polyak                    # Soft target update
        self.gradient_clip = gradient_clip      # (clip_critic, clip_actor)
        self.target_ratio = target_ratio        # Double Q technique
        self.lr_ratio = dict()
        if type(lr_ratio) is dict:
            self.lr_ratio['actor'] = lr_ratio.get('actor', 0.1)
            self.lr_ratio['log_lambda'] = lr_ratio.get('log_lambda', 0.01)
        elif type(lr_ratio) is float:
            self.lr_ratio['actor'] = lr_ratio
            self.lr_ratio['log_lambda'] = min(lr_ratio ** 2, 1.)
        else:
            raise ValueError("Provide proper learning rates. Example: {'actor': 0.1, 'log_lambda':0.01}")
        print("Learning rate ratios: {}".format(self.lr_ratio))
        self.replay_reached = None
        self.replay_double = replay_double

        super(LyapunovDDPG, self).__init__(
            env, confidence, extractor, extractor_params, decoder, decoder_params, seed, lr, batch_size, gamma, ob_side,
            grid_points, strict_done, replay_size, replay_prioritized, baseline_dir, baseline_step, re_init, save_dir)

        # Necessary units to compute Lagrangian multiplier,
        # We assume that episode_length <= replay_size / 100 here.
        self.trajectory = Replay(int(replay_size / 100))
        self.record_margin = [EPS] * 100
        self.record_lambda = [1.] * 100
        self.auxiliary_cost = 0.
        self.threshold = 1. - self.confidence

    def setup_model(self, extractor=None, extractor_params=None, decoder=None, decoder_params=None):
        """
        actor:      Marginally safe policy.
        critic:     Reachability value function w.r.t. the actor.
        """
        # model_ = Cnn
        # params_ = {'channels_': [16, 32, 32], 'kernel_sizes_': [5, 5, 5], 'strides_': [2, 2, 2],}
        self.critic1 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                  decoder=decoder, decoder_params=decoder_params)
        self.critic2 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                  decoder=decoder, decoder_params=decoder_params)
        self.actor = DetActor(self.ob_space, self.ac_space, extractor, extractor_params,
                              decoder=decoder, decoder_params=decoder_params)
        self.step_critic = GeneralCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                         decoder=decoder, decoder_params=decoder_params)
        self.target_critic1 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                         decoder=decoder, decoder_params=decoder_params)
        self.target_critic2 = ProbCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                         decoder=decoder, decoder_params=decoder_params)
        self.target_actor = DetActor(self.ob_space, self.ac_space, extractor, extractor_params,
                                     decoder=decoder, decoder_params=decoder_params)
        self.target_step_critic = GeneralCritic(self.ob_space, self.ac_space, extractor, extractor_params,
                                                decoder=decoder, decoder_params=decoder_params)
        self.log_lambda = Lambda(self.ob_space, self.ac_space, extractor, extractor_params,
                                 decoder=decoder, decoder_params=decoder_params)

    def setup_optimizer(self, reload=False, baseline_dir=None, baseline_step=None):
        if reload:
            print("Check if you called agent.load_weights(load_dir)!")
        elif baseline_dir is None or baseline_step is None:
            print("Start from scratch.")
            init_weights(self.actor)
            init_weights(self.critic1, conservative_init=True)
            init_weights(self.critic2, conservative_init=True)
            init_weights(self.step_critic)
            init_weights(self.log_lambda)
            self.hard_target_update()
        else:
            assert isinstance(baseline_step, int), "Provide train step. File name should be [steps]-critic/actor"
            print("Start from baseline.")
            actor_path = os.path.abspath(os.path.join(baseline_dir, '{}-actor'.format(baseline_step)))
            # critic_path = os.path.abspath(os.path.join(baseline_dir, '{}-critic'.format(baseline_step)))
            # if not os.path.exists(critic_path):
            #     critic_path = os.path.abspath(os.path.join(baseline_dir, '{}-critic1'.format(baseline_step)))
            self.actor.load_state_dict(torch.load(actor_path))
            init_weights(self.critic1, conservative_init=True)
            init_weights(self.critic2, conservative_init=True)
            # self.critic1.load_state_dict(torch.load(critic_path))
            # self.critic2.load_state_dict(torch.load(critic_path))
            init_weights(self.step_critic)
            init_weights(self.log_lambda)
            self.hard_target_update()

        self.actor.cuda(device)
        self.target_actor.cuda(device)
        self.critic1.cuda(device)
        self.critic2.cuda(device)
        self.target_critic1.cuda(device)
        self.target_critic2.cuda(device)
        self.step_critic.cuda(device)
        self.target_step_critic.cuda(device)
        self.log_lambda.cuda(device)

        critic_params = [self.critic1.parameters(), self.critic2.parameters(), self.step_critic.parameters()]
        self.critic_optimizer = optim.Adam(chain(*critic_params), lr=self.lr, amsgrad=True)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr * self.lr_ratio['actor'], amsgrad=True)
        self.lagrangian_optimizer = optim.Adam(self.log_lambda.parameters(), lr=self.lr * self.lr_ratio['log_lambda'],
                                               amsgrad=True)

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
                obs_t, act_t, obs_tp1, reached, done, idxs, weight_mask = self._draw_batch_prioritized()
            elif self.replay_double:
                obs_t, act_t, obs_tp1, reached, done = self._draw_batch()
            else:
                obs_t, act_t, obs_tp1, reached, done = super(LyapunovDDPG, self)._draw_batch()
        except ValueError:
            return

        act_tp1 = self.target_actor(obs_tp1)

        q1_t = self.critic1(obs_t, act_t).squeeze(1)
        q2_t = self.critic2(obs_t, act_t).squeeze(1)
        q1_tp1 = self.target_critic1(obs_tp1, act_tp1).squeeze(1)
        q2_tp1 = self.target_critic2(obs_tp1, act_tp1).squeeze(1)
        q_tp1 = self.target_ratio * torch.max(q1_tp1, q2_tp1) + \
                (1. - self.target_ratio) * torch.min(q1_tp1, q2_tp1)
        q_t_target = ((1. - reached) * (1. - done) * (self.gamma * q_tp1) + reached).detach()

        step_q_t = self.step_critic(obs_t, act_t).squeeze(1)
        step_q_tp1 = self.target_step_critic(obs_tp1, act_tp1).squeeze(1)
        step_q_t_target = ((1. - done) * (1. - reached) * (1. + self.gamma * step_q_tp1)).detach()

        if self.replay_prioritized:
            # Importance sampling mask is applied to all samples.
            critic_loss = (weight_mask *\
                          (F.mse_loss(q1_t, q_t_target, reduction='none') +
                           F.mse_loss(q2_t, q_t_target, reduction='none') +
                           F.mse_loss(step_q_t, step_q_t_target, reduction='none'))).mean()
            errors = torch.abs(q1_t - q_t_target).data.cpu().numpy()
            for idx, error in zip(idxs, errors):
                self.replay.update(idx, error)
        else:
            critic_loss = F.mse_loss(q1_t, q_t_target) + F.mse_loss(q2_t, q_t_target) + \
                          F.mse_loss(step_q_t, step_q_t_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip[0])
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip[0])
        nn.utils.clip_grad_norm_(self.step_critic.parameters(), self.gradient_clip[0])
        self.critic_optimizer.step()

        # Do not improve actors until the critics are sufficiently trained.
        if self.steps < args[-1]:
            return

        # Actor update ----------------------------------------
        # Coordinate descent (somehow this works better)
        actor_t = self.actor(obs_t)
        q_actor_t = self.critic1(obs_t, actor_t)
        step_q_actor_t = self.step_critic(obs_t, actor_t)
        lyapunov_q_t = q_actor_t + self.auxiliary_cost * step_q_actor_t
        lambda_t = torch.exp(self.log_lambda(obs_t))
        lambda_const_t = lambda_t.detach()

        actor_loss = (q_actor_t + lambda_const_t * lyapunov_q_t).mean()
        # actor_loss = ((q_actor_t + lambda_const_t * lyapunov_q_t) / (1. + lambda_const_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip[1])
        self.actor_optimizer.step()

        n_actor_t = self.actor(obs_t)
        n_q_actor_t = self.critic1(obs_t, n_actor_t)
        n_step_q_actor_t = self.step_critic(obs_t, n_actor_t)
        n_lyapunov_q_t = n_q_actor_t + self.auxiliary_cost * n_step_q_actor_t

        # Heuristic in Lagrangian: optimize for the worst-case (where the inequality constraint does not hold)
        lagrangian_loss = -(lambda_t * (n_lyapunov_q_t - lyapunov_q_t - self.auxiliary_cost)).mean()
        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        nn.utils.clip_grad_norm_(self.log_lambda.parameters(), self.gradient_clip[0])
        self.lagrangian_optimizer.step()

    def store_sample(self, obs, action, next_obs, reach, done):
        # Normalize action first.
        action = (action - self.act_low) / (self.act_high - self.act_low) * 2. - 1.

        # Store the sample in both replay & trajectory buffer,
        if self.replay_double and reach == 1.:
            self.replay_reached.store((obs, action, next_obs, reach, done))
        else:
            self.replay.store((obs, action, next_obs, reach, done))
        self.trajectory.store((obs, action, next_obs, reach, done))
        return

    def _draw_batch(self):
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

    def close_episode(self):
        # Fetch a trajectory as batch & clear the trajectory buffer.
        obs_t, _, _, _, _ = self.trajectory.get(0, len(self.trajectory))
        obs_t = self.transform(obs_t).to(device)

        # Compute auxiliary cost.
        p_t = self.target_critic1(obs_t, self.target_actor(obs_t)).squeeze(1).data.cpu().numpy()
        try:
            margin = np.min(self.threshold - p_t[p_t <= self.threshold])
        except ValueError:
            margin = 0.
        if not np.isnan(margin):
            self.record_margin[0:0] = [margin]
            self.record_margin = self.record_margin[:100]
        self.auxiliary_cost = np.min(np.array(self.record_margin)) * (1. - self.gamma)

        # Track the value of log-lambda.
        trajectory_lambda = np.mean(np.exp(self.log_lambda(obs_t).squeeze(1).data.cpu().numpy()))
        if not np.isnan(trajectory_lambda):
            self.record_lambda[0:0] = [trajectory_lambda]
            self.record_lambda = self.record_lambda[:100]

        self.trajectory.clear()

    def step(self, obs, *args):
        pre_act = self.actor(self.transform(obs).unsqueeze(0).to(device))
        act = pre_act.data.cpu().numpy()[0] + self.ou_noise.sample()
        act = np.clip(act, -1., 1.)
        return (act + 1.) * 0.5 * (self.act_high - self.act_low) + self.act_low

    def print_log(self, summary_writer):
        average_safety = np.mean(self.record_safety)
        average_step = np.mean(self.record_step)
        average_margin = np.mean(self.record_margin)
        average_lambda = np.mean(self.record_lambda)
        summary_writer.add_scalar('train/average_safety', average_safety, self.steps)
        summary_writer.add_scalar('train/average_step', average_step, self.steps)
        summary_writer.add_scalar('train/lambda', average_lambda, self.steps)
        print('Episode log\t:: Average safety={}, Margin={}, Average lambda={}'.format(
            average_safety, average_margin, average_lambda))

    def verify_state(self, obs):
        with torch.no_grad():
            obs_b = self.transform(obs).unsqueeze(0).to(device)
            act_b = self.target_actor(obs_b)
            reachability = self.target_critic1(obs_b, act_b)
            return reachability.data.cpu().numpy()[0]

    def target_update(self, **kwargs):
        """
        See "Stabilizing Off-Policy Reinforcement Learning with Conservative Policy Gradients" for more information:
            https://arxiv.org/abs/1910.01062
        Also keep in mind of student t-test:
            https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
        :param kwargs: A dictionary contains "eval_interval (necessary)", "eval_trials."
        """
        # Soft target update for critic & objective actor.
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_step_critic.parameters(), self.step_critic.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))

        # Conservative target update for behavioral (exploratory) actor.
        eval_interval = kwargs.get('eval_interval', None)
        episode_length = kwargs.get('episode_length', 1000)
        if eval_interval is None:
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        else:
            if self.steps % eval_interval == 0:
                eval_env = copy.deepcopy(self.env)
                eval_trials = kwargs.get('eval_trials', 100)

                # target_episode_safety_mean, target_episode_safety_var =\
                #     self.eval(eval_env, eval_trials, episode_length, self.target_actor)

                # current_episode_safety_mean, current_episode_safety_var = \
                #     self.eval(eval_env, eval_trials, episode_length, self.actor)

                current_episode_safety_mean, current_episode_safety_var =\
                    np.mean(self.record_safety), np.var(self.record_safety, ddof=1)

                # t-statistics - Improve if current policy is more safe than target
                # tvalue = (current_episode_safety_mean - target_episode_safety_mean) /\
                #     (np.sqrt((current_episode_safety_var + target_episode_safety_var) / eval_trials) + EPS)
                # df = 2 * eval_trials - 2
                # pvalue = 1. - stats.t.cdf(tvalue, df=df)
                # if pvalue > 0.9:
                #     for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                #         target_param.data.copy_(param.data)

                # t-statistics - Improve if current policy is safe as the safety confidence
                tvalue = (current_episode_safety_mean - self.confidence) /\
                         (np.sqrt(current_episode_safety_var / eval_trials) + EPS)
                df = 2 * eval_trials - 2
                pvalue = 1. - stats.t.cdf(tvalue, df=df)
                if pvalue > 0.9:
                    for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data)

                else:
                    target_episode_safety_mean, target_episode_safety_var =\
                        self.eval(eval_env, eval_trials, episode_length, self.target_actor)
                    tvalue = (current_episode_safety_mean - target_episode_safety_mean) /\
                        (np.sqrt((current_episode_safety_var + target_episode_safety_var) / eval_trials) + EPS)
                    df = 2 * eval_trials - 2
                    pvalue = 1. - stats.t.cdf(tvalue, df=df)
                    if pvalue > 0.9:
                        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                            target_param.data.copy_(param.data)

                del eval_env

    def eval(self, eval_env, eval_trials, episode_length, actor=None):
        # Evaluate for current policy
        episode_safety_log = np.zeros((eval_trials,))

        trials = 0
        episode_step, episode_safety = 0, 1.
        obs = eval_env.reset()
        while trials < eval_trials:
            # action = self.step(obs)
            act = actor(self.transform(obs).unsqueeze(0).to(device))
            act = np.clip(act.data.cpu().numpy()[0] + self.ou_noise.sample(), -1., 1.)
            act = (act + 1.) * 0.5 * (self.act_high - self.act_low) + self.act_low
            next_obs, _, done, info = eval_env.step(act)
            episode_step += 1
            episode_safety *= info['safety']

            if episode_safety < 1. or done or (episode_step == episode_length) > 0.:
                episode_safety_log[trials] = episode_safety
                trials += 1
                episode_step = 0
                episode_safety = 1.
                obs = eval_env.reset()
            else:
                obs = next_obs
        episode_safety_mean, episode_safety_var = \
            np.mean(episode_safety_log), np.var(episode_safety_log, ddof=1)

        return episode_safety_mean, episode_safety_var

    def hard_target_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_step_critic.parameters(), self.step_critic.parameters()):
            target_param.data.copy_(param.data)

    def soft_target_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
        for target_param, param in zip(self.target_step_critic.parameters(), self.step_critic.parameters()):
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
            "target_ratio": self.target_ratio,
        }

        with open(os.path.join(self.save_dir, "params.pkl".format(self.steps)), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_weights(self):
        torch.save(self.actor.state_dict(), os.path.join(self.save_dir, "{}-actor".format(self.steps)))
        torch.save(self.critic1.state_dict(), os.path.join(self.save_dir, "{}-critic1".format(self.steps)))
        torch.save(self.critic2.state_dict(), os.path.join(self.save_dir, "{}-critic2".format(self.steps)))
        torch.save(self.step_critic.state_dict(), os.path.join(self.save_dir, "{}-step_critic".format(self.steps)))
        torch.save(self.log_lambda.state_dict(), os.path.join(self.save_dir, "{}-log_lambda".format(self.steps)))

        # TODO: save replay as well.
        with open(os.path.join(self.save_dir, "{}-replay".format(self.steps)), 'wb') as f:
            pickle.dump(self.replay, f, pickle.HIGHEST_PROTOCOL)
        if self.replay_double:
            with open(os.path.join(self.save_dir, "{}-replay_reached".format(self.steps)), 'wb') as f:
                pickle.dump(self.replay_reached, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "{}-actor".format(self.steps))))
        self.critic1.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic1".format(self.steps))))
        self.critic2.load_state_dict(torch.load(os.path.join(load_dir, "{}-critic2".format(self.steps))))
        self.step_critic.load_state_dict(torch.load(os.path.join(load_dir, "{}-step_critic".format(self.steps))))
        self.log_lambda.load_state_dict(torch.load(os.path.join(load_dir, "{}-log_lambda".format(self.steps))))

        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.step_critic.eval()
        self.log_lambda.eval()
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
    debug = LyapunovDDPG(env, confidence, extractor=Mlp, extractor_params={'channels_': [128, 128]},
                         seed=1234, gamma=gamma, save_dir=os.path.join(name, 'baseline'))
    debug.run(steps, episode_length, log_interval=log_interval, save_interval=save_interval, )
