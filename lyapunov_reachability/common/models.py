import math
import numpy as np
import copy
import torch
import torch.nn as nn
from lyapunov_reachability.common.networks import functional_finder, Mlp, Cnn
from lyapunov_reachability.common.utils import output_shape, clip_but_pass_gradient

EPS = 1e-8
MATH_LOG_2PI = np.log(2 * np.pi)


def normal_likelihood(input_, mu, log_std):
    return -0.5 * (((input_ - mu) / torch.exp(log_std)) ** 2 + 2 * log_std + MATH_LOG_2PI).sum(1, keepdim=True)


def normal_entropy(log_std):
    return log_std + 0.5 + 0.5 * MATH_LOG_2PI


def atanh(x):
    return 0.5 * torch.log((1. + EPS + x)/(1. + EPS - x))


class ProbCritic(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        In:     state & action
        Out:    Q-value for given state-action pair
        :param ob_space             : Shape of the observation space.
        :param ac_space             : Shape of the action space. (must be 1-dimensional)
        :param extractor            : Class of the extractor network.
        :param extractor_params     : Keyword arguments for the extractor network. (optional)
        :param decoder         : Class of the decoder network. (optional)
        :param decoder_params  : Keyword arguments for the decoder network.
        """
        super(ProbCritic, self).__init__()
        self.ac_size = ac_space

        self.decoder = None
        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size + self.ac_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size + self.ac_size, extractor_params), dtype=np.int)
        self.value_layer = nn.Linear(self.feature_size, 1)

    def forward(self, observation, action):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        # Compute Q--value.
        feature = self.extractor(torch.cat((state, action), dim=-1))
        pre_q = self.value_layer(feature.view(-1, self.feature_size))
        return clip_but_pass_gradient(pre_q, 0., 1.)
        # return torch.exp(clip_but_pass_gradient(pre_q, lower=-10., upper=0.))


class GeneralCritic(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None,
                 value_processing='none'):
        """
        In:     state & action
        Out:    Q-value for given state-action pair
        :param ob_space             : Shape of the observation space.
        :param ac_space             : Shape of the action space. (must be 1-dimensional)
        :param extractor            : Class of the extractor network.
        :param extractor_params     : Keyword arguments for the extractor network. (optional)
        :param decoder         : Class of the decoder network. (optional)
        :param decoder_params  : Keyword arguments for the decoder network.
        """
        super(GeneralCritic, self).__init__()
        self.ac_size = ac_space

        self.decoder = None
        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size + self.ac_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size + self.ac_size, extractor_params), dtype=np.int)
        self.value_layer = nn.Linear(self.feature_size, 1)
        self.value_processing = functional_finder(value_processing)

    def forward(self, observation, action):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        # Compute Q--value.
        feature = self.extractor(torch.cat((state, action), dim=-1))
        pre_q = self.value_layer(feature.view(-1, self.feature_size))
        return self.value_processing(pre_q)


class Value(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        In:     state & action
        Out:    Q-value for given state-action pair
        :param ob_space             : Shape of the observation space.
        :param ac_space             : Shape of the action space. (must be 1-dimensional)
        :param extractor            : Class of the extractor network.
        :param extractor_params     : Keyword arguments for the extractor network. (optional)
        :param decoder         : Class of the decoder network. (optional)
        :param decoder_params  : Keyword arguments for the decoder network.
        """
        super(Value, self).__init__()
        self.ac_size = ac_space

        self.decoder = None
        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size, extractor_params), dtype=np.int)
        self.value_layer = nn.Linear(self.feature_size, 1)

    def forward(self, observation):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        # Compute value function.
        feature = self.extractor(state)
        return self.value_layer(feature.view(-1, self.feature_size))


class DetActor(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        Deterministic actor for DDPG implementation.
        """
        super(DetActor, self).__init__()
        self.ac_size = ac_space

        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size, extractor_params), dtype=np.int)
        self.mean_layer = nn.Linear(self.feature_size, self.ac_size)

    def forward(self, observation):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation

        feature = self.extractor(state)
        return torch.tanh(self.mean_layer(feature.view(-1, self.feature_size)))


class Actor(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None,):
        super(Actor, self).__init__()
        self.ac_size = ac_space

        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size + self.ac_size, extractor_params), dtype=np.int)
        self.mean_layer = nn.Linear(self.feature_size, self.ac_size)
        self.logstd_layer = nn.Linear(self.feature_size, self.ac_size)

    def forward(self, observation):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        feature = self.extractor(state).view(-1, self.feature_size)
        mean = self.mean_layer(feature)
        logstd = clip_but_pass_gradient(self.logstd_layer(feature), -6., 2.)
        std = torch.exp(logstd)

        # Reparameterization trick
        pre_sample = mean + std * torch.randn(mean.size(), dtype=mean.dtype, device=mean.device)
        sample = torch.tanh(pre_sample)
        log_prob = normal_likelihood(pre_sample, mean, logstd) - torch.log(-sample ** 2 + 1. + EPS).sum(1, keepdim=True)

        return sample, torch.tanh(mean), log_prob

    def sample(self, observation, deterministic=False):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        feature = self.extractor(state).view(-1, self.feature_size)
        mean = self.mean_layer(feature)
        logstd = clip_but_pass_gradient(self.logstd_layer(feature), -6., 2.)
        std = torch.exp(logstd)

        if deterministic:
            return torch.tanh(mean)
        else:
            pre_sample = mean + std * torch.randn(mean.size(), dtype=mean.dtype, device=mean.device)
            return torch.tanh(pre_sample)

    def log_prob(self, observation, action):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        feature = self.extractor(state).view(-1, self.feature_size)
        mean = self.mean_layer(feature)
        logstd = clip_but_pass_gradient(self.logstd_layer(feature), -6., 2.)

        pre_action = atanh(action)
        log_prob = normal_likelihood(pre_action, mean, logstd) - torch.log(-action ** 2 + 1. + EPS).sum(1, keepdim=True)

        return log_prob


class VAE(nn.Module):
    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        Conditional VAE, designed to match with BCQ (Fujimoto, 2019).
        """
        super(VAE, self).__init__()
        self.ac_size = ac_space

        self.obs_decoder = None
        if decoder is not None:
            self.obs_decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.obs_decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        vae_params = copy.deepcopy(extractor_params)
        vae_params['activ'] = 'relu'
        self.feature_size = np.prod(output_shape(self.ob_size, vae_params), dtype=np.int)

        # encoder
        self.enc_net = extractor(self.ob_size + self.ac_size, **vae_params)
        self.mu_layer = nn.Linear(self.feature_size, 2 * self.ac_size)
        self.logstd_layer = nn.Linear(self.feature_size, 2 * self.ac_size)
        # decoder
        self.dec_net = extractor(self.ob_size + 2 * self.ac_size, **vae_params)
        self.recon_layer = nn.Linear(self.feature_size, self.ac_size)

        del vae_params

    def encode(self, obs, act):
        if self.obs_decoder is not None:
            state = self.obs_decoder(obs)
        else:
            state = obs
        encoded = self.enc_net(torch.cat((state, act), dim=-1))
        return self.mu_layer(encoded), self.logstd_layer(encoded)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.clamp(torch.randn_like(std), min=-0.5, max=0.5)
        return mu + eps * std

    def decode(self, obs, z):
        if self.obs_decoder is not None:
            state = self.obs_decoder(obs)
        else:
            state = obs
        pre_recon = self.dec_net(torch.cat((state, z), dim=-1))
        return torch.tanh(self.recon_layer(pre_recon))

    def forward(self, obs, act):
        mu, logstd = self.encode(obs, act)
        z = self.reparameterize(mu, logstd)
        return self.decode(obs, z), mu, logstd

    def generate(self, obs):
        """
        :param obs: Observation. shape=(batch_size, ob_space)
        :return:    Generated actions. shape=(batch_size, ac_size)
        """
        z_n = torch.randn((list(obs.shape)[0], 2 * self.ac_size), device=obs.device)
        return self.decode(obs, z_n)


class Perturb(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        In:     state & action
        Out:    Action perturbation, pre-scaled.
        :param ob_space             : Shape of the observation space.
        :param ac_space             : Shape of the action space. (must be 1-dimensional)
        :param extractor            : Class of the extractor network.
        :param extractor_params     : Keyword arguments for the extractor network. (optional)
        :param decoder         : Class of the decoder network. (optional)
        :param decoder_params  : Keyword arguments for the decoder network.
        """
        super(Perturb, self).__init__()
        self.ac_size = ac_space

        self.decoder = None
        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size + self.ac_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size + self.ac_size, extractor_params), dtype=np.int)
        self.perturb_layer = nn.Linear(self.feature_size, self.ac_size)

    def forward(self, observation, action):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        feature = self.extractor(torch.cat((state, action), dim=-1))
        perturbation = self.perturb_layer(feature.view(-1, self.feature_size))
        return torch.tanh(perturbation)


class Lambda(nn.Module):

    def __init__(self, ob_space, ac_space, extractor, extractor_params, decoder=None, decoder_params=None, ):
        """
        In:     state,
        Out:    state-wise log-lambda (Lagrangian multiplier), scalar.
        :param ob_space             : Shape of the observation space.
        :param ac_space             : Shape of the action space. (must be 1-dimensional)
        :param extractor            : Class of the extractor network.
        :param extractor_params     : Keyword arguments for the extractor network. (optional)
        :param decoder         : Class of the decoder network. (optional)
        :param decoder_params  : Keyword arguments for the decoder network.
        """
        super(Lambda, self).__init__()
        self.ac_size = ac_space

        self.decoder = None
        if decoder is not None:
            self.decoder = decoder(ob_space, **decoder_params)
            self.ob_size = np.prod(np.array(output_shape(ob_space, decoder_params)), dtype=np.int)
        else:
            self.decoder = None
            self.ob_size = np.prod(np.array(ob_space), dtype=np.int)

        self.extractor = extractor(self.ob_size, **extractor_params)
        self.feature_size = np.prod(output_shape(self.ob_size, extractor_params), dtype=np.int)
        self.lambda_layer = nn.Linear(self.feature_size, 1)

    def forward(self, observation):
        if self.decoder is not None:
            state = self.decoder(observation)
        else:
            state = observation
        feature = self.extractor(state)
        log_lambda = self.lambda_layer(feature.view(-1, self.feature_size))
        return clip_but_pass_gradient(log_lambda, lower=-10., upper=6.)
