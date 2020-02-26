import os.path
import numpy as np

from lyapunov_reachability.speculation_tabular.base import QBase


class DefaultQAgent(QBase):

    def __init__(self, env, confidence, nb_states, nb_actions, initial_policy, terminal_states, seed=None,
                 strict_done=True, safe_init=False, baseline_dir=None, baseline_step=None, save_dir='../../spec-tb-default'):
        """
        A simple Q-learning-like (actually, actor-critic-like) agent.
        It improves the policy to be the argmin of reachability Q-function.
        Note that, unlike the older version, it improves its policy for all states.
        """
        super(DefaultQAgent, self).__init__(
            env, confidence, nb_states, nb_actions, initial_policy, terminal_states, seed=seed, strict_done=strict_done,
            safe_init=safe_init, baseline_dir=baseline_dir, baseline_step=baseline_step, save_dir=save_dir)

    def save_model(self, path):
        info = dict()
        info['policy'] = self.policy
        info['steps'] = self.steps

        # Q learning-specific properties
        info['reachability_q'] = self.reachability_q
        info['updates_q'] = self.updates_q

        np.savez(path + '.npz', **info)

    def load_model(self, load_dir):
        data = np.load(os.path.join(load_dir, '{}.npz'.format(self.steps)))
        self.reachability_q = data['reachability_q']
        self.updates_q = data['updates_q']

    def step(self, state, **kwargs):
        try:
            eps = kwargs.get('epsilon', 0.)
            action = np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
            if eps > np.random.rand():
                action = self.env.action_space.sample()
            return action
        except ValueError:
            print("Error: stochastic policy is not feasible. Policy=\t" + str(self.policy[state, :]))

    def _log_auxiliary(self, **kwargs):
        return

    def _iteration(self, t, state, action, next_state, safety, done, **kwargs):
        improve_interval = kwargs.get('improve_interval', 1)
        lr = kwargs.get('learning_rate', 1.)
        gamma = kwargs.get('gamma', .99)
        criterion = kwargs.get('criterion', 1e2)

        # Approximate the Q-functions ---------------------------------------------------------------------------------
        self.updates_q[state, action] += 1.
        _lr = lr / (0.99 + 0.01 * self.updates_q[state, action])

        if safety == 0.:
            self.reachability_q[state, :] = 1.
        else:
            self.reachability_q[state, action] =\
                (1. - _lr) * self.reachability_q[state, action] +\
                _lr * gamma * np.sum(self.reachability_q[next_state, :] * self.policy[next_state, :]) * (1. - done)

        # Improve the policy ------------------------------------------------------------------------------------------
        if t % improve_interval == 0:
            convergence_mask = np.min(self.updates_q, -1) > criterion
            self.updates_q[convergence_mask] *= 0.
            self._policy_improvement(1. * convergence_mask)
        return

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]
        # for state in converged_states:
        for state in range(self.nb_states):
            # Then update the policy.
            minimum_value = np.min(self.reachability_q[state, :])
            optimal_actions = np.where(self.reachability_q[state, :] == minimum_value)[0]

            self.policy[state, :] = 0.
            self.policy[state, optimal_actions] = 1. / optimal_actions.size
        return
