import os.path
import numpy as np

from lyapunov_reachability.speculation_tabular.base import QBase
import cplex
from cplex.exceptions import CplexSolverError


class ExplorerQAgent(QBase):

    def __init__(
            self, env, confidence, nb_states, nb_actions, initial_policy, terminal_states, seed=None, strict_done=True,
            safe_init=False, baseline_dir=None, baseline_step=None, save_dir='../../spec-tb-lyapunov'):
        self.operative_q = np.ones((nb_states, nb_actions))
        self.operative_q[terminal_states] = 0.
        self.time_q = np.zeros((nb_states, nb_actions))
        self.lyapunov_q = self.operative_q * 1.
        self.auxiliary_cost = 0.
        super(ExplorerQAgent, self).__init__(
            env, confidence, nb_states, nb_actions, initial_policy, terminal_states, seed=seed, strict_done=strict_done,
            safe_init=safe_init, baseline_dir=baseline_dir, baseline_step=baseline_step, save_dir=save_dir)
    
    def load_baseline(self, baseline):
        data = np.load(baseline)
        self.reachability_q[:] = data['reachability_q']
        self.updates_q[:] = data['updates_q']
        if 'policy' in data.keys():
            self.policy = data['policy']
        else:
            safest_reachability = np.min(self.reachability_q, axis=-1, keepdims=True)
            self.policy[:] = (self.reachability_q - safest_reachability == 0.) * 1.
            self.policy[:] = self.policy / np.sum(self.policy, axis=-1, keepdims=True)
        self.operative_q[:] = data['reachability_q']
    
    def save_model(self, path):
        info = dict()
        info['policy'] = self.policy
        info['steps'] = self.steps

        # Q learning-specific properties
        info['reachability_q'] = self.reachability_q
        info['updates_q'] = self.updates_q
        info['operative_q'] = self.operative_q
        info['time_q'] = self.time_q
        info['lyapunov_q'] = self.lyapunov_q
        
        # Other values...
        info['auxiliary_cost'] = self.auxiliary_cost

        np.savez(path + '.npz', **info)

    def load_model(self, load_dir):
        data = np.load(os.path.join(load_dir, '{}.npz'.format(self.steps)))
        self.reachability_q = data['reachability_q']
        self.updates_q = data['updates_q']
        self.operative_q = data['operative_q']
        self.time_q = data['time_q']
        self.lyapunov_q = data['lyapunov_q']
        self.auxiliary_cost = data['auxiliary_cost']

    def step(self, state, **kwargs):
        try:
            rs = np.random.rand() 
            eps = kwargs.get('epsilon', 0.)
            if eps > rs:
                action = self.env.action_space.sample()
            elif rs < 1. - eps:
                action = np.argmin(self.reachability_q[state, :])
            else:
                action = np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
            return action
        except ValueError:
            print("Error: stochastic policy is not feasible. Policy=\t" + str(self.policy[state, :]))

    def extra_setup(self, steps, episode_length, improve_interval, log_interval, save_interval, **kwargs):
        self.time_q[:] = episode_length

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
            self.operative_q[state, :] = 1.
        else:
            self.reachability_q[state, action] =\
                (1. - _lr) * self.reachability_q[state, action] +\
                _lr * gamma * np.min(self.reachability_q[next_state, :]) * (1. - done)
            self.operative_q[state, action] =\
                (1. - _lr) * self.operative_q[state, action] +\
                _lr * gamma * np.sum(self.operative_q[next_state, :] * self.policy[next_state, :]) * (1. - done)
        self.time_q[state, action] =\
            (1. - _lr) * self.time_q[state, action] +\
            _lr * (1. + np.sum(self.time_q[next_state, :] * self.policy[next_state, :])) * (1. - done)
                
        # Improve the policy ------------------------------------------------------------------------------------------
        if t % improve_interval == 0:
            convergence_mask = np.min(self.updates_q, -1) > criterion
            self.updates_q[convergence_mask] *= 0.
            self._policy_improvement(1. * convergence_mask)
        return

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]
        _operative_v = np.sum(self.operative_q * self.policy, -1)
        _operative_t = np.sum(self.time_q * self.policy, -1)
        try:
            _max_reachability = np.max(_operative_v[_operative_v <= 1. - self.confidence])
        except ValueError:
            _max_reachability = 1. - self.confidence
        epsilon = ((1. - self.confidence) - _max_reachability) / np.max(_operative_t)

        _lyapunov_q = self.operative_q + self.time_q * epsilon
        invalid_indices = np.isnan(_lyapunov_q)
        valid_indices = ~invalid_indices
        self.lyapunov_q[valid_indices] = _lyapunov_q[valid_indices]
        self.lyapunov_q[invalid_indices] = self.operative_q[invalid_indices]
        
        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)

        # for state in converged_states:
        for state in range(self.nb_states):
            c.variables.delete()
            c.linear_constraints.delete()

            # Objective: Minimize -pi(*|x) * Q_L(x,*) for each x.* Get the most aggressive*
            # Bounds: pi(a|x) >= 0 for all a (same as default setting)
            obj = - (self.reachability_q[state, :] - np.min(self.reachability_q[state, :]))
            lb = [0.0] * self.nb_actions
            indices = list(c.variables.add(obj=list(obj), lb=lb))

            # Subject to: (1) sum(pi(*|x)) == 1, (2) pi(*|x) * Q_L(x,*) <= L(x)
            # (2) is inequality, (1) is equality constraint. ("L")
            A = [cplex.SparsePair(indices[:], [1.] * self.nb_actions)]
            b = [1.]
            senses = ["E"]
            # (2) only applies when the state is safe.
            A.append(cplex.SparsePair(indices[:], list(self.lyapunov_q[state, :])))
            b.append(np.sum(self.lyapunov_q[state, :] * self.policy[state, :]) + epsilon)
            senses.append("L")
            c.linear_constraints.add(lin_expr=A, senses=senses, rhs=b)
            try:
                c.solve()
                _answer = np.array(c.solution.get_values())
                if np.sum(_answer) == 1. and np.sum(_answer > 1.) == 0 and np.sum(_answer < 0.) == 0:
                    self.policy[state, :] = _answer
            except CplexSolverError:
                print("Error: unable to find feasible policy at [state ID: %d]." % state)
        return
