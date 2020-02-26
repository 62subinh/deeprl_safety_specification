import os.path
import pickle
import numpy as np
from timeit import default_timer as timer

from lyapunov_reachability.shortest_path.dp import SafeDPAgentBase
import cplex
from cplex.exceptions import CplexSolverError


class SimpleQAgent(SafeDPAgentBase):

    stats = ['runtime', 'value_function', 'probabilistic_safety', 'approximate_reachability', '']

    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, initial_state, terminal_states, episode_length):
        """
        A simple Q-learning-like (actually, actor-critic-like) agent.
        It improves the policy to be the argmin of reachability Q-function.
        Note that, unlike the older version, it improves its policy for all states.
        """
        super(SimpleQAgent, self).__init__(
            nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)
        self.env = env
        self.iterations = 0

        # Extended MDP
        self.extended_kernel = np.zeros((2 * self.nb_states, self.nb_actions, 2 * self.nb_states))
        for s_0 in range(self.nb_states):
            for s_1 in range(self.nb_states):
                # All cases in order: (x, x') = (1, 1) / (1, 0) / (0, 1) / (0, 0)
                self.extended_kernel[s_0, :, s_1] = self.kernel[s_0, :, s_1] * self.safety[s_1]
                self.extended_kernel[s_0, :, s_1 + self.nb_states] = self.kernel[s_0, :, s_1] * (1. - self.safety[s_1])
                if self.episode_length is None:
                    self.extended_kernel[s_0 + self.nb_states, :, self.terminal_states[0] + self.nb_states] = 1.
                else:
                    self.extended_kernel[s_0 + self.nb_states, :, s_1] = 0.
                    self.extended_kernel[s_0 + self.nb_states, :, s_1 + self.nb_states] = self.kernel[s_0, :, s_1]

        self.extended_policy = np.ones((2 * self.nb_states, self.nb_actions)) * (1. / self.nb_actions)
        self.extended_policy[0:self.nb_states, :] = self.policy

        # Q-functions (for learning)
        self.unsafety_q = np.ones((2 * self.nb_states, self.nb_actions))
        self.unsafety_q[self.terminal_states, :] = 0.

        self.time_q = np.zeros((2 * self.nb_states, self.nb_actions))

        self.updates = np.zeros((2 * self.nb_states, self.nb_actions))

        # Value functions (for check only)
        self.objective_v = None
        self.safety_v = None

    def save(self, path):
        info = dict()

        # Common properties
        info['nb_states'] = self.nb_states
        info['nb_actions'] = self.nb_actions
        info['kernel'] = self.kernel
        info['reward'] = self.reward
        info['safety'] = self.safety
        info['policy'] = self.policy
        info['confidence'] = self.confidence
        info['gamma'] = self.gamma
        info['initial_state'] = self.initial_state
        info['terminal_states'] = self.terminal_states
        info['episode_length'] = self.episode_length
        info['iterations'] = self.iterations

        # Q learning-specific properties
        info['objective_v'] = self.objective_v
        info['safety_v'] = self.safety_v
        info['unsafety_q'] = self.unsafety_q
        info['time_q'] = self.time_q
        info['updates'] = self.updates

        np.savez(path + '.npz', **info)
        return

    @classmethod
    def load(cls, directory, iteration):
        f = open(os.path.join(directory, 'env'), 'r')
        env = pickle.load(f)
        f.close()

        data = np.load(os.path.join(directory, '%d.npz'))

        # Common properties
        nb_states = data['nb_states']
        nb_actions = data['nb_actions']
        kernel = data['kernel']
        reward = data['reward']
        safety = data['safety']
        policy = data['policy']
        confidence = data['confidence']
        gamma = data['gamma']
        initial_state = data['initial_state']
        terminal_states = data['terminal_states']
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['lyapunov']
        unsafety_q = data['unsafety_q']
        time_q = data['time_q']
        updates = data['updates']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy,
                    confidence, gamma, initial_state, terminal_states, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.unsafety_q = unsafety_q
        agent.time_q = time_q
        agent.updates = updates
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        if 'approximate_reachability' in stats:
            log['approximate_reachability'].append(np.copy(np.sum(self.unsafety_q * self.extended_policy, -1)))
        return log

    def _iteration(self, verbose, **kwargs):
        iteration_length = kwargs.get('iteration_length', self.episode_length)
        epsilon_greedy = kwargs.get('epsilon_greedy', 0.)
        lr = kwargs.get('learning_rate', 1.)
        criterion = kwargs.get('criterion', 1e2)

        # Evaluate value functions (for safety check only) ------------------------------------------------------------
        self.objective_v = self._reward_value_evaluation()
        self.safety_v = self._safety_value_evaluation()

        # Approximate the Q-functions ---------------------------------------------------------------------------------
        self.env.reset()
        state = self.initial_state
        extra_state = 1
        steps = 0

        for t in range(iteration_length):
            # Fetch a sample state transition tuple.
            if np.random.rand() > epsilon_greedy:
                action = self.step(state)
            else:
                action = self.env.action_space.sample()
            _, reward, done, info = self.env.step(action)
            next_state = info['state']
            safety = info['safety']

            _done = (done + (steps == self.episode_length)) > 0.
            _state = state + self.nb_states * (1 - extra_state)
            _next_state = next_state + self.nb_states * (1 - extra_state * (safety > 0.))
            _indicator = (1. - safety) * extra_state

            # Soft update Q-functions
            self.updates[_state, action] += 1.
            _lr = lr / (0.99 + 0.01 * self.updates[_state, action])

            if _indicator > 0.:
                self.unsafety_q[_state, :] = 1.
            else:
                self.unsafety_q[_state, action] =\
                    (1. - _lr) * self.unsafety_q[_state, action] + _lr * _indicator +\
                    _lr * np.sum(self.unsafety_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1.-_done)
            self.time_q[_state, action] =\
                (1. - _lr) * self.time_q[_state, action] + _lr * 1. +\
                _lr * np.sum(self.time_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1.-_done)

            if _done or _indicator > 0.:
                self.env.reset()
                state = self.initial_state
                extra_state = 1
                steps = 0
            else:
                state = next_state
                extra_state = extra_state * (safety > 0.)
                steps += 1

        # Improve the policy ------------------------------------------------------------------------------------------
        convergence_mask = np.min(self.updates, -1) > criterion
        self.updates[convergence_mask] *= 0.
        self._policy_improvement(1. * convergence_mask)
        self.policy = self.extended_policy[0:self.nb_states, :]
        return

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]
        # for state in converged_states:
        for state in range(self.nb_states):
            # Then update the policy.
            minimum_value = np.min(self.unsafety_q[state, :])
            optimal_actions = np.where(self.unsafety_q[state, :] == minimum_value)[0]

            self.extended_policy[state, :] = 0.
            self.extended_policy[state, optimal_actions] = 1. / optimal_actions.size
        return


class SafeQAgent(SimpleQAgent):

    stats = ['runtime', 'value_function', 'probabilistic_safety', 'approximate_reachability', 'lyapunov']

    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, initial_state, terminal_states, episode_length):
        """
        A simple Q-learning-like (actually, actor-critic-like) agent.
        It improves the policy to be the argmin of reachability Lyapunov Q-function.
        Reachability Lyapunov function is a function that is an expected sum of 'augmented' costs;
        that is, it is a sum of reachability Q-function & the weighted running time Q-function.
        Note that, unlike the older version, it improves its policy for all states.
        """
        super(SafeQAgent, self).__init__(
            env, nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)

        self.lyapunov_q = np.ones((2 * self.nb_states, self.nb_actions))

    def save(self, path):
        info = dict()

        # Common properties
        info['nb_states'] = self.nb_states
        info['nb_actions'] = self.nb_actions
        info['kernel'] = self.kernel
        info['reward'] = self.reward
        info['safety'] = self.safety
        info['policy'] = self.policy
        info['confidence'] = self.confidence
        info['gamma'] = self.gamma
        info['initial_state'] = self.initial_state
        info['terminal_states'] = self.terminal_states
        info['episode_length'] = self.episode_length
        info['iterations'] = self.iterations

        # Q learning-specific properties
        info['objective_v'] = self.objective_v
        info['safety_v'] = self.safety_v
        info['unsafety_q'] = self.unsafety_q
        info['time_q'] = self.time_q
        info['lyapunov_q'] = self.lyapunov_q
        info['updates'] = self.updates

        np.savez(path + '.npz', **info)
        return

    @classmethod
    def load(cls, directory, iteration):
        f = open(os.path.join(directory, 'env'), 'r')
        env = pickle.load(f)
        f.close()

        data = np.load(os.path.join(directory, '%d.npz'))

        # Common properties
        nb_states = data['nb_states']
        nb_actions = data['nb_actions']
        kernel = data['kernel']
        reward = data['reward']
        safety = data['safety']
        policy = data['policy']
        confidence = data['confidence']
        gamma = data['gamma']
        initial_state = data['initial_state']
        terminal_states = data['terminal_states']
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['lyapunov']
        unsafety_q = data['unsafety_q']
        time_q = data['time_q']
        lyapunov_q = data['lyapunov_q']
        updates = data['updates']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy,
                    confidence, gamma, initial_state, terminal_states, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.unsafety_q = unsafety_q
        agent.time_q = time_q
        agent.lyapunov_q = lyapunov_q
        agent.updates = updates
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        if 'approximate_reachability' in stats:
            log['approximate_reachability'].append(np.copy(np.sum(self.unsafety_q * self.extended_policy, -1)))
        if 'lyapunov' in stats:
            log['lyapunov'].append(np.copy(np.sum(self.lyapunov_q * self.extended_policy, -1)))
        return log

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]

        _unsafety_v = np.sum(self.unsafety_q * self.extended_policy, -1)[converged_states]
        try:
            _max_unsafety_v = np.max(_unsafety_v[_unsafety_v <= 1. - self.confidence])
        except ValueError:
            _max_unsafety_v = 1. - self.confidence
        epsilon = ((1. - self.confidence) - _max_unsafety_v) / np.max(self.time_q)

        self.lyapunov_q[:] = np.minimum(self.lyapunov_q, self.unsafety_q + self.time_q * epsilon)

        converged_states = np.where(convergence_mask > 0.)[0]
        # for state in converged_states:
        for state in range(self.nb_states):
            # Then update the policy.
            minimum_value = np.min(self.lyapunov_q[state, :])
            optimal_actions = np.where(self.lyapunov_q[state, :] == minimum_value)[0]

            self.extended_policy[state, :] = 0.
            self.extended_policy[state, optimal_actions] = 1. / optimal_actions.size
        return


class MarginalSimpleQAgent(SafeQAgent):
    stats = ['runtime', 'value_function', 'probabilistic_safety', 'approximate_reachability',
             'lyapunov', 'average_safety']

    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, initial_state, terminal_states, episode_length):
        """
        An agent similar to SimpleQAgent,
        but it updates its policy to be 'marginally' safe.
        """
        super(MarginalSimpleQAgent, self).__init__(
            env, nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)
        # Default policy is the most safe, whereas exploration_policy is the least safe one.
        self.average_safety = None

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        if 'approximate_reachability' in stats:
            log['approximate_reachability'].append(np.copy(np.sum(self.unsafety_q * self.extended_policy, -1)))
        if 'lyapunov' in stats:
            log['lyapunov'].append(np.copy(np.sum(self.lyapunov_q * self.extended_policy, -1)))
        if 'average_safety' in stats:
            log['average_safety'].append(self.average_safety)
        return log

    def _iteration(self, verbose, **kwargs):
        iteration_length = kwargs.get('iteration_length', self.episode_length)
        epsilon_greedy = kwargs.get('epsilon_greedy', 0.)
        lr = kwargs.get('learning_rate', 1.)
        criterion = kwargs.get('criterion', 1e2)

        # Evaluate value functions (for safety check only) ------------------------------------------------------------
        self.objective_v = self._reward_value_evaluation()
        self.safety_v = self._safety_value_evaluation()

        # Approximate the Q-functions ---------------------------------------------------------------------------------
        self.env.reset()
        state = self.initial_state
        extra_state = 1
        steps = 0
        self.average_safety = 0.

        for t in range(iteration_length):
            # Fetch a sample state transition tuple.
            if np.random.rand() > epsilon_greedy:
                action = np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
            else:
                action = self.env.action_space.sample()
            _, reward, done, info = self.env.step(action)
            next_state = info['state']
            safety = info['safety']

            _done = done * extra_state
            _state = state + self.nb_states * (1 - extra_state)
            _next_state = next_state + self.nb_states * (1 - extra_state * (safety > 0.))
            _indicator = (1. - safety) * extra_state

            # Soft update Q-functions
            self.average_safety += 1. * (safety < 1.)
            self.updates[_state, action] += 1.
            _lr = lr / self.updates[_state, action]

            if _indicator > 0.:
                self.unsafety_q[_state, :] = 1.
            else:
                self.unsafety_q[_state, action] =\
                    (1. - _lr) * self.unsafety_q[_state, action] + _lr * _indicator +\
                    _lr * np.sum(self.unsafety_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1.-done)
            self.time_q[_state, action] =\
                (1. - _lr) * self.time_q[_state, action] + _lr * 1. +\
                _lr * np.sum(self.time_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1. - done)

            if done or steps == self.episode_length:
                self.env.reset()
                state = self.initial_state
                extra_state = 1
                steps = 0
            else:
                state = next_state
                extra_state = extra_state * (safety > 0.)
                steps += 1

        # Improve the policy ------------------------------------------------------------------------------------------
        self.average_safety /= iteration_length
        convergence_mask = np.min(self.updates, -1) > criterion

        self.updates[convergence_mask] *= 0.
        self._policy_improvement(1. * convergence_mask)
        self.policy = self.extended_policy[0:self.nb_states, :]
        return

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]

        _unsafety_v = np.sum(self.unsafety_q * self.extended_policy, -1)[converged_states]
        try:
            _max_unsafety_v = np.max(_unsafety_v[_unsafety_v <= 1. - self.confidence])
        except ValueError:
            _max_unsafety_v = 1. - self.confidence
        epsilon = ((1. - self.confidence) - _max_unsafety_v) / np.max(self.time_q)

        self.lyapunov_q[:] = np.minimum(self.lyapunov_q, self.unsafety_q + self.time_q * epsilon)

        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)

        # Then update the policy.
        for state in range(self.nb_states):
            if np.sum(convergence_mask == state) > 0 and np.min(self.unsafety_q[state, :]) <= 1. - self.confidence:
                c.variables.delete()
                c.linear_constraints.delete()

                # Objective: Minimize -pi(*|x) * Q_L(x,*) for each x.
                # Bounds: pi(a|x) >= 0 for all a (same as default setting)
                obj = - (self.unsafety_q[state, :] - np.min(self.unsafety_q[state, :]))
                lb = [0.0] * self.nb_actions
                indices = list(c.variables.add(obj=list(obj), lb=lb))

                # Subject to: (1) sum(pi(*|x)) == 1, (2) pi(*|x) * Q_L(x,*) <= L_k(x) + epsilon
                # (2) is inequality, (1) is equality constraint. ("L")
                A = [cplex.SparsePair(indices[:], [1.] * self.nb_actions)]
                b = [1.]
                senses = ["E"]
                # (2) only applies when the state is safe.
                A.append(cplex.SparsePair(
                    indices[:], list(self.unsafety_q[state, :])))
                b.append(np.sum(self.unsafety_q[state, :] * self.extended_policy[state, :], -1))
                senses.append("L")
                c.linear_constraints.add(lin_expr=A, senses=senses, rhs=b)
                try:
                    c.solve()
                    _answer = np.array(c.solution.get_values())
                    if np.sum(_answer) == 1. and np.sum(_answer > 1.) == 0 and np.sum(_answer < 0.) == 0:
                        self.extended_policy[state, :] = _answer
                except CplexSolverError:
                    print("Error: unable to find feasible policy at [state ID: %d]." % state)
            else:
                minimum_value = np.min(self.lyapunov_q[state, :])
                optimal_actions = np.where(self.lyapunov_q[state, :] == minimum_value)[0]

                self.extended_policy[state, :] = 0.
                self.extended_policy[state, optimal_actions] = 1. / optimal_actions.size

            self.extended_policy[self.nb_states + state, :] = self.extended_policy[state, :]


class SafeExplorerQAgent(SafeQAgent):

    stats = ['runtime', 'value_function', 'probabilistic_safety', 'approximate_reachability',
             'lyapunov', 'average_safety']

    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, initial_state, terminal_states, episode_length):
        """
        Here, Lyapunov function is used to approximate the reachability value function of
            marginally safe policy with which we use to explore the environment.
        Approximated optimal reachability (= approximated maximal safety) is not affecting the exploration itself,
            but it is used to ensure the next policy to be marginally 'safe.'
        """
        super(SafeExplorerQAgent, self).__init__(
            env, nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)
        # Default policy is the most safe, whereas exploration_policy is the least safe one.
        self.average_safety = None

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        if 'approximate_reachability' in stats:
            log['approximate_reachability'].append(np.copy(np.sum(self.unsafety_q * self.extended_policy, -1)))
        if 'lyapunov' in stats:
            log['lyapunov'].append(np.copy(np.sum(self.lyapunov_q * self.extended_policy, -1)))
        if 'average_safety' in stats:
            log['average_safety'].append(self.average_safety)
        return log

    def _iteration(self, verbose, **kwargs):
        iteration_length = kwargs.get('iteration_length', self.episode_length)
        epsilon_greedy = kwargs.get('epsilon_greedy', 0.)
        lr = kwargs.get('learning_rate', 1.)
        criterion = kwargs.get('criterion', 1e2)
        exploratory_init = kwargs.get('exploratory_init', 0.)

        # Evaluate value functions (for safety check only) ------------------------------------------------------------
        self.objective_v = self._reward_value_evaluation()
        self.safety_v = self._safety_value_evaluation()

        # Approximate the Q-functions ---------------------------------------------------------------------------------
        self.env.reset()
        if np.random.random() < exploratory_init:
            _unsafety = np.min(self.unsafety_q[:self.nb_states], -1)
            state = np.argmax(_unsafety - (_unsafety > 1. - self.confidence))
            self.env.forced_step(state)
        else:
            state = self.initial_state
        extra_state = 1
        steps = 0
        self.average_safety = 0.

        for t in range(iteration_length):
            # Fetch a sample state transition tuple.
            if np.random.rand() > epsilon_greedy:
                action = np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
            else:
                action = self.env.action_space.sample()
            _, reward, done, info = self.env.step(action)
            next_state = info['state']
            safety = info['safety']

            _done = (done + (steps == self.episode_length)) > 0.
            _state = state + self.nb_states * (1 - extra_state)
            _next_state = next_state + self.nb_states * (1 - extra_state * (safety > 0.))
            _indicator = (1. - safety) * extra_state

            # Soft update Q-functions
            self.average_safety += 1. * (safety < 1.)
            self.updates[_state, action] += 1.
            _lr = lr / (0.99 + 0.01 * self.updates[_state, action])

            if _indicator > 0.:
                self.unsafety_q[_state, :] = 1.
                self.lyapunov_q[_state, :] = 1.
            else:
                self.unsafety_q[_state, action] =\
                    (1. - _lr) * self.unsafety_q[_state, action] + _lr * _indicator +\
                    _lr * np.min(self.unsafety_q[_next_state, :]) * (1.-_done)
                self.lyapunov_q[_state, action] = \
                    (1. - _lr) * self.unsafety_q[_state, action] + _lr * _indicator + \
                    _lr * np.sum(self.unsafety_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1.-_done)
            self.time_q[_state, action] =\
                (1. - _lr) * self.time_q[_state, action] + _lr * 1. +\
                _lr * np.sum(self.time_q[_next_state, :] * self.extended_policy[_next_state, :]) * (1.-_done)

            if _done or _indicator > 0.:
                self.env.reset()
                if np.random.random() < exploratory_init:
                    _unsafety = np.min(self.unsafety_q[:self.nb_states], -1)
                    state = np.argmax(_unsafety - (_unsafety > 1. - self.confidence))
                    self.env.forced_step(state)
                else:
                    state = self.initial_state
                extra_state = 1
                steps = 0
            else:
                state = next_state
                extra_state = extra_state * (safety > 0.)
                steps += 1

        # Improve the policy ------------------------------------------------------------------------------------------
        self.average_safety /= iteration_length
        convergence_mask = np.min(self.updates, -1) > criterion

        self.updates[convergence_mask] *= 0.
        self._policy_improvement(1. * convergence_mask)
        self.policy = self.extended_policy[0:self.nb_states, :]
        return

    def _policy_improvement(self, convergence_mask):
        converged_states = np.where(convergence_mask > 0.)[0]

        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)

        # Then update the policy.
        for state in range(self.nb_states):
            if np.sum(converged_states == state) > 0 and np.min(self.unsafety_q[state, :]) <= 1. - self.confidence:
                c.variables.delete()
                c.linear_constraints.delete()

                # Objective: Minimize -pi(*|x) * Q_L(x,*) for each x.
                # Bounds: pi(a|x) >= 0 for all a (same as default setting)
                obj = - (self.unsafety_q[state, :] - np.min(self.unsafety_q[state, :]))
                lb = [0.0] * self.nb_actions
                indices = list(c.variables.add(obj=list(obj), lb=lb))

                # Subject to: (1) sum(pi(*|x)) == 1, (2) pi(*|x) * Q_L(x,*) <= L_k(x) + epsilon
                # (2) is inequality, (1) is equality constraint. ("L")
                A = [cplex.SparsePair(indices[:], [1.] * self.nb_actions)]
                b = [1.]
                senses = ["E"]
                # (2) only applies when the state is safe.
                A.append(cplex.SparsePair(
                    indices[:], list(self.lyapunov_q[state, :])))
                b.append(np.sum(self.lyapunov_q[state, :] * self.extended_policy[state, :], -1))
                senses.append("L")
                c.linear_constraints.add(lin_expr=A, senses=senses, rhs=b)
                try:
                    c.solve()
                    _answer = np.array(c.solution.get_values())
                    if np.sum(_answer) == 1. and np.sum(_answer > 1.) == 0 and np.sum(_answer < 0.) == 0:
                        self.extended_policy[state, :] = _answer
                except CplexSolverError:
                    print("Error: unable to find feasible policy at [state ID: %d]." % state)
            else:
                minimum_value = np.min(self.lyapunov_q[state, :])
                optimal_actions = np.where(self.lyapunov_q[state, :] == minimum_value)[0]

                self.extended_policy[state, :] = 0.
                self.extended_policy[state, optimal_actions] = 1. / optimal_actions.size

            self.extended_policy[self.nb_states + state, :] = self.extended_policy[state, :]

        return
