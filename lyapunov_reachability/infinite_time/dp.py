import os.path
from timeit import default_timer as timer
import pickle
import numpy as np

#from scipy.optimize import linprog
import cplex
from cplex.exceptions import CplexSolverError


class SafeDPAgentBase(object):

    stats = ['runtime', 'value_function', 'probabilistic_safety']

    def __init__(self, nb_states, nb_actions, kernel, reward, safety, policy, confidence, gamma, episode_length,):
        self.env = None
        self.nb_states = nb_states      # S
        self.nb_actions = nb_actions    # A
        self.kernel = kernel            # transition kernel, S x A x S'
        self.reward = reward            # reward, S x A
        self.safety = safety            # safety, S (1: safe, 0: unsafe)
        self.policy = policy            # policy, S x A (initially safe)
        self.confidence = confidence    # safety confidence,
        self.gamma = gamma              # discount factor, should be (strictly) less than 1
        self.episode_length = episode_length
        self.iterations = 0

    def step(self, state):
        try:
            return np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
        except ValueError:
            print("Error: stochastic policy is not feasible. Policy=\t" + str(self.policy[state, :]))

    def run(self, N, print_freq=1, save_freq=1, verbose=True, name='default', **kwargs):
        result = dict()
        for stat in self.__class__.stats:
            result[stat] = list()
        result['runtime'].append(0.0)

        agent_dir = os.path.join(os.path.abspath(os.getcwd()), name)
        if not os.path.exists(agent_dir):
            os.mkdir(agent_dir)
            f = open(os.path.join(agent_dir, 'env.pkl'), 'wb')
            pickle.dump(self.env, f)
            f.close()

        for n in range(N):
            tic = timer()
            self._iteration(verbose, **kwargs)
            toc = timer() - tic

            self.iterations += 1

            result['runtime'].append(result['runtime'][-1] + toc)

            self._log_status(result, self.__class__.stats)
            _size = np.sum(result['probabilistic_safety'][-1] > self.confidence)
            if n % save_freq == 0:
                self.save(os.path.join(agent_dir, '-%d.json' % self.iterations))
            if n % print_freq == 0:
                print("Iteration: %d; Safe set size: %d" % (n, _size))
            if np.sum(result['probabilistic_safety'][-1] > self.confidence) == np.sum(self.safety):
                print("[Finished]")
                break

        del result['runtime'][0]
        return result

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, directory, iteration):
        raise NotImplementedError

    def _log_status(self, log, stats):
        raise NotImplementedError

    def _iteration(self, verbose, **kwargs):
        raise NotImplementedError

    def _reward_value_op(self, v, reward=None):
        reward = self.reward if reward is None else reward
        return np.sum(reward * self.policy, -1) +\
               np.sum(np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1) * np.expand_dims(v, 0), -1) * self.gamma

    def _safety_value_op(self, v, safety=None):
        safety = self.safety if safety is None else safety
        return safety * np.sum(np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1) * np.expand_dims(v, 0), -1)

    def _reward_q_op(self, q, reward=None):
        reward = self.reward if reward is None else reward
        _value = np.expand_dims(np.expand_dims(np.sum(q * self.policy, -1), 0), 0)
        return reward + np.sum(self.kernel * _value, -1) * self.gamma

    def _safety_q_op(self, q, safety=None):
        safety = self.safety if safety is None else safety
        _value = np.expand_dims(np.expand_dims(np.sum(q * self.policy, -1), 0), 0)
        return safety * np.sum(self.kernel * _value, -1)

    def _reward_value_evaluation(self):
        # Get value function that is the solution of Bellman equation.
        state_transition = np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1)
        v_ftn = np.zeros((self.nb_states,))

        if self.episode_length is None:
            # May raise np.linalg.LinAlgError...
            v_ftn = np.linalg.solve(np.eye(self.nb_states) - state_transition,
                                    np.sum(self.policy * self.reward, -1) * self.gamma)
            return v_ftn
        else:
            # Repeat Bellman operator till episode_length.
            for i in range(self.episode_length):
                v_ftn = self._reward_value_op(v_ftn)
            return v_ftn

    def _safety_value_evaluation(self):
        episode_length = self.episode_length
        if self.episode_length is None:
            episode_length = 1. / (1. - self.gamma)

        v_ftn = 1. * self.safety
        for i in range(episode_length):
            v_ftn = self._safety_value_op(v_ftn)
        return v_ftn


class SimplePIAgent(SafeDPAgentBase):
    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, episode_length):
        super(SimplePIAgent, self).__init__(
            nb_states, nb_actions, kernel, reward, safety, policy, confidence, gamma, episode_length,)
        self.env = env
        self.objective_v = np.zeros(self.nb_states,)
        self.safety_v = np.zeros(self.nb_states, )

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
        info['episode_length'] = self.episode_length
        info['iterations'] = self.iterations

        # PI-specific properties
        info['objective_v'] = self.objective_v
        info['safety_v'] = self.safety_v

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
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['safety_v']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy,
                    confidence, gamma, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        return log

    def _iteration(self, verbose, **kwargs):
        # Evaluate value functions
        self.objective_v = self._reward_value_evaluation()
        self.safety_v = self._safety_value_evaluation()
        # Update the policy
        self.policy = self._safety_policy_improvement()
        return

    def _safety_policy_improvement(self):
        _policy = np.zeros((self.nb_states, self.nb_actions))

        for state in range(self.nb_states):
            if self.safety[state] > 0:
                optimal_action = np.argmax(np.sum(self.kernel[state, :, :] * np.expand_dims(self.safety_v, 0), -1))
                _policy[state, optimal_action] = 1.
            else:
                _policy[state, :] = 1. / self.nb_actions
        return _policy

    def _constrained_policy_improvement(self, verbose):
        _policy = np.zeros((self.nb_states, self.nb_actions))

        c = cplex.Cplex()
        if not verbose:
            c.set_log_stream(None)
            c.set_error_stream(None)
            c.set_warning_stream(None)
            c.set_results_stream(None)

        for state in range(self.nb_states):
            c.variables.delete()
            c.linear_constraints.delete()

            # Objective: Minimize -T_*,R[V_pi_k](x) for each x.
            # Bounds: pi(a|x) >= 0 for all a (same as default setting)
            obj = - self.reward[state, :] - np.sum(self.kernel[state, :, :] * np.expand_dims(self.objective_v, 0), 1)
            lb = [0.0] * self.nb_actions
            indices = list(c.variables.add(obj=list(obj), lb=lb))

            # Subject to: (1) sum(pi(*|x)) == 1, (2) T_*,S[S_k](x) >= S_k(x)
            # (2) is inequality, (1) is equality constraint. ("G", not "L")
            A = [cplex.SparsePair(indices[:], [1.] * self.nb_actions)]
            b = [1.]
            senses = ["E"]
            # (2) only applies when the state is safe.
            if self.safety[state] > 0:
                A.append(cplex.SparsePair(
                    indices[:], list(np.sum(self.kernel[state, :, :] * np.expand_dims(self.safety_v, 0), -1))))
                b.append(self.safety_v[state])
                senses.append("G")
            c.linear_constraints.add(lin_expr=A, senses=senses, rhs=b)
            try:
                c.solve()
                _policy[state, :] = np.array(c.solution.get_values())
            except CplexSolverError:
                print("Error: unable to find feasible policy at [state ID: %d]." % state)
                _policy[state, :] = self.policy[state, :]
        return _policy


class SafePIAgent(SafeDPAgentBase):

    stats = ['runtime', 'expected_return', 'value_function', 'probabilistic_safety', 'lyapunov']

    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, episode_length):
        super(SafePIAgent, self).__init__(
            nb_states, nb_actions, kernel, reward, safety, policy, confidence, gamma, episode_length,)
        self.env = env
        self.objective_v = np.zeros(self.nb_states,)
        self.safety_v = np.zeros(self.nb_states,)
        self.time_v = np.zeros(self.nb_states,)

        # Mappings defined as a function of extended state (defined in S x {0,1})
        self.extended_kernel = np.zeros((2 * self.nb_states, self.nb_actions, 2 * self.nb_states))
        for s_0 in range(self.nb_states):
            for s_1 in range(self.nb_states):
                # All cases in order: (x, x') = (1, 1) / (1, 0) / (0, 1) / (0, 0)
                self.extended_kernel[s_0, :, s_1] = self.kernel[s_0, :, s_1] * self.safety[s_1]
                self.extended_kernel[s_0, :, s_1 + self.nb_states] = self.kernel[s_0, :, s_1] * (1. - self.safety[s_1])
                self.extended_kernel[s_0 + self.nb_states, :, s_1] = 0.
                self.extended_kernel[s_0 + self.nb_states, :, s_1 + self.nb_states] = self.kernel[s_0, :, s_1]

        self.lyapunov = np.zeros((2 * self.nb_states,))
        self.extended_policy = np.ones((2 * self.nb_states, self.nb_actions)) * (1. / self.nb_actions)
        self.extended_policy[0:self.nb_states, :] = self.policy

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
        info['episode_length'] = self.episode_length
        info['iterations'] = self.iterations

        # PI-specific properties
        info['objective_v'] = self.objective_v
        info['safety_v'] = self.safety_v
        info['time_v'] = self.time_v
        info['lyapunov'] = self.lyapunov

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
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['safety_v']
        time_v = data['time_v']
        lyapunov = data['lyapunov']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy, confidence, gamma, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.time_v = time_v
        agent.lyapunov = lyapunov
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'value_function' in stats:
            log['value_function'].append(np.copy(self.objective_v))
        if 'probabilistic_safety' in stats:
            log['probabilistic_safety'].append(np.copy(self.safety_v))
        if 'lyapunov' in stats:
            log['lyapunov'].append(np.copy(self.lyapunov))
        return log

    def _iteration(self, verbose, **kwargs):
        # Evaluate value functions
        self.objective_v = self._reward_value_evaluation()
        self.safety_v = self._safety_value_evaluation()
        # Evaluate the Lyapunov function
        self.lyapunov = self._lyapunov_evaluation()
        # Update the policy
        self.extended_policy = self._lyapunov_policy_improvement()
        self.policy = self.extended_policy[0:self.nb_states, :]
        return

    def _lyapunov_evaluation(self):

        episode_length = self.episode_length
        if self.episode_length is None:
            episode_length = 1. / (1. - self.gamma)

        lyapunov = np.ones((2 * self.nb_states,))

        min_safety = np.min(self.safety_v[self.safety_v > self.confidence])
        epsilon = np.maximum((- self.confidence + min_safety) / self.episode_length, 0.)

        state_transition = np.sum(np.expand_dims(self.extended_policy, -1) * self.extended_kernel, 1)

        augmented_cost = np.ones((2, self.nb_states)) * epsilon
        augmented_cost[0, :] += (self.safety < 1.)
        augmented_cost = augmented_cost.flatten()

        for i in range(episode_length):
            lyapunov = augmented_cost + np.sum(state_transition * np.expand_dims(lyapunov, 0), -1)
        return lyapunov

    def _lyapunov_policy_improvement(self):
        _policy = np.zeros((2 * self.nb_states, self.nb_actions))

        for state in range(2 * self.nb_states):
            values = np.sum(self.extended_kernel[state, :, :] * np.expand_dims(self.lyapunov, 0), -1)
            minimum_value = np.min(values)
            optimal_actions = np.where(values == minimum_value)[0]
            _policy[state, optimal_actions] = 1. / optimal_actions.size
        return _policy

    def _constrained_policy_improvement(self, verbose):
        _policy = np.zeros((2 * self.nb_states, self.nb_actions))

        c = cplex.Cplex()
        if not verbose:
            c.set_log_stream(None)
            c.set_error_stream(None)
            c.set_warning_stream(None)
            c.set_results_stream(None)

        for state in range(self.nb_states):
            c.variables.delete()
            c.linear_constraints.delete()

            # Objective: Minimize -T_*,R[V_pi_k](x) for each x.
            # Bounds: pi(a|x) >= 0 for all a (same as default setting)
            obj = - self.reward[state, :] - np.sum(self.kernel[state, :, :] * np.expand_dims(self.objective_v, 0), 1)
            lb = [0.0] * self.nb_actions
            indices = list(c.variables.add(obj=list(obj), lb=lb))

            # Subject to: (1) sum(pi(*|x)) == 1, (2) T_*,D[L_k](x) <= L_k(x)
            # (2) is inequality, (1) is equality constraint. ("L")
            A = [cplex.SparsePair(indices[:], [1.] * self.nb_actions)]
            b = [1.]
            senses = ["E"]

            # TODO: perhaps this should apply only when the state is safe?
            A.append(cplex.SparsePair(
                indices[:], list(np.sum(self.kernel[state, :, :] * np.expand_dims(self.lyapunov[0:self.nb_states], 0), -1))))
            b.append(self.lyapunov[state])
            senses.append("L")

            c.linear_constraints.add(lin_expr=A, senses=senses, rhs=b)
            try:
                c.solve()
                _policy[state, :] = np.array(c.solution.get_values())
            except CplexSolverError:
                print("Error: unable to find feasible policy at [state ID: %d]." % state)
                _policy[state, :] = self.policy[state, :]
        return _policy


