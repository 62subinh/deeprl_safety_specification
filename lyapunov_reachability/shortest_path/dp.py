import os.path
from timeit import default_timer as timer
import pickle
import numpy as np

#from scipy.optimize import linprog
import cplex
from cplex.exceptions import CplexSolverError


class SafeDPAgentBase(object):

    stats = ['runtime', 'expected_return', 'value_function', 'probabilistic_safety']

    def __init__(self, nb_states, nb_actions, kernel, reward, safety, policy, confidence,
                 gamma, initial_state, terminal_states, episode_length,):
        self.env = None
        self.nb_states = nb_states      # S
        self.nb_actions = nb_actions    # A
        self.kernel = kernel            # transition kernel, S x A x S'
        self.reward = reward            # reward, S x A
        self.safety = safety            # safety, S (1: safe, 0: unsafe)
        self.policy = policy            # policy, S x A (initially safe)
        self.confidence = confidence    # safety confidence,
        self.gamma = gamma              # discount factor,
        self.initial_state = initial_state
        self.terminal_states = terminal_states
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
            os.makedirs(agent_dir)
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

            if n % save_freq == 0:
                self.save(os.path.join(agent_dir, '-%d.json' % self.iterations))
            if n % print_freq == 0:
                print("Iteration: %d; Safe set size: %d" %
                      (n, np.sum(result['probabilistic_safety'][-1] > self.confidence)))

        self.save(os.path.join(agent_dir, '-%d-final.json' % self.iterations))
        print("[Finished]")

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

    def _time_value_op(self, v,):
        time = np.ones((self.nb_states,)) * 1.
        time[self.terminal_states] = 0.
        return time + np.sum(np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1) * np.expand_dims(v, 0), -1)

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

        # Note that, value function is 0 at terminal states.
        # Remove the known values from the value function vector, we have...
        if self.episode_length is None:
            # May raise np.linalg.LinAlgError...
            valid_states = np.arange(0, self.nb_states)
            valid_states = np.delete(valid_states, self.terminal_states)

            valid_state_transition = state_transition[valid_states, :]
            valid_state_transition = valid_state_transition[:, valid_states]
            valid_v_ftn = np.linalg.solve(np.eye(valid_states.size) - valid_state_transition,
                                          np.sum(self.policy * self.reward, -1)[valid_states] * self.gamma)
            # valid_v_ftn = np.matmul(np.linalg.inv(np.eye(valid_states.size) - valid_state_transition),
            #                         np.sum(self.policy * self.reward, -1)[valid_states])
            v_ftn[valid_states] = valid_v_ftn
            v_ftn[self.terminal_states] = 0.
            return v_ftn
        else:
            # Repeat Bellman operator till episode_length.
            for i in range(self.episode_length):
                v_ftn = self._reward_value_op(v_ftn)
            return v_ftn

    def _safety_value_evaluation(self):
        # Get safety value function that is the solution of Bellman equation.
        state_transition = np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1)
        v_ftn = np.zeros((self.nb_states,))

        # Note that, value function is 0 at unsafe states & 1 at terminal states.
        # Remove the known values from the value function vector, we have...
        if self.episode_length is None:
            valid_states = np.where(self.safety > 0)[0]
            for s in range(self.terminal_states.size):
                valid_states = valid_states[valid_states != self.terminal_states[s]]
            safe_state_transition = state_transition[valid_states, :]
            valid_state_transition = safe_state_transition[:, valid_states]
            terminal_state_transition = safe_state_transition[:, self.terminal_states]
            valid_v_ftn = np.linalg.solve(np.eye(valid_states.size) - valid_state_transition,
                                          np.sum(terminal_state_transition, -1))
            v_ftn[valid_states] = valid_v_ftn
            v_ftn[self.terminal_states] = 1.
            return v_ftn
        else:
            v_ftn = 1. * self.safety
            for i in range(self.episode_length):
                v_ftn = self._safety_value_op(v_ftn)
            return v_ftn

    def _time_value_evaluation(self):
        # Get expected first-hitting time that is the solution of Bellman equation.
        state_transition = np.sum(np.expand_dims(self.policy, -1) * self.kernel, 1)
        v_ftn = np.zeros((self.nb_states,))

        # Note that, value function is 0 at unsafe states & 1 at terminal states.
        # Remove the known values from the value function vector, we have...
        if self.episode_length is None:
            valid_states = np.delete(np.arange(0, self.nb_states), self.terminal_states)

            valid_state_transition = state_transition[valid_states, :]
            valid_state_transition = valid_state_transition[:, valid_states]
            valid_v_ftn = np.linalg.solve(np.eye(valid_states.size) - valid_state_transition,
                                          np.ones((valid_states.size,)))
            v_ftn[valid_states] = valid_v_ftn
            v_ftn[self.terminal_states] = 0.
            return v_ftn
        else:
            for i in range(self.episode_length):
                v_ftn = self._time_value_op(v_ftn)
            return v_ftn


class SimplePIAgent(SafeDPAgentBase):
    def __init__(self, env, nb_states, nb_actions, kernel, reward, safety, policy,
                 confidence, gamma, initial_state, terminal_states, episode_length):
        super(SimplePIAgent, self).__init__(
            nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)
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
        info['initial_state'] = self.initial_state
        info['terminal_states'] = self.terminal_states
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
        initial_state = data['initial_state']
        terminal_states = data['terminal_states']
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['lyapunov']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy,
                    confidence, gamma, initial_state, terminal_states, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'expected_return' in stats:
            log['expected_return'].append(self.objective_v[self.initial_state])
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
        self.policy = self._safety_policy_improvement(verbose)
        return

    def _safety_policy_improvement(self, verbose):
        _policy = np.zeros((self.nb_states, self.nb_actions))

        for state in range(self.nb_states):
            values = np.sum(self.kernel[state, :, :] * np.expand_dims(self.safety_v, 0), -1)
            maximum_value = np.max(values)
            optimal_actions = np.where(values == maximum_value)[0]
            _policy[state, optimal_actions] = 1. / optimal_actions.size

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

            # Subject to: (1) sum(pi(*|x)) == 1, (2) T_*,S[L_k](x) >= L_k(x)
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
                 confidence, gamma, initial_state, terminal_states, episode_length):
        super(SafePIAgent, self).__init__(
            nb_states, nb_actions, kernel, reward, safety, policy,
            confidence, gamma, initial_state, terminal_states, episode_length,)
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
                # Case1: when arrived at x=0, you are sent to terminal state! -----------------------------------------
                # self.extended_kernel[s_0 + self.nb_states, :, self.terminal_states[0] + self.nb_states] = 1.
                # Case2: transition from x=0 is similar to that of x=0! -----------------------------------------------
                self.extended_kernel[s_0 + self.nb_states, :, s_1] = 0.
                self.extended_kernel[s_0 + self.nb_states, :, s_1 + self.nb_states] = self.kernel[s_0, :, s_1]

        self.lyapunov = np.zeros((2 * self.nb_states,))
        self.extended_policy = np.concatenate((self.policy, self.policy), 0)

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
        initial_state = data['initial_state']
        terminal_states = data['terminal_states']
        episode_length = data['episode_length']
        iterations = data['iterations']

        # PI-specific properties
        objective_v = data['objective_v']
        safety_v = data['safety_v']
        time_v = data['time_v']
        lyapunov = data['lyapunov']

        data.close()

        agent = cls(env, nb_states, nb_actions, kernel, reward, safety, policy,
                    confidence, gamma, initial_state, terminal_states, episode_length)
        agent.objective_v = objective_v
        agent.safety_v = safety_v
        agent.time_v = time_v
        agent.lyapunov = lyapunov
        agent.iterations = iterations
        return agent

    def _log_status(self, log, stats):
        if 'expected_return' in stats:
            log['expected_return'].append(self.objective_v[self.initial_state])
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
        self.time_v = self._time_value_evaluation()
        # Evaluate the Lyapunov function
        self.lyapunov = self._lyapunov_evaluation()
        # Update the policy
        self.extended_policy = self._lyapunov_policy_improvement()
        self.policy = self.extended_policy[0:self.nb_states, :]
        return

    def _lyapunov_evaluation(self):
        lyapunov = np.ones((2 * self.nb_states,))

        # Note that, reachability is 0 at terminal states.
        # Remove the known values from the value function vector, we have...

        if self.episode_length is None:
            # Case1: Only difference of epsilon is at terminal state. -------------------------------------------------
            # epsilon = np.maximum(
            #     (self.safety_v[self.initial_state]- self.confidence) / (1. - self.safety_v[self.initial_state]), 0.)
            #
            # state_transition = np.sum(np.expand_dims(self.extended_policy, -1) * self.extended_kernel, 1)
            #
            # augmented_cost = np.zeros((self.nb_states, 2))
            # augmented_cost[:, 0] += (self.safety < 1.) * 1.
            # augmented_cost = augmented_cost.flatten()
            #
            # state_transition = state_transition[valid_states, :]
            # valid_state_transition = state_transition[:, valid_states]
            # unsafe_state_transition = np.sum(state_transition[:, self.nb_states:-1], -1)
            # valid_lyapunov = np.linalg.solve(np.eye(valid_states.size) - valid_state_transition,
            #                                  augmented_cost[valid_states] + unsafe_state_transition * epsilon)
            # lyapunov[valid_states] = valid_lyapunov
            # lyapunov[self.terminal_states] = 0.
            # return lyapunov

            # Case2 on epsilon (similar to Chow, 2019) ----------------------------------------------------------------
            # terminal_states = np.concatenate((self.terminal_states, self.terminal_states + self.nb_states), 0)
            # valid_states = np.arange(0, 2 * self.nb_states)
            # valid_states = np.delete(valid_states, terminal_states)
            #
            _extra_safety = self.safety_v[(self.time_v > 0.) * (self.safety_v >= self.confidence)]
            try:
                epsilon = (np.min(_extra_safety) - self.confidence) / np.max(self.time_v)
            except ValueError:
                epsilon = 0.
            #
            # state_transition = np.sum(np.expand_dims(self.extended_policy, -1) * self.extended_kernel, 1)
            #
            # augmented_cost = np.ones((2, self.nb_states)) * epsilon * 0.
            # augmented_cost[0, :] += (self.safety < 1.)
            # augmented_cost = augmented_cost.flatten()
            # augmented_cost[terminal_states] = 0.
            #
            # valid_state_transition = state_transition[valid_states, valid_states]
            # valid_lyapunov = np.linalg.solve(np.eye(valid_states.size) - valid_state_transition,
            #                                  augmented_cost[valid_states])
            # lyapunov[valid_states] = valid_lyapunov
            # lyapunov[terminal_states] = 0.
            lyapunov[0:self.nb_states] = (1. - self.safety_v) + epsilon * self.time_v
            return lyapunov
        else:
            # Case2 on epsilon (similar to Chow, 2019)
            min_safety = np.min(self.safety_v[self.safety_v > self.confidence])
            epsilon = np.maximum((- self.confidence + min_safety) / self.episode_length, 0.)

            state_transition = np.sum(np.expand_dims(self.extended_policy, -1) * self.extended_kernel, 1)

            augmented_cost = np.ones((2, self.nb_states)) * epsilon
            augmented_cost[0, :] += (self.safety < 1.)
            augmented_cost = augmented_cost.flatten()
            augmented_cost[self.terminal_states] = 0.

            lyapunov[self.terminal_states] = 0.
            for i in range(self.episode_length):
                lyapunov = augmented_cost + np.sum(state_transition * np.expand_dims(lyapunov, 0), -1)
            return lyapunov

    def _lyapunov_policy_improvement(self):
        _policy = np.zeros((2 * self.nb_states, self.nb_actions))

        for state in range(self.nb_states):
            values = np.sum(self.extended_kernel[state, :, :] * np.expand_dims(self.lyapunov, 0), -1)
            minimum_value = np.min(values)
            optimal_actions = np.where(values == minimum_value)[0]
            _policy[state, optimal_actions] = 1. / optimal_actions.size

        for state in range(self.nb_states, 2 * self.nb_states):
            _policy[state, :] = _policy[state - self.nb_states, :]

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

