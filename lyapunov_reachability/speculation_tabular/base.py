import os.path
from timeit import default_timer as timer
import pickle
import numpy as np

#from scipy.optimize import linprog
import cplex
from cplex.exceptions import CplexSolverError


class QBase(object):

    stats = ['runtime', 'average_safety', 'average_steps', 'approximate_reachability']

    def __init__(self, env, confidence, nb_states, nb_actions, initial_policy, terminal_states, seed=None,
                 strict_done=True, safe_init=False, baseline_dir=None, baseline_step=None, save_dir='../../q-base'):
        self.env = env
        self.confidence = confidence    # safety confidence,
        self.nb_states = nb_states      # S
        self.nb_actions = nb_actions    # A
        self.terminal_states = terminal_states
        self.strict_done = strict_done
        self.safe_init = None if safe_init is False else 0  # Adaptive initial state, or None
        self.steps = 0

        self.policy = initial_policy    # policy, S x A (initially safe)
        self.reachability_q = np.ones((self.nb_states, self.nb_actions))
        self.reachability_q[terminal_states] = 0.
        self.updates_q = np.zeros((self.nb_states, self.nb_actions))

        self.seed = seed
        if seed is not None:
            env.seed(seed)
            np.random.seed(seed)

        if baseline_dir is not None:
            baseline = os.path.join(os.path.abspath(baseline_dir), '{}.npz'.format(baseline_step))
            if os.path.exists(baseline):
                self.load_baseline(baseline)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = os.path.abspath(save_dir)
        self.save_setup()

    def _log_auxiliary(self, **kwargs):
        raise NotImplementedError

    def _iteration(self, t, obs, action, next_obs, safety, done, **kwargs):
        raise NotImplementedError

    def step(self, state, **kwargs):
        try:
            return np.random.choice(self.nb_actions, 1, p=self.policy[state, :])[0]
        except ValueError:
            print("Error: stochastic policy is not feasible. Policy=\t" + str(self.policy[state, :]))

    def run(self, steps, episode_length, improve_interval=1, log_interval=None, save_interval=None, **kwargs):
        # Print setup before begin.
        print(kwargs)

        # Setup for logging
        if log_interval is None:
            log_interval = episode_length
        if save_interval is None:
            save_interval = episode_length

        result = dict()
        for stat in self.__class__.stats:
            result[stat] = list()
        result['runtime'].append(0.0)

        episode_step = 0
        episode_safety = 1.
        record_step = list()
        record_safety = list()
        runtime = 0.
        
        # Implement class-specific initialization here.
        self.extra_setup(steps, episode_length, improve_interval, log_interval, save_interval, **kwargs)

        # First reset
        state = self.initialize_env(improve_interval)
        self.save_model(os.path.join(self.save_dir, '%d' % self.steps))
        
        # Epsilon-decay
        eps = kwargs.get('epsilon', 0.)
        eps_decay = kwargs.get('epsilon_decay', 0.)
        eps_last = kwargs.get('epsilon_last', 0.)

        for t in range(1, steps+1):

            # [1] Fetch a sample state transition tuple.
            action = self.step(state, **kwargs)
            _, _, done, info = self.env.step(action)
            next_state = info['state']
            safety = info['safety']
            #reached = (1. - safety) * extra_state

            # [2] Record episodic information.
            episode_step += 1
            episode_safety *= safety

            # [3] Update networks
            episode_done = done + (episode_step == episode_length) > 0.
            done = (done + (1. - self.strict_done) * (episode_step == episode_length) > 0.) * 1.

            tic = timer()
            self._iteration(t, state, action, next_state, safety, done, improve_interval=improve_interval, **kwargs)
            runtime += timer() - tic

            # [4] Reset the environment if the agent visits the target set for the first time.
            if safety < 1. or episode_done == 1.:
                record_step[0:0] = [episode_step]
                episode_step = 0
                record_step = record_step[:100]
                record_safety[0:0] = [episode_safety]
                episode_safety = 1.
                record_safety = record_safety[:100]

                state = self.initialize_env(improve_interval, np.mean(record_safety))
            else:
                state = next_state
                # extra_state = extra_state * (safety > 0.)

            # [5] Epsilon-decay
            kwargs['epsilon'] = np.maximum(eps * (1. - t * eps_decay), eps_last)
                
            self.steps += 1

            if t % log_interval == 0:
                average_safety = np.mean(record_safety)
                average_step = np.mean(record_step)

                result['runtime'].append(result['runtime'][-1] + runtime)
                result['average_safety'].append(average_safety)
                result['average_steps'].append(average_step)
                self._log_auxiliary(**kwargs)

                print('log\t:: steps={}, episode_safety={}, episode_runtime={}'.format(
                    self.steps, average_safety, average_step))
            if t % save_interval == 0:
                self.save_model(os.path.join(self.save_dir, '%d' % self.steps))
                print('chart\t:: safe_set_size={}'.format(
                    np.mean(np.min(self.reachability_q, -1) <= 1. - self.confidence)))
        print('done.')
        del result['runtime'][0]
        self.save_model(os.path.join(self.save_dir, '%d' % self.steps))
        return result

    def initialize_env(self, improve_interval=1, average_safety=1.):
        _ = self.env.reset()
        if self.safe_init is None:
            state = self.env.quantize(self.env.state)
        else:
            if self.steps % improve_interval == 0 or average_safety < self.confidence:
                reach = np.min(self.reachability_q, axis=-1)
                safe_mask = (reach <= 1. - self.confidence) * (reach > 0.)
                try:
                    state = self.safe_init = np.random.choice(np.where(reach == np.max(reach[safe_mask]))[0])
                except ValueError:
                    state = self.env.quantize(self.env.state)
                # state = self.safe_init = np.random.choice(np.where(safe_mask)[0])
            else:
                state = self.safe_init
            self.env.set_state(state)
        return state

    def load_baseline(self, baseline):
        data = np.load(baseline)
        self.reachability_q = data['reachability_q']
        self.updates_q = data['updates_q']
        if 'policy' in data.keys():
            self.policy = data['policy']
        else:
            safest_reachability = np.min(self.reachability_q, axis=-1, keepdims=True)
            self.policy[:] = (self.reachability_q - safest_reachability == 0.) * 1.
            self.policy[:] = self.policy / np.sum(self.policy, axis=-1, keepdims=True)
    
    def extra_setup(self, steps, episode_length, improve_interval, log_interval, save_interval, **kwargs):
        return
    
    def save_setup(self):
        info = dict()

        # Common properties
        info['confidence'] = self.confidence
        info['nb_states'] = self.nb_states
        info['nb_actions'] = self.nb_actions
        info['initial_policy'] = self.policy
        info['terminal_states'] = self.terminal_states
        info['seed'] = self.seed
        info['strict_done'] = self.strict_done
        info['safe_init'] = True if self.safe_init is not None else False

        with open(os.path.join(self.save_dir, "params.pkl"), 'wb') as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

    def save_model(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, load_dir, steps, env=None, ** new_kwargs):
        data = None

        try:
            with open(os.path.join(load_dir, 'params.pkl'), 'rb') as f:
                data = pickle.load(f)
            if env is None:
                with open(os.path.join(load_dir, 'env.pkl'), 'rb') as f:
                    env = pickle.load(f)
        except FileNotFoundError or UnicodeDecodeError:
            print('Unable to restore parameters/environment.')

        confidence = data.pop('confidence')
        nb_states, nb_actions = data.pop('nb_states'), data.pop('nb_actions')
        initial_policy, terminal_states = data.pop('initial_policy'), data.pop('terminal_states')

        model = cls(env, confidence, nb_states, nb_actions, initial_policy, terminal_states, **data, save_dir=load_dir)
        model.__dict__.update({'steps': steps})
        model.__dict__.update(**new_kwargs)
        model.load_model(load_dir)
        return model

    def load_model(self, load_dir):
        raise NotImplementedError
