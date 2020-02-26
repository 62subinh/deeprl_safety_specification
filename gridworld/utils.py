import matplotlib.pyplot as plt
from cv2 import resize
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

"""
:param env: gym.Env that satisfies all the following conditions.
    It provides:
        a)state as an index(info['state_index']), b)constraint cost(info['constraint']),
        c)the number of states & actions, respectively(env.nb_states & env.nb_acts),
        d)index of the initial state(env.initial_state).
    Its properties are:
        a)takes a single integer as 'action' for each step,
        b)always starts at the fixed state,
"""


def play(env, act, gif_name, gamma=1., feed_obs=False, episode_length=None, frame_size=(180, 180)):
    """
    Given an agent, play the environment once, then show the observation while saving it as an image.
    Reference: http://nbviewer.jupyter.org/github/patrickmineault/xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
    :param env:             Instance of gym.env
    :param act:             Agent targeting env, should have *.__call__(state_index)
    :param gamma:           Discount factor.
    :param feed_obs:        If True, feed observation to act.
    :param episode_length:  Maximum length of an episode.
    :param gif_name:        Name of the output gif, should end with '.gif'
    :param frame_size:      Size of the output gif, should be a tuple (int, int)
    """
    if episode_length is None:
        if hasattr(act, 'episode_length'):
            episode_length = act.episode_length
        elif hasattr(env, 'episode_length'):
            episode_length = env.episode_length
        else:
            episode_length = 2147483647

    obs, done = env.reset(), False
    state = None
    if not feed_obs:
        state = env.cheat()
    episode_rew = 0
    episode_safety = 0
    frames = []
    t = 0
    while not done:
        if t > episode_length:
            break
        # Create image
        frame = env.render(mode='rgb_array')
        frames.append(resize(frame, dsize=frame_size,))

        # Do step
        input_ = obs if feed_obs else state
        _, rew, done, info = env.step(act.step(input_))
        if not feed_obs:
            state = info['state']
        episode_safety = episode_safety * info['safety']
        episode_rew = gamma * episode_rew + rew
        t += 1
    print("Total reward: %.6f" % episode_rew)
    print("Total safety: %.1f" % episode_safety)
    env.close()

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(gif_name, dpi=80, writer='imagemagick')
    display(display_animation(anim, default_mode='loop'))
    return frames


def test(env, act, nb_simulations, gamma=1., episode_length=None):
    """
    Test the given agent with the given environment, and show the results.
    :param env:             Instance of gym.env
    :param act:             Agent targeting env, should have *.__call__(state_index)
    :param nb_simulations:  Number of trials to do.
    :param gamma:           Discount factor.
    :param episode_length:  Maximum length of an episode.
    :return: Percentage of safe / success runs, average return of successful runs / failed runs.
    """
    if episode_length is None:
        if hasattr(act, 'episode_length'):
            episode_length = act.episode_length
        else:
            episode_length = 2147483647

    count = 0
    safe_run, success_run, success_reward, failure_reward = 0, 0, 0., 0.
    while count < nb_simulations:
        env.reset()
        state_index, done = act.initial_state, False
        episode_rew = 0.
        episode_safety = 1.
        t = 0
        while t <= episode_length and not done:
            _, rew, done, info = env.step(act.step(state_index))
            state_index = info['state']
            episode_safety *= info['safety']
            episode_rew = gamma * episode_rew + rew
            t += 1
        count += 1
        if episode_safety > 0.:
            safe_run += 1

        if done:
            success_run += 1
            success_reward += episode_rew
        else:
            failure_reward += episode_rew

    env.close()
    avg_success_return = None if success_run == 0 else 1. * success_reward / success_run
    avg_failure_return = None if success_run == nb_simulations else 1. * failure_reward / (nb_simulations - success_run)
    return 1. * safe_run / nb_simulations, 1. * success_run / nb_simulations, avg_success_return, avg_failure_return


def visualize(frames, gif_name, lb=None, ub=None):
    """
    Create a gif file from the given frames, and display the result.
    :param frames:      A list of numpy 2D arrays having same size; a single element is a value function.
    :param gif_name:    Name of the outcome.
    :param lb:          Lower bound of value function.
    :param ub:          Upper bound of value function.
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0], cmap='plasma')
    if lb is not None and ub is not None:
        plt.clim(lb, ub)
    plt.colorbar()
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(gif_name, dpi=80, writer='imagemagick')
    display(display_animation(anim, default_mode='loop'))
