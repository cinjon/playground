from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

import numpy as np
from scipy.misc import imresize as resize
import time


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # NOTE: added .all() to work with multi-agent scenarios.
            if type(done) == list:
                done = np.array(done)

            if done.all():
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_render_fps':
            remote.send((env.render_fps))
        elif cmd == 'render':
            remote.send((env.render('rgb_array')))
        elif cmd == 'get_training_ids':
            remote.send((env.get_training_ids()))
        elif cmd == 'get_expert_obs':
            remote.send((env.get_expert_obs()))
        elif cmd == 'get_global_obs':
            remote.send((env.get_global_obs()))
        elif cmd == 'get_game_type':
            remote.send((env.get_game_type()))
        else:
            raise NotImplementedError


class _VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    NOTE: This was taken from OpenAI's baselines package.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        logger.warn('Render not defined for %s'%self)


class _CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to
    use pickle). NOTE: This was taken from OpenAI's baselines package.
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(_VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        env_fns: list of gym environments to run in subprocesses
        """
        self._viewer = None
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker,
                    args=(work_remote, remote, _CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes,
                                                     self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True # Don't hang if the main process crashes.
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        _VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.remotes[0].send(('get_render_fps', None))
        self._render_fps = self.remotes[0].recv()

    def render(self):
        self.remotes[0].send(('render', None))
        frame = self.remotes[0].recv()
        from PIL import Image
        from gym.envs.classic_control import rendering
        human_factor = 32
        board_size = 13
        img = resize(frame, (board_size*human_factor, board_size*human_factor), interp='nearest')
        if self._viewer is None:
            self._viewer = rendering.SimpleImageViewer()
        self._viewer.imshow(img)
        time.sleep(1.0 / self._render_fps)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_expert_obs(self):
        for remote in self.remotes:
            remote.send(('get_expert_obs', None))
        return [remote.recv() for remote in self.remotes]

    def get_global_obs(self):
        for remote in self.remotes:
            remote.send(('get_global_obs', None))
        return [remote.recv() for remote in self.remotes]

    def get_training_ids(self):
        self.remotes[0].send(('get_training_ids', None))
        return self.remotes[0].recv()

    def get_game_type(self):
        self.remotes[0].send(('get_game_type', None))
        return self.remotes[0].recv()
