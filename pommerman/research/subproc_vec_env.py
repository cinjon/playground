from multiprocessing import Process, Pipe
import numpy as np
from scipy.misc import imresize as resize
import time

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
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
        elif cmd == 'render':
            remote.send((env.render('rgb_array')))
        else:
            raise NotImplementedError


class SubprocVecEnvRender(SubprocVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self._viewer = None
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)))
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
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def render(self):
        self.remotes[0].send(('render', None))
        frame = self.remotes[0].recv()
        from PIL import Image
        from gym.envs.classic_control import rendering
        # Fails either here at resize
        human_factor = 32
        board_size = 13
        img = resize(frame, (board_size*human_factor, board_size*human_factor), interp='nearest')
        if self._viewer is None:
            self._viewer = rendering.SimpleImageViewer()
        self._viewer.imshow(img)
        time.sleep(1.0 / 10)

    def reset(self):
        if self._viewer is not None:
            self._viewer.close()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
