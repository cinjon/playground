from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
import os
import time

import numpy as np
from scipy.misc import imresize as resize

import pommerman


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # NOTE: handles both multi-agent and single-aget
            if type(done) == list:
                done = np.array(done)
                if done.all():
                    ob = env.reset()
            elif type(done) == np.ndarray:
                if done.all():
                    ob = env.reset()
            elif done:
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
        elif cmd == 'get_expert_actions':
            action = env.get_expert_actions(data)
            remote.send((action))
        elif cmd == 'observation':
            obs = env.observation(data)
            remote.send((obs))
        elif cmd == 'get_states_actions_json':
            remote.send(env.get_states_actions_json(data))
        elif cmd == 'reset_state_file':
            remote.send(env.reset_state_file(data))
        elif cmd == 'get_all_state_files':
            remote.send(env.get_all_state_files())
        elif cmd == 'get_init_states_json':
            remote.send(env.get_init_states_json(data))
        elif cmd == 'get_actions':
            remote.send((env.get_actions()))
        elif cmd == 'change_game_state_distribution':
            remote.send((env.change_game_state_distribution()))
        elif cmd == 'get_global_obs':
            remote.send((env.get_global_obs()))
        elif cmd == 'get_non_training_obs':
            remote.send((env.get_non_training_obs()))
        elif cmd == 'get_dead_agents':
            remote.send((env.get_dead_agents()))
        elif cmd == 'get_game_type':
            remote.send((env.get_game_type()))
        elif cmd == 'record_json':
            remote.send((env.record_json(data)))
        elif cmd == 'record_actions_json':
            d1, d2 = data
            remote.send((env.record_actions_json(d1, d2)))
        elif cmd == 'set_bomb_penalty_lambda':
            remote.send((env.set_bomb_penalty_lambda(data)))
        elif cmd == 'set_uniform_v':
            remote.send((env.set_uniform_v(data)))
        elif cmd == 'enable_selfbombing':
            remote.send((env.enable_selfbombing()))
        elif cmd == 'get_json_info':
            remote.send((env.get_json_info()))
        elif cmd == 'set_json_info':
            remote.send((env.set_json_info(data)))
        elif cmd == 'set_florensa_starts':
            remote.send((env.set_florensa_starts(data)))
        elif cmd == 'get_florensa_start':
            remote.send((env.get_florensa_start()))
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

    def render(self, record_pngs_dir=None, game_step_counts=None, num_env=0,
               game_type=None):
        if game_type == pommerman.constants.GameType.Grid:
            self.remotes[num_env].send(('render', None))
            self.remotes[num_env].recv()
        else:
            self.remotes[num_env].send(('render', None))
            frame = self.remotes[num_env].recv()
            from PIL import Image
            from gym.envs.classic_control import rendering
            human_factor = 32
            board_size = 13
            new_size = board_size * human_factor
            img = resize(frame, (new_size, new_size), interp='nearest')
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)
            if record_pngs_dir and game_step_counts is not None:
                step = game_step_counts[num_env]
                if not os.path.exists(record_pngs_dir):
                    os.makedirs(record_pngs_dir)
                im = Image.fromarray(img)
                im.save(os.path.join(record_pngs_dir, '%d.png' % step))
            time.sleep(1. / self._render_fps)

    def reset(self, acting_agent_ids=None):
        for remote in self.remotes:
            remote.send(('reset', acting_agent_ids))
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

    def get_expert_actions(self, observations, expert):
        for remote, obs in zip(self.remotes, observations):
            remote.send(('get_expert_actions', (obs, expert)))
        self.waiting = True

        actions = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return np.stack(actions)

    def get_states_actions_json(self, directory, grid, use_second_place):
        for remote in self.remotes:
            remote.send(('get_states_actions_json', (directory, grid, use_second_place)))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states, actions, files, experts = zip(*results)
        return np.stack(states), np.stack(actions), np.stack(files), np.stack(experts)

    def reset_state_file(self, directory):
        for remote in self.remotes:
            remote.send(('reset_state_file', directory))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def set_json_info(self, directory=None):
        for remote in self.remotes:
            remote.send(('set_json_info', directory))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states, actions, files = zip(*results)
        return np.stack(states), np.stack(actions), np.stack(files)

    def set_florensa_starts(self, starts):
        for remote in self.remotes:
            remote.send(('set_florensa_starts', starts))
        return [remote.recv() for remote in self.remotes]

    def get_florensa_start(self):
        for remote in self.remotes:
            remote.send(('get_florensa_start', None))
        return [remote.recv() for remote in self.remotes]

    def get_all_state_files(self):
        for remote in self.remotes:
            remote.send(('get_all_state_files', None))
        return [remote.recv() for remote in self.remotes]

    def get_init_states_json(self, directory):
        for remote in self.remotes:
            remote.send(('get_init_states_json', directory))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def observation(self, obs):
        for remote in self.remotes:
            remote.send(('observation', obs))
        return [remote.recv() for remote in self.remotes]

    def get_actions(self):
        for remote in self.remotes:
            remote.send(('get_actions', None))
        return [remote.recv() for remote in self.remotes]

    def change_game_state_distribution(self):
        for remote in self.remotes:
            remote.send(('change_game_state_distribution', None))
        return [remote.recv() for remote in self.remotes]

    def get_global_obs(self):
        for remote in self.remotes:
            remote.send(('get_global_obs', None))
        return [remote.recv() for remote in self.remotes]

    def get_training_ids(self):
        for remote in self.remotes:
            remote.send(('get_training_ids', None))
        return [remote.recv() for remote in self.remotes]

    def get_json_info(self):
        for remote in self.remotes:
            remote.send(('get_json_info', None))
        return [remote.recv() for remote in self.remotes]

    def set_json_info(self, game_states=None):
        if game_states is None:
            game_states = [None] * len(self.remotes)
        for remote, game_state in zip(self.remotes, game_states):
            remote.send(('set_json_info', game_state))
        return [remote.recv() for remote in self.remotes]

    def get_game_type(self):
        self.remotes[0].send(('get_game_type', None))
        return self.remotes[0].recv()

    def record_json(self, directories, num_env=None):
        if num_env:
            self.remotes[num_env].send(('record_json', directories[num_env]))
            return self.remotes[num_env].recv()
        else:
            for remote, directory in zip(self.remotes, directories):
                remote.send(('record_json', directory))
            return [remote.recv() for remote in self.remotes]

    def record_actions_json(self, directories, actions, num_env=None):
        if num_env:
            self.remotes[num_env].send(('record_actions_json', directories[num_env], actions[num_env]))
            return self.remotes[num_env].recv()
        else:
            for remote, directory, action in zip(self.remotes, directories, actions):
                remote.send(('record_actions_json', (directory, action)))
            return [remote.recv() for remote in self.remotes]

    def get_non_training_obs(self):
        for remote in self.remotes:
            remote.send(('get_non_training_obs', None))
        return [remote.recv() for remote in self.remotes]

    def get_dead_agents(self):
        for remote in self.remotes:
            remote.send(('get_dead_agents', None))
        return [remote.recv() for remote in self.remotes]

    def set_bomb_penalty_lambda(self, l):
        for remote in self.remotes:
            remote.send(('set_bomb_penalty_lambda', l))
        return [remote.recv() for remote in self.remotes]

    def enable_selfbombing(self):
        for remote in self.remotes:
            remote.send(('enable_selfbombing', None))
        return [remote.recv() for remote in self.remotes]

    def set_uniform_v(self, v):
        for remote in self.remotes:
            remote.send(('set_uniform_v', v))
        return [remote.recv() for remote in self.remotes]
