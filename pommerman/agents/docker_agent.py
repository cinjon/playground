import json
import time
import os
import threading
import requests
import docker

from . import BaseAgent
from .. import utility
from .. import characters


class DockerAgent(BaseAgent):
    """The Docker Agent that Connects to a Docker container where the character runs."""

    def __init__(self,
                 docker_image,
                 port,
                 server='http://localhost',
                 character=characters.Bomber,
                 docker_client=None,
                 env_vars=None):
        super(DockerAgent, self).__init__(character)

        self._docker_image = docker_image
        self._docker_client = docker_client or docker.from_env()
        self._server = server
        self._port = port
        self._container = None
        self._env_vars = env_vars or {}

        container_thread = threading.Thread(
            target=self._run_container, daemon=True)
        container_thread.start()
        self._wait_for_docker(self._server, self._port, 32)

    def _run_container(self):
        print("Starting container...")

        # Any environment variables that start with DOCKER_AGENT are passed to the container
        env_vars = self._env_vars
        for key, value in os.environ.items():
            if not key.startswith("DOCKER_AGENT_"):
                continue
            env_key = key.replace("DOCKER_AGENT_", "")
            env_vars[env_key] = value

        self._container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            auto_remove=True,
            ports={10080: self._port},
            environment=env_vars)

    @staticmethod
    def _wait_for_docker(server, port, timeout=None):
        """Wait for network service to appear.

        Args:
          port: Integer port.
          timeout: Seconds to wait. 0 waits forever.
        """
        backoff = .25
        max_backoff = min(timeout, 16)

        if timeout:
            # time module is needed to calc timeout shared between two exceptions
            end = time.time() + timeout

        while True:
            try:
                now = time.time()
                if timeout and end < now:
                    return False

                request_url = '%s:%s/ping' % (server, port
                                             )  # 'http://localhost', 83
                req = requests.get(request_url)
                return True
            except requests.exceptions.ConnectionError as e:
                print("ConnectionError: ", e)
                backoff = min(max_backoff, backoff * 2)
                time.sleep(backoff)
            except requests.exceptions.HTTPError as e:
                print("HTTPError: ", e)
                backoff = min(max_backoff, backoff * 2)
                time.sleep(backoff)
            except docker.errors.APIError as e:
                print("This is a Docker error. Please fix: ", e)
                raise

    def act(self, obs, action_space):
        obs_serialized = json.dumps(obs, cls=utility.PommermanJSONEncoder)
        request_url = "http://localhost:{}/action".format(self._port)
        try:
            req = requests.post(
                request_url,
                timeout=0.25,
                json={
                    "obs":
                    obs_serialized,
                    "action_space":
                    json.dumps(action_space, cls=utility.PommermanJSONEncoder)
                })
            action = req.json()['action']
        except requests.exceptions.Timeout as e:
            print('Timeout!')
            # TODO: Fix this. It's ugly.
            action = [0] * len(action_space.shape)
            if len(action) == 1:
                action = action[0]
        return action

    def shutdown(self):
        print("Stopping container..")
        if self._container:
            try:
                return self._container.remove(force=True)
            except docker.errors.NotFound as e:
                return True
