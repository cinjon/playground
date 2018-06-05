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
        self._docker_client = docker_client
        if not self._docker_client:
            self._docker_client = docker.from_env()
            self._docker_client.login(os.getenv("PLAYGROUND_DOCKER_LOGIN"),
                                      os.getenv("PLAYGROUND_DOCKER_PASSWORD"))

        self._acknowledged = False # Becomes True when the container is ready.
        self._server = server
        self._port = port
        self._timeout = 32
        self._container = None
        self._env_vars = env_vars or {}
        # Pass env variables starting with DOCKER_AGENT to the container.
        for key, value in os.environ.items():
            if not key.startswith("DOCKER_AGENT_"):
                continue
            env_key = key.replace("DOCKER_AGENT_", "")
            self._env_vars[env_key] = value
        
        # Start the docker agent if it is on this computer. Otherwise, it's far
        # away and we need to tell that server to start it.
        if 'localhost' in server:
            container_thread = threading.Thread(
                target=self._run_container, daemon=True)
            container_thread.start()
            print("Waiting for docker agent at {}:{}...".format(server, port))
            self._wait_for_docker()
        else:
            request_url = "{}:8000/run_container".format(server)
            request_json = {'docker_image': self._docker_image,
                            'env_vars': self._env_vars,
                            'port': port}
            requests.post(request_url, json=request_json)
            waiting_thread = threading.Thread(
                target=self._wait_for_docker, daemon=True)
            waiting_thread.start()

    def _run_container(self):
        print("Starting container...")
        self._container = self._docker_client.containers.run(
            self._docker_image,
            detach=True,
            auto_remove=True,
            ports={10080: self._port},
            environment=self._env_vars)
        for line in self._container.logs(stream=True):
            print(line.decode("utf-8").strip())

    def _wait_for_docker(self):
        """Wait for network service to appear. A timeout of 0 waits forever."""
        timeout = self._timeout
        backoff = .25
        max_backoff = min(timeout, 16)

        if timeout:
            # time module is needed to calc timeout shared between two exceptions
            end = time.time() + timeout

        while True:
            try:
                now = time.time()
                if timeout and end < now:
                    print("Timed out - %s:%s" % (self._server, self._port))
                    raise

                request_url = '%s:%s/ping' % (self._server, self._port)
                req = requests.get(request_url)
                self._acknowledged = True
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
                timeout=0.15,
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
