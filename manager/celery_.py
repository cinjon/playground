# TODO: Add Documentation.

from collections import defaultdict
import os
import random
import shutil
import sys
import time
import traceback

import docker
from celery import Celery
import requests
import subprocess

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pommerman

client = docker.from_env()
client.login(os.getenv("PLAYGROUND_DOCKER_LOGIN"),
             os.getenv("PLAYGROUND_DOCKER_PASSWORD"))

game_directory = os.path.expanduser('~/battles')
json_directory = os.path.join(game_directory, 'json')
png_directory = os.path.join(game_directory, 'png')
if not os.path.exists(json_directory):
    os.makedirs(json_directory)
if not os.path.exists(png_directory):
    os.makedirs(png_directory)

celery = Celery("playground")
celery.conf.broker_url = 'redis://localhost:6379/0'


@celery.task
def run_test(docker_build_path, github_repo, private_key, name, agent_id, user,
             config):
    user = user.lower()
    name = name.lower()

    record_pngs_dir = os.path.join(png_directory, 'test-%s-%s' % (user, name))
    record_json_dir = os.path.join(json_directory, 'test-%s-%s' % (user, name))
    
    test = Test(name, user, config, private_key, github_repo, agent_id,
                docker_build_path, record_pngs_dir, record_json_dir)
    return test.run()


server_ready_notifs = defaultdict(set)
@celery.task
def add_server_ready_notif(notif):
    """Add this notification to the queue. Run battle if it completes a set."""
    battle_info = notif['battle_info']
    server_ready_notifs[battle_info].add(notif)
    battle_notifs = server_ready_notifs[battle_info]
    if len(battle_notifs) > 4:
        logging.warn("We have too many battle notifs. How did this happen?")
        logging.warn(", ".join(
            ["{}.{}.{}".format(notif['agent_id'],
                               notif['aid'],
                               notif['docker_image'])
             for notif in battle_notifs]))
    elif len(battle_notifs) == 4:
        # Everyone is ready. Let's do this.
        battle_notifs = sorted(list(battle_notifs),
                               key=lambda k: k['agent_id'])
        agents = {k:v for k, v in battle_notifs \
                  if k in ['aid', 'docker_image', 'agent_id']}
        run_battle(battle_info, agents)
        del server_ready_notifs[battle_info]


# TODO: Ensure there is >1 worker to accept requests while battles occur.
@celery.task
def run_battle(battle_info, agents):
    """Agents is a list of dicts specifying docker image, aid, and agent_id."""
    suffix = battle_info
    record_pngs_dir = os.path.join(png_directory, suffix)
    record_json_dir = os.path.join(json_directory, suffix)
    battle = Battle(agents, record_pngs_dir, record_json_dir)
    info = battle.run()
    logging.warn("Info from the run_battle on %s:" % battle_info)
    logging.warn(info)

    # Tell the server about the result.
    request_json = {
        'result': info['result'].value,
        'winners': info.get('winners', []),
        'battle_info': battle_info,
        'access': os.getenv("PLAYGROUND_GAME_MANAGER_ACCESS")
    }
    request_url = os.getenv("PLAYGROUND_SERVER_URL") + "/admin/report_result"
    requests.post(request_url, json=request_json)


class Battle(object):
    def __init__(self, agents, record_pngs_dir, record_json_dir):
        """A battle object.

        Args:
          agents: A list of dicts with agent_id, aid, and docker_image.
          record_pngs_dir: The string directory path to place recorded pngs.
          record_json_dir: The string directory path to place recorded json.
        """
        self._agents = agents
        self._record_pngs_dir = record_pngs_dir
        self._record_json_dir = record_json_dir

    def run(self):
        # At this point, we can assume that the containers have been pulled.
        # We now start the game, which notifies those containers to start the
        # agents and let us know they're ready.
        agents = ["docker::%s" % agent['docker_image']
                  for agent in self._agents]
        agents = ",".join(agents)
        args = Args(agents, self._config, self._record_pngs_dir,
                    self._record_json_dir, agent_env_vars, None, render,
                    do_sleep=False)
        
        seed = os.getenv("PLAYGROUND_GAME_SEED")
        infos = pommerman.cli.run_battle.run(args, num_times=1, seed=seed)
        return infos[0]

    # TODO: Add logging in case the post fails.
    @staticmethod
    def _fail(aids, error):
        """Sends ping to server that this was a failed battle."""
        request_url = os.getenv('PLAYGROUND_SERVER_URL') + '/fail_battle'
        requests.post(request_url, json={
            "aids": aids,
            "error": error,
            "access": os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        })

    # TODO: Add logging in case the post fails.
    @staticmethod
    def _pass(aids, info):
        """Sends ping that this was a successful test."""
        request_url = os.getenv('PLAYGROUND_SERVER_URL') + '/pass_battle'
        requests.post(request_url, json={
            "aids": aids,
            "info": info,
            "access": os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        })


class Test(object):
    def __init__(self, name, user, config, private_key, github_repo, agent_id,
                 docker_build_path, record_pngs_dir, record_json_dir):
        self._name = name
        self._user = user
        self._config = config
        self._private_key = private_key
        self._github_repo = github_repo
        self._agent_id = agent_id
        self._docker_build_path = docker_build_path
        self._record_pngs_dir = record_pngs_dir
        self._record_json_dir = record_json_dir

    def run(self):
        img = None
        message = None

        for func in [
                lambda: save_ssh_pk(name, private_key),
                lambda: download_repo(name, github_repo),
                lambda: docker_build(docker_build_path, name, user),
                lambda: self._battle(num_times=5, agent_env_vars={},
                                     render=True, seed=1)
        ]:
            # TODO: This won't be necessary if we can fix the image purging.
            is_success, ret = func()
            if type(ret) == docker.models.images.Image:
                img = ret
            else:
                message = ret

            if not is_success:
                return self._fail(agent_id, ret)

        # This has been successful. Hurray.
        # Push the Docker image to the cloud and then clean it up.
        docker_push_and_clean_up(img, name, user)
        return self._pass(agent_id, message)

    # TODO: Add logging in case the post fails.
    @staticmethod
    def _fail(agent_id, error):
        """Sends ping to server that this was a failed test."""
        request_url = os.getenv('PLAYGROUND_SERVER_URL') + '/fail_test'
        requests.post(request_url, json={
            "agent_id": agent_id,
            "error": error,
            "access": os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        })

    # TODO: Add logging in case the post fails.
    @staticmethod
    def _pass(agent_id, win_percent):
        """Sends ping that this was a successful test."""
        request_url = os.getenv('PLAYGROUND_SERVER_URL') + '/pass_test'
        requests.post(request_url, json={
            "agent_id": agent_id,
            "win_percent": win_percent,
            "access": os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        })

    def _battle(self, num_times, agent_env_vars, render, seed=None):
        return battle(num_times, agent_env_vars, render, seed, self._user,
                      self._name, self._config, self._record_pngs_dir,
                      self._record_json_dir)


def battle(num_times, agent_env_vars, render, seed, user, name, config,
           record_pngs_dir, record_json_dir):
    try:
        agents = ["test::agents.SimpleAgent"]*3
        agent_id = random.randint(0, len(agents))
        agent_tag = "docker::multiagentlearning/pommerman-%s-%s" % (user, name)
        agents.insert(agent_id, agent_tag)
        agents = ",".join(agents)
        
        args = Args(agents, config, record_pngs_dir, record_json_dir,
                    agent_env_vars, None, render, do_sleep=True)
        
        infos = []
        infos = pommerman.cli.run_battle.run(args, num_times=num_times,
                                             seed=seed)
        win_count = len([i for i in infos if 'winners' in i \
                         and agent_id in i['winners']])
        win_percent = 1. * win_count / len(infos)
        return True, win_percent, info
    except Exception as e:
        traceback.print_exc()
        return False, "battle: %s" % e

def save_ssh_pk(name, private_key):
    try:
        path = os.path.expanduser('~/.ssh/id_%s' % name)
        if os.path.exists(path):
            os.remove(path)
            
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as f:
            f.write(private_key)
            
        return True, ""
    except Exception as e:
        return False, "save_ssh_pk: %s" % e

def download_repo(name, github_repo):
    try:
        directory = os.path.expanduser('~/Repos/%s' % name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            
        from git import Repo
        from git import Git
            
        git_ssh_identity_file = os.path.expanduser('~/.ssh/id_%s' % name)
        git_ssh_cmd = 'ssh -i %s' % git_ssh_identity_file
        Repo.clone_from(github_repo, directory, env={'GIT_SSH_COMMAND': git_ssh_cmd})
        return True, ""
    except Exception as e:
        return False, "download_repo: %s" % e

def docker_build(docker_build_path, name, user):
    try:
        repo_path = os.path.join(os.path.expanduser('~/Repos'), name)
        img = client.images.build(
            path=repo_path, dockerfile=docker_build_path,
            tag='multiagentlearning/pommerman-%s-%s' % (user, name))
        return True, img
    except Exception as e:
        return False, "docker_build: %s" % e

def docker_push_and_clean_up(img, name, user):
    try:
        client.images.push("multiagentlearning/pommerman-%s-%s" % (user, name))
        # TODO: Fix this so that it purges correctly.
        # ... Consider just using subprocess(docker system prune -a -f)
        client.images.prune("%s-%s" % (user, name), filters={'dangling': False})
        return True
    except Exception as e:
        print("push / clean up: ", e)
        traceback.print_exc()
        return False


class Args(object):
    """Args object to replicate the args in run_battle."""
    def __init__(self, agents, config, record_pngs_dir, record_json_dir,
                 agent_env_vars, game_state_file, render, do_sleep):
        self.config = config
        self.record_pngs_dir = record_pngs_dir
        self.record_json_dir = record_json_dir
        self.agent_env_vars = agent_env_vars
        self.game_state_file = game_state_file
        self.render = render
        self.render_mode = 'human'
        self.agents = agents
        self.do_sleep = do_sleep
