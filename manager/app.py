import logging
import os

import celery_ as celery
import docker
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

import pommerman


# Game Manager and Servers
@a.app.before_request
def check_for_access():
    incoming = request.get_json()
    if request.path != '/ping':
        incoming = request.get_json()
        access = incoming.get('access')
        game_manager_access = os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        if not access or access != game_manager_access:
            return jsonify(received=False, error="Access Denied"), 400
    

# To Game Manager and Servers, from Web.
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify(success=True)


# To Game Manager, from Web.
@app.route('/test', methods=['POST'])
def test():
    """Build and run this docker agent locally."""
    try:
        incoming = request.get_json()
        docker_build_path = incoming["docker_build_path"]
        github_repo = incoming["github_repo"]
        private_key = incoming["private_key"]
        name = incoming["name"]
        agent_id = incoming["agent_id"]
        user = incoming["user"]
        config = incoming["config"]
        celery.run_test(docker_build_path, github_repo, private_key, name,
                        agent_id, user, config)
        return jsonify(received=True, error="")
    except Exception as e:
        return jsonify(received=False, error=e)


# To Game Manager, from Web.
@app.route('/request_battle', methods=['POST'])
def request_battle():
    """Process a request to do a battle among four agents.

    This is on the game manager server. The request includes the docker images
    for each agent, along with their agent id (aid) and the config.

    The execution order is:
    1. Tell each of the four servers to pull their given agent's container.
    2. They'll then send us back container_is_ready notifications.
    3. When we receive all of those notifications, we'll fire run_battle here.
    4. The run_battle script will then manage speaking to each of the servers.
    5. After the game is over, a result will be sent back to the web server.
    """
    try:
        incoming = request.get_json()
        agents = [{
            'docker_image': incoming.get('docker_image_agent_%d' % agent_id),
            'aid': incoming.get('aid_%d' % agent_id)
            'agent_id': agent_id
            for agent_id in range(4)
        }]
        battle_info = incoming['config']
        battle_info += '-%d-%d-%d-%d' % [agent['aid'] for agent in agents]
        success, message = notify_containers(agents, battle_info):
        if success:
            return jsonify(success=True, error="")
        else:
            return jsonify(success=False, error=message)
    except Exception as e:
        return jsonify(success=False, error=e)


def notify_containers(agents, battle_info):
    """Tell the servers to pull and start the given containers."""
    for agent in enumerate(agents):
        if pommerman.helpers.use_game_servers:
            server = pommerman.helpers.game_servers[agent['agent_id']]
        else:
            server = "http://localhost"
            
        port = "8000"
        url = ':'.join([server, port])
        request_url = url + "/start_container"
        # This includes the aid, the docker_image, and the agent_id
        request_json = agent.copy()
        request_json["access"] = os.getenv('PLAYGROUND_GAME_MANAGER_ACCESS')
        request_json["battle_info"] = battle_info
        request_json["url"] = url
        requests.post(request_url, json=request_json)


# From Game Manager, To Game Servers.
@app.route('/start_container', methods=['POST'])
def start_container():
    """Server endpoint for requests to pull and then start containers."""
    game_manager_url = os.getenv("PLAYGROUND_GAME_MANAGER_SERVER") + ":8000"

    try:
        incoming = request.get_json()

        # The battle's unique identifier, my url (server:port), and the docker
        # image. I'm going to send these back when I report we're good to go.
        agent_id = incoming["agent_id"]
        battle_info = incoming["battle_info"]
        docker_image = incoming["docker_image"]
        url = incoming["url"]
        
        client = docker.from_env()
        client.login(os.getenv("PLAYGROUND_DOCKER_LOGIN"),
                     os.getenv("PLAYGROUND_DOCKER_PASSWORD"))
        logging.warn("Pulling the image %s..." % docker_image)
        img = client.images.pull(docker_image, tag="latest")

        if img:
            request_url = game_manager_url + "/container_is_ready"
            request_json = {'aid': incoming['aid'], 'battle_info': battle_info,
                            'docker_image': docker_image, 'agent_id': agent_id}
            requests.post(request_url, json=request_json)
        else:
            pass
    except Exception as e:
        print("Failed to pull container: %s" % )


# From Game Servers, To Game Manager.
@app.route('/container_is_ready', methods=['POST'])
def container_is_ready():
    """A ready container alert from a server came in. Feed this to celery."""
    try:
        incoming = request.get_json()
        celery_.add_server_ready_notif(incoming)
        return jsonify(success=True, error="")
    except Exception as e:
        return jsonify(success=False, error=e)


# From Game Manager, To Game Servers.
@app.route('/run_container', methods=['POST'])
def run_container():
    incoming = request.get_json()
    docker_image = incoming['docker_image']
    env_vars = incoming['env_vars']
    port = incoming['port']
    
    client = docker.from_env()
    client.login(os.getenv("PLAYGROUND_DOCKER_LOGIN"),
                 os.getenv("PLAYGROUND_DOCKER_PASSWORD"))
    container = client.containers.run(
        docker_image,
        detach=True,
        auto_remove=True,
        ports={10080: port},
        environment=env_vars
    )
    for line in container.logs(stream=True):
        print(line.decode("utf-8").strip())


if __name__ == '__main__':
    app.run()
