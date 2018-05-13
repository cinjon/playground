# Getting Started

# Pre-requisites

* [Python 3.6.0](https://www.python.org/downloads/release/python-360/)+ (including `pip`)
* [Docker](https://www.docker.com/) (only needed for `DockerAgent`)
* [virtualenv](https://virtualenv.pypa.io/en/stable/) (optional, for isolated Python environment)

# Installation

* **OPTIONAL**: Setup an isolated virtual Python environment by running the following commands
```
$ virtualenv ~/venv
```
This environment needs to be activated for usage. Any package installations will now persist
in this virtual environment folder only.
```
source ~/venv/bin/activate
```

* Clone the repository
```
$ git clone https://github.com/MultiAgentLearning/playground ~/playground
```

* Install the `pommerman` package. This needs to be done every time the code is updated to get the
latest modules
```
$ cd ~/playground
$ pip install -U .
```

# Examples

## A Simple Example

The [simple_ffa_run.py](../examples/simple_ffa_run.py) runs a sample Free-For-All game with two
[SimpleAgent](../pommerman/agents/simple_agent.py)s and two [RandomAgent](../pommerman/agents/random_agent.py)s
on the board.

## Using A Docker Agent

The above example can be extended to use [DockerAgent](../pommerman/agents/docker_agent.py) instead of a
[RandomAgent](../pommerman/agents/random_agent.py). [examples/docker-agent](../examples/docker-agent) contains
the code to wrap a [SimpleAgent](../pommerman/agents/simple_agent.py) inside Docker.


* We will build a docker image with the name "pommerman/simple-agent" using the `Dockerfile` provided.
```
$ cd ~/playground
$ docker build -t pommerman/simple-agent -f examples/docker-agent/Dockerfile .
```

* The agent list seen in the previous example can now be updated. Note that a `port` argument (of an unoccupied port) is
needed to expose the HTTP server.
```python
agent_list = [
    agents.SimpleAgent(),
    agents.RandomAgent(),
    agents.SimpleAgent(),
    agents.DockerAgent("pommerman/simple-agent", port=12345)
]
```

## Playing an interactive game

You can also play the game! See below for an example where one [PlayerAgent](../pommerman/agents/player_agent.py)
controls with the `arrow` keys and the other with the `wasd` keys.


```python
agent_list = [
    agents.SimpleAgent(),
    agents.PlayerAgent(agent_control="arrows"), # arrows to move, space to lay bomb
    agents.SimpleAgent(),
    agents.PlayerAgent(agent_control="wasd"), # W,A,S,D to move, E to lay bomb
]
```
