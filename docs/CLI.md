# Command-Line Interface

Pommerman comes with a CLI tool that allows you to quickly launch a game. This can be used to test how well a trained agent plays againts other agents.

Call this with a config, a game, and a list of agents. The script will start separate threads to operate the agents and then report back the result.

An example with all four test agents running ffa:

```bash
pom_battle --agents=test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=PommeFFACompetition-v0
```

An example with one player, two random agents, and one test agent:

```bash
pom_battle --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --config=PommeFFACompetition-v0
```

An example with a docker agent:

```bash
pom_battle --agents=player::arrows,docker::pommerman/test-agent,random::null,random::null --config=PommeFFACompetition-v0
```

## Configurations and Options

To get a list of active options you can run `pom_battle --help`. Below is a list of options the CLI tool supports with detials on how it effects the run time.

- `--game` allows you to change the game your agent plays. The default is `pommerman`. Currently only supports `pommerman`

- `--config` changes the type of game the agents will play. The default is `PommeFFACompetition-v0`. Other options are `PommeFFACompetition-v0`, `PommeFFAFast-v0`, `PommeFFA-v1`, `PommeRadio-v2`, `PommeTeam-v0`, and `PommeTeamFast-v0`.

- `--agents` defines the agents participating in the game. The default is 4 simple agents. To changes the agents in the game use a comma delineated list of agent.

- `--agent_env_vars` sends enviroment variables to to Docker agents and only Docker agents. The default is "". An example is '0:foo=bar:baz=lar,3:foo=lam', which would send two arguments to Docker Agent 0 and one to Docker Agent 3.

- `--record_pngs_dir` defines the directory to record PNGs of the game board for each step. The default is `None`. If the directory doesn't exist, it will be created. The PNGs are saved with the format `%m-%d-%y_%-H-%M-%S_(STEP).png` (`04-17-18_15-54-39_3.png`).

- `--record_json_dir` defines the directory to record the JSON representations of the game. The default is `None`. If the directory doesn't exist, it will be created.

- `--render` allows you to turn of rendering of the game. The default is `False`.

- `--render_mode` changes the render mode of the game. The default is `human`. Available options are `human`, `rgb_pixel`, and `rgb_array`.

- `--game_state_file` changes the initial state of the game. The file is expected to be in JSON format.  The format of the file is defined below.
    
    - agents: list of agents serialized (agent_id, is_alive, position, ammo, blast_strength, can_kick)
    - board: board matrix topology (board_size^2)
    - board_size: board size
    - bombs: list of bombs serialized (position, bomber_id, life, blast_strength, moving_direction)
    - flames: list of flames serialized (position, life)
    - items: list of item by position
    - step_count: step count


# Training an agent using Tensorforce

Pommerman comes with a trainable agent out of the box. The agent uses a Proximal Policy Optimization (PPO) algorithm. This agent is a good place to start if you want to train your own agent. All of the options that are available in the CLI tool are available in the Tensorforce CLI.

An example with all three simple agents running ffa:

```bash
pom_tf_battle --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=PommeFFACompetition-v0
```