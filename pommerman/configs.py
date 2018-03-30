from . import envs
from . import characters


def ffa_v0():
    """Start up a FFA config with the default settings."""
    env = envs.v0.Pomme
    game_type = envs.utility.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFA-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': envs.utility.RENDER_FPS,
    }
    agent = characters.Bomber
    return locals()


def ffa_v3():
    """Start up a FFA config with the default settings."""
    env = envs.v3.Pomme
    game_type = envs.utility.GameType.FFA
    env_entry_point = 'pommerman.envs.v3:Pomme'
    env_id = 'PommeFFA-v3'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': envs.utility.RENDER_FPS,
    }
    agent = characters.Agent
    return locals()


def ffa_v0_fast():
    """Start up a FFA config with the default settings."""
    env = envs.v0.Pomme
    game_type = envs.utility.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFAFast-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': 1000,
    }
    agent = characters.Bomber
    return locals()


def ffa_v1():
    """Start up a collapsing FFA config with the default settings."""
    env = envs.v1.Pomme
    game_type = envs.utility.GameType.FFA
    env_entry_point = 'pommerman.envs.v1:Pomme'
    env_id = 'PommeFFA-v1'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'first_collapse': envs.utility.FIRST_COLLAPSE,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': envs.utility.RENDER_FPS,
    }
    agent = characters.Bomber
    return locals()


def team_v0():
    """Start up a team config with the default settings."""
    env = envs.v0.Pomme
    game_type = envs.utility.GameType.Team
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeTeam-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': envs.utility.RENDER_FPS,
    }
    agent = characters.Bomber
    return locals()


def team_v0_fast():
    """Start up a team config with the default settings."""
    env = envs.v0.Pomme
    game_type = envs.utility.GameType.Team
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeTeamFast-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'render_fps': 2000,
    }
    agent = characters.Bomber
    return locals()


def radio_v2():
    """Start up a team radio config with the default settings."""
    env = envs.v2.Pomme
    game_type = envs.utility.GameType.TeamRadio
    env_entry_point = 'pommerman.envs.v2:Pomme'
    env_id = 'PommeRadio-v2'
    env_kwargs = {
        'game_type': game_type,
        'board_size': envs.utility.BOARD_SIZE,
        'agent_view_size': envs.utility.AGENT_VIEW_SIZE,
        'num_rigid': envs.utility.NUM_RIGID,
        'num_wood': envs.utility.NUM_WOOD,
        'num_items': envs.utility.NUM_ITEMS,
        'max_steps': envs.utility.MAX_STEPS,
        'is_partially_observable': True,
        'radio_vocab_size': envs.utility.RADIO_VOCAB_SIZE,
        'radio_num_words': envs.utility.RADIO_NUM_WORDS,
        'render_fps': envs.utility.RENDER_FPS,
    }
    agent = characters.Bomber
    return locals()
