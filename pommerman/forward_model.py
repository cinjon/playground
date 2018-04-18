from collections import defaultdict

import numpy as np

from . import constants
from . import characters
from . import utility
from .agents import SimpleAgent


class ForwardModel(object):
    """Class for helping with the [forward] modeling of the game state."""
    def __init__(self):
        self.reset_times()

    def reset_times(self):
        self._time_avg = defaultdict(float)
        self._time_max = defaultdict(float)
        self._time_cnt = defaultdict(int)

    def _update_times(self, t, key):
        avg = self._time_avg[key]
        cnt = self._time_cnt[key]
        new_avg = (float(avg)*float(cnt) + float(t))
        new_avg /= float(cnt + 1)
        self._time_cnt[key] = cnt + 1
        self._time_avg[key] = new_avg
        self._time_max[key] = max(self._time_max[key], float(t))

    def run(self, num_times, board, agents, bombs, items, flames, is_partially_observable, agent_view_size, action_space, training_agent=None, is_communicative=False):
        """Run the forward model.

        Args:
          num_times: The number of times to run it for. This is a maximum and it will stop early if we reach a done.
          board: The board state to run it from.
          agents: The agents to use to run it.
          bombs: The starting bombs.
          items: The starting items.
          flames: The starting flames.
          is_partially_observable: Whether the board is partially observable or not. Only applies to TeamRadio.
          agent_view_size: If it's partially observable, then the size of the square that the agent can view.
          action_space: The actions that each agent can take.
          training_agent: The training agent to pass to done.
          is_communicative: Whether the action depends on communication observations as well.

        Returns:
          steps: The list of step results, which are each a dict of "obs", "next_obs", "reward", "action".
          board: Updated board.
          agents: Updated agents, same models though.
          bombs: Updated bombs.
          items: Updated items.
          flames: Updated flames.
          done: Whether we completed the game in these steps.
          info: The result of the game if it's completed.
        """
        steps = []
        for _ in num_times:
            obs = self.get_observations(
                board, agents, bombs, is_partially_observable, agent_view_size)
            actions = self.act(agents, obs, action_space,
                               is_communicative=is_communicative)
            board, agents, bombs, items, flames = self.step(
                actions, board, agents, bombs, items, flames)
            next_obs = self.get_observations(
                board, agents, bombs, is_partially_observable, agent_view_size)
            reward = self.get_rewards(agents, game_type, step_count, max_steps)
            done = self.get_done(agents, game_type, step_count, max_steps,
                                 training_agent)
            info = self.get_info(done, rewards, game_type, agents)

            steps.append({
                "obs": obs,
                "next_obs": next_obs,
                "reward": reward,
                "actions": actions,
            })
            if done:
                # Callback to let the agents know that the game has ended.
                for agent in agents:
                    agent.episode_end(reward[agent.agent_id])
                break
        return steps, board, agents, bombs, items, flames, done, info

    # @staticmethod               
    # def act(agents, obs, action_space, is_communicative=False):
    def act(self, agents, obs, action_space, is_communicative=False):
        """Returns actions for each agent in this list.

        Args:
          agents: A list of agent objects.
          obs: A list of matching observations per agent.
          action_space: The action space for the environment using this model.
          is_communicative: Whether the action depends on communication observations as well.

        Returns a list of actions.
        """
        # TODO: Use a timeout here.
        # @utils.timeout(0.15)
        def act_ex_communication(agent):
            if agent.is_alive:
                return agent.act(obs[agent.agent_id], action_space=action_space)
            else:
                return constants.Action.Stop.value

        def act_with_communication(agent):
            if agent.is_alive:
                action = agent.act(obs[agent.agent_id], action_space=action_space)
                if type(action) == int:
                    action = [action] + [0, 0]
                assert(type(action) == list)
                return action
            else:
                return [constants.Action.Stop.value, 0, 0]

        ret = []
        for agent in agents:
            with utility.Timer() as t:
                if is_communicative:
                    ret.append(act_with_communication(agent))
                else:
                    ret.append(act_ex_communication(agent))
            self._update_times(
                t.interval, "%s-%d" % (str(type(agent)), agent.agent_id))
        return ret

    def step(self, actions, curr_board, curr_agents, curr_bombs, curr_items,
             curr_flames):
        # We track what the agents each do in this dict.
        self.step_info = defaultdict(list)

        board_size = len(curr_board)

        # Tick the flames. Replace any dead ones with passages. If there is an item there, then reveal that item.
        flames = []
        for flame in curr_flames:
            position = flame.position
            if flame.is_dead():
                item_value = curr_items.get(position)
                if item_value:
                    del curr_items[position]
                else:
                    item_value = constants.Item.Passage.value
                curr_board[position] = item_value
            else:
                flame.tick()
                flames.append(flame)
        curr_flames = flames

        # Step the living agents.
        # If two agents try to go to the same spot, they should bounce back to their previous spots.
        # This is a little complicated because what if there are three agents all in a row.
        # If the one in the middle tries to go to the left and bounces with the one on the left,
        # and then the one on the right tried to go to the middle one's position, she should also bounce.
        # A way of doing this is to gather all the new positions before taking any actions.
        # Then, if there are disputes, correct those disputes iteratively.
        def make_counter(next_positions):
            counter = defaultdict(list)
            for num, next_position in enumerate(next_positions):
                if next_position is not None:
                    counter[next_position].append(num)
            return counter

        def has_position_conflict(counter):
            return any([len(agent_ids) > 1 for next_position, agent_ids in counter.items() if next_position])

        def dist_nearest_enemy(agent):
            agent_position = agent.position
            target_position = None
            distance = None
            enemies = agent.enemies
            seen = set()
            Q = [agent_position]
            while Q:
                position = Q.pop(0)
                if not utility.position_on_board(curr_board, position):
                    continue

                if position in seen:
                    continue
                seen.add(position)

                if utility.position_is_enemy(curr_board, position, enemies):
                    target_position = position
                    break

                x, y = position
                Q.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])

            if target_position is not None:
                xt, yt = target_position
                xa, ya = agent_position
                distance = abs(yt - ya) + abs(xt - xa)
            return distance

        curr_positions = [agent.position for agent in curr_agents]
        next_positions = [agent.position for agent in curr_agents]
        for agent, action in zip(curr_agents, actions):
            if agent.is_alive:
                position = agent.position

                if action == constants.Action.Stop.value:
                    agent.stop()
                elif action == constants.Action.Bomb.value:
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
                        dist_to_enemy = dist_nearest_enemy(agent)
                        if dist_to_enemy:
                            self.step_info[agent.agent_id].append(
                                'bomb:%d' % dist_to_enemy)
                elif utility.is_valid_direction(curr_board, position, action):
                    next_position = agent.get_next_position(action)

                    # This might be a bomb position. Only move in that case if the agent can kick.
                    if not utility.position_is_bomb(curr_board, next_position):
                        next_positions[agent.agent_id] = next_position
                    elif not agent.can_kick:
                        agent.stop()
                    else:
                        next_positions[agent.agent_id] = next_position
                else:
                    # The agent made an invalid direction.
                    agent.stop()
            else:
                next_positions[agent.agent_id] = None

        counter = make_counter(next_positions)
        while has_position_conflict(counter):
            for next_position, agent_ids in counter.items():
                if next_position and len(agent_ids) > 1:
                    for agent_id in agent_ids:
                        next_positions[agent_id] = curr_positions[agent_id]
            counter = make_counter(next_positions)

        for agent, curr_position, next_position, direction in zip(curr_agents, curr_positions, next_positions, actions):
            if not agent.is_alive:
                continue

            if curr_position != next_position:
                agent.move(direction)
                if agent.can_kick:
                    bombs = [bomb for bomb in curr_bombs if bomb.position == agent.position]
                    if bombs:
                        bombs[0].moving_direction = constants.Action(direction)

            if utility.position_is_powerup(curr_board, agent.position):
                agent.pick_up(constants.Item(curr_board[agent.position]))
                if constants.Item(curr_board[agent.position]) == constants.Item.Skull:
                    self.step_info[agent.agent_id].append('bad_item')
                else:
                    self.step_info[agent.agent_id].append('good_item')
                curr_board[agent.position] = constants.Item.Passage.value

        # Explode bombs.
        next_bombs = []
        exploded_map = np.zeros_like(curr_board)
        for bomb in curr_bombs:
            bomb.tick()
            if bomb.is_moving():
                invalid_values = list(range(len(constants.Item)+1))[1:]
                if utility.is_valid_direction(curr_board, bomb.position, bomb.moving_direction.value, invalid_values=invalid_values):
                    curr_board[bomb.position] = constants.Item.Passage.value
                    bomb.move()
                else:
                    bomb.stop()

            if bomb.exploded():
                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all([r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        if curr_board[r][c] == constants.Item.Wood.value:
                            break
            else:
                next_bombs.append(bomb)

        # Remove bombs that were in the blast radius.
        curr_bombs = []
        for bomb in next_bombs:
            if bomb.in_range(exploded_map):
                bomb.bomber.incr_ammo()
            else:
                curr_bombs.append(bomb)

        # Kill these agents.
        for agent in curr_agents:
            if agent.is_alive and agent.in_range(exploded_map):
                agent.die()
                self.step_info[agent.agent_id].append('dead')
        exploded_map = np.array(exploded_map)

        # Update the board
        for bomb in curr_bombs:
            curr_board[bomb.position] = constants.Item.Bomb.value

        for agent in curr_agents:
            position = np.where(curr_board == utility.agent_value(agent.agent_id))
            curr_board[position] = constants.Item.Passage.value
            if agent.is_alive:
                curr_board[agent.position] = utility.agent_value(agent.agent_id)

        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            curr_flames.append(characters.Flame((row, col)))
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        return curr_board, curr_agents, curr_bombs, curr_items, curr_flames

    def get_observations(self, curr_board, agents, bombs, is_partially_observable, agent_view_size):
        """Gets the observations as an np.array of the visible squares.

        The agent gets to choose whether it wants to keep the fogged part in memory.
        """
        board_size = len(curr_board)

        def make_bomb_maps(position):
            blast_strengths = np.zeros((board_size, board_size))
            life = np.zeros((board_size, board_size))

            for bomb in bombs:
                x, y = bomb.position
                if not is_partially_observable or in_view_range(position, x, y):
                    blast_strengths[(x, y)] = bomb.blast_strength
                    life[(x, y)] = bomb.life
            return blast_strengths, life

        def in_view_range(position, vrow, vcol):
            row, col = position
            return all([
                row >= vrow - agent_view_size, row < vrow + agent_view_size,
                col >= vcol - agent_view_size, col < vcol + agent_view_size])

        attrs = ['position', 'blast_strength', 'can_kick', 'teammate', 'ammo',
                 'enemies', 'is_alive']

        observations = []
        for agent in agents:
            agent_obs = {}
            board = curr_board
            if is_partially_observable:
                board = board.copy()
                for row in range(board_size):
                    for col in range(board_size):
                        if not in_view_range(agent.position, row, col):
                            board[row, col] = constants.Item.Fog.value
            agent_obs['board'] = board

            bomb_blast_strengths, bomb_life = make_bomb_maps(agent.position)
            agent_obs['bomb_blast_strength'] = bomb_blast_strengths
            agent_obs['bomb_life'] = bomb_life

            for attr in attrs:
                assert hasattr(agent, attr)
                agent_obs[attr] = getattr(agent, attr)
            observations.append(agent_obs)
        return observations

    @staticmethod
    def get_done(agents, step_count, max_steps, game_type, training_agents,
                 all_agents=False):
        alive = [agent for agent in agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if step_count >= max_steps:
            # The game is done. Return True.
            return [True]*4 if all_agents else True
        elif game_type == constants.GameType.FFA:
            training_agents_dead = all([agent not in alive_ids
                                        for agent in training_agents])
            if training_agents and training_agents_dead:
                # We have training_agents and they are all dead.
                return [True]*4 if all_agents else True
            else:
                if len(alive) <= 1:
                    # We have one or fewer agents left. The game is over.
                    return [True]*4 if all_agents else True
                elif all_agents:
                    # The game isn't over but we want data on all agents.
                    return [not agent.is_alive for agent in agents] 
                else:
                    # The game isn't over and we only want True or False.
                    return False
        else:
            if any([
                    len(alive_ids) <= 1,
                    alive_ids == [0, 2],
                    alive_ids == [1, 3],
            ]):
                # The game is done.
                return [True]*4 if all_agents else True
            else:
                # The game is not done. Return which are alive.
                if all_agents:
                    return [not agent.is_alive for agent in agents]
                else:
                    return False

    @staticmethod
    def get_info(done, rewards, game_type, agents):
        if type(done) == list:
            done = all(done)

        if game_type == constants.GameType.FFA:
            alive = [agent for agent in agents if agent.is_alive]
            if done and len(alive) > 1:
                return {
                    'result': constants.Result.Tie,
                }
            elif done:
                return {
                    'result': constants.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1]
                }
            else:
                return {
                    'result': constants.Result.Incomplete,
                }
        elif done:
            # We are playing a team game.
            if rewards == [-1]*4:
                return {
                    'result': constants.Result.Tie,
                }
            else:
                return {
                    'result': constants.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1],
                }
        else:
            return {
                'result': constants.Result.Incomplete,
            }

    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):
        def any_lst_equal(lst, values):
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]
        if game_type == constants.GameType.FFA:
            if len(alive_agents) == 1:
                # An agent won. Give them +1, others -1.
                return [2*int(agent.is_alive) - 1 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -1.
                return [-1]*4
            else:
                # Game running: 0 for alive, -1 for dead.
                return [int(agent.is_alive) - 1 for agent in agents]
        else:
            # We are playing a team game.
            if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
                # Team [0, 2] wins.
                return [1, -1, 1, -1]
            elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
                # Team [1, 3] wins.
                return [-1, 1, -1, 1]
            elif step_count >= max_steps:
                # Game is over by max_steps. All agents tie.
                return [-1]*4
            else:
                # No team has yet won or lost.
                return [0]*4
