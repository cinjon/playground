from collections import defaultdict

import numpy as np

from ..characters import Flame
from . import utility


class ForwardModel(object):
    """Class for helping with the [forward] modeling of the game state."""

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
                break
        return steps, board, agents, bombs, items, flames, done, info

    @staticmethod
    def act(agents, obs, action_space, is_communicative=False):
        """Returns actions for each agent in this list.

        Args:
          agents: A list of agent objects.
          obs: A list of matching observations per agent.
          action_space: The action space for the environment using this model.
          is_communicative: Whether the action depends on communication observations as well.

        Returns a list of actions.
        """
        def act_ex_communication(agent):
            if agent.is_alive:
                return agent.act(obs[agent.agent_id], action_space=action_space)
            else:
                return utility.Action.Stop.value

        def act_with_communication(agent):
            if agent.is_alive:
                action = agent.act(obs[agent.agent_id], action_space=action_space)
                if type(action) == int:
                    action = [action] + [0, 0]
                assert(type(action) == list)
                return action
            else:
                return [utility.Action.Stop.value, 0, 0]

        ret = []
        for agent in agents:
            if is_communicative:
                ret.append(act_with_communication(agent))
            else:
                ret.append(act_ex_communication(agent))
        return ret

    @staticmethod
    def step(actions, curr_board, curr_agents, curr_bombs, curr_items,
             curr_flames):
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
                    item_value = utility.Item.Passage.value
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

        curr_positions = [agent.position for agent in curr_agents]
        next_positions = [agent.position for agent in curr_agents]
        for agent, action in zip(curr_agents, actions):
            if agent.is_alive:
                position = agent.position

                if action == utility.Action.Stop.value:
                    agent.stop()
                elif action == utility.Action.Bomb.value:
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
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
                        bombs[0].moving_direction = utility.Action(direction)

            if utility.position_is_powerup(curr_board, agent.position):
                agent.pick_up(utility.Item(curr_board[agent.position]))
                curr_board[agent.position] = utility.Item.Passage.value

        # Explode bombs.
        next_bombs = []
        exploded_map = np.zeros_like(curr_board)
        for bomb in curr_bombs:
            bomb.tick()
            if bomb.is_moving():
                invalid_values = list(range(len(utility.Item)+1))[1:]
                if utility.is_valid_direction(curr_board, bomb.position, bomb.moving_direction.value, invalid_values=invalid_values):
                    curr_board[bomb.position] = utility.Item.Passage.value
                    bomb.move()
                else:
                    bomb.stop()

            if bomb.exploded():
                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all([r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == utility.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        if curr_board[r][c] == utility.Item.Wood.value:
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
            if agent.in_range(exploded_map):
                agent.die()
        exploded_map = np.array(exploded_map)

        # Update the board
        for bomb in curr_bombs:
            curr_board[bomb.position] = utility.Item.Bomb.value

        for agent in curr_agents:
            position = np.where(curr_board == utility.agent_value(agent.agent_id))
            curr_board[position] = utility.Item.Passage.value
            if agent.is_alive:
                curr_board[agent.position] = utility.agent_value(agent.agent_id)

        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            curr_flames.append(Flame((row, col)))
        for flame in curr_flames:
            curr_board[flame.position] = utility.Item.Flames.value

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
                 'enemies']

        observations = []
        for agent in agents:
            agent_obs = {}
            board = curr_board
            if is_partially_observable:
                board = board.copy()
                for row in range(board_size):
                    for col in range(board_size):
                        if not in_view_range(agent.position, row, col):
                            board[row, col] = utility.Item.Fog.value
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
    def get_done(agents, step_count, max_steps, game_type, training_agent):
        alive = [agent for agent in agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if step_count >= max_steps:
            return True
        elif game_type == utility.GameType.FFA:
            if training_agent is not None and training_agent not in alive_ids:
                return True
            return len(alive) <= 1
        elif any([
                len(alive_ids) <= 1,
                alive_ids == [0, 2],
                alive_ids == [1, 3],
        ]):
            return True
        return False

    @staticmethod
    def get_info(done, rewards, game_type, agents):
        if game_type == utility.GameType.FFA:
            alive = [agent for agent in agents if agent.is_alive]
            if done and len(alive) > 1:
                return {
                    'result': utility.Result.Tie,
                }
            elif done:
                return {
                    'result': utility.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1],
                }
            else:
                return {
                    'result': utility.Result.Incomplete,
                }
        elif done:
            if rewards == [1]*4:
                return {
                    'result': utility.Result.Tie,
                }
            else:
                return {
                    'result': utility.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1],
                }
        else:
            return {
                'result': utility.Result.Incomplete,
            }

    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):
        def any_lst_equal(lst, values):
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]
        if game_type == utility.GameType.FFA:
            if len(alive_agents) == 1 or step_count >= max_steps:
                # Game is over. All of the alive agents get reward.
                return [2*int(agent.is_alive) - 1 for agent in agents]
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
                return [1]*4
            else:
                # No team has yet won or lost.
                return [0]*4
