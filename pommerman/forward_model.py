from collections import defaultdict
import random
import sys

import numpy as np

from . import constants
from . import characters
from . import utility


class ForwardModel(object):
    """Class for helping with the [forward] modeling of the game state."""
    def run(self,
            num_times,
            board,
            agents,
            bombs,
            items,
            flames,
            is_partially_observable,
            agent_view_size,
            action_space,
            training_agent=None,
            is_communicative=False):
        """Run the forward model.
        Args:
          num_times: The number of times to run it for. This is a maximum and
            it will stop early if we reach a done.
          board: The board state to run it from.
          agents: The agents to use to run it.
          bombs: The starting bombs.
          items: The starting items.
          flames: The starting flames.
          is_partially_observable: Whether the board is partially observable or
            not. Only applies to TeamRadio.
          agent_view_size: If it's partially observable, then the size of the
            square that the agent can view.
          action_space: The actions that each agent can take.
          training_agent: The training agent to pass to done.
          is_communicative: Whether the action depends on communication
            observations as well.

        Returns:
          steps: The list of step results, which are each a dict of "obs",
            "next_obs", "reward", "action".
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
            actions = self.act(
                agents, obs, action_space, is_communicative=is_communicative)
            board, agents, bombs, items, flames = self.step(
                actions, board, agents, bombs, items, flames)
            next_obs = self.get_observations(
                board, agents, bombs, is_partially_observable, agent_view_size,
                max_steps)
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
    def act(self, agents, obs, action_space, is_communicative=False):
        """Returns actions for each agent in this list.
        Args:
          agents: A list of agent objects.
          obs: A list of matching observations per agent.
          action_space: The action space for the environment using this model.
          is_communicative: Whether the action depends on communication
            observations as well.

        Returns a list of actions.
        """
        def act_ex_communication(agent):
            if agent.is_alive:
                return agent.act(obs[agent.agent_id], action_space=action_space)
            else:
                return constants.Action.Stop.value

        def act_with_communication(agent):
            if agent.is_alive:
                action = agent.act(
                    obs[agent.agent_id], action_space=action_space)
                if type(action) == int:
                    action = [action] + [0, 0]
                assert (type(action) == list)
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

        return ret

    @staticmethod
    def expert_act(expert, obs, action_space):
        """Returns actions that the expert i.e. SimpleAgent would take."""
        # NOTE: the loop is so that it works for homogenous
        # if you want to give supervision to all 3 training agents
        # NOTE: this only works for simple
        actions = []
        for i in range(len(obs)):
            a = expert.act(obs[i], action_space=action_space)
            actions.append(a)
        return actions

    def step(self, actions, curr_board, curr_agents, curr_bombs, curr_items,
             curr_flames, max_blast_strength=10, selfbombing=True, do_print=False):
        self.step_info = {agent.agent_id: {} for agent in curr_agents}

        board_size = len(curr_board)
        tmp_board = curr_board.copy()

        # Tick the flames. Replace any dead ones with passages. If there is an
        # item there, then reveal that item.
        flames_dict = {}
        for flame in curr_flames:
            position = flame.position
            if flame.is_dead():
                item_value = curr_items.get(position)
                if item_value:
                    del curr_items[position]
                else:
                    item_value = constants.Item.Passage.value

                # NOTE: We add this in to deal with the fact that there may be
                # an agent there because of selfbombing.
                if curr_board[position] == constants.Item.Flames.value:
                    curr_board[position] = item_value
            else:
                flame.tick()
                flames_dict[position] = flame

        # Step the living agents and moving bombs.
        # If two agents try to go to the same spot, they should bounce back to
        # their previous spots. This is complicated with one example being when
        # there are three agents all in a row. If the one in the middle tries
        # to go to the left and bounces with the one on the left, and then the
        # one on the right tried to go to the middle one's position, she should
        # also bounce. A way of doing this is to gather all the new positions
        # before taking any actions. Then, if there are disputes, correct those
        # disputes iteratively.
        # Additionally, if two agents try to switch spots by moving into each
        # Figure out desired next position for alive agents
        alive_agents = [agent for agent in curr_agents if agent.is_alive]
        desired_agent_positions = [agent.position for agent in alive_agents]

        for num_agent, agent in enumerate(alive_agents):
            position = agent.position
            # We change the curr_board here as a safeguard. We will later
            # update the agent's new position.
            curr_board[position] = constants.Item.Passage.value
            action = actions[agent.agent_id]

            if action == constants.Action.Stop.value:
                pass
            elif action == constants.Action.Bomb.value:
                position = agent.position
                if not utility.position_is_bomb(curr_bombs, position):
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
            elif utility.is_valid_direction(curr_board, position, action):
                desired_agent_positions[num_agent] = agent.get_next_position(
                    action)

        # Gather desired next positions for moving bombs. Handle kicks later.
        desired_bomb_positions = [bomb.position for bomb in curr_bombs]

        for bomb_num, bomb in enumerate(curr_bombs):
            curr_board[bomb.position] = constants.Item.Passage.value
            if bomb.is_moving():
                desired_position = utility.get_next_position(
                    bomb.position, bomb.moving_direction)
                if utility.position_on_board(curr_board, desired_position) \
                   and not utility.position_is_powerup(curr_board, desired_position) \
                   and not utility.position_is_wall(curr_board, desired_position):
                    desired_bomb_positions[bomb_num] = desired_position

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        crossings = dict()
        def crossing(current, desired):
            current_x, current_y = current
            desired_x, desired_y = desired
            if current_x != desired_x:
                assert current_y == desired_y
                return ('X', min(current_x, desired_x), current_y)
            assert current_x == desired_x
            return ('Y', current_x, min(current_y, desired_y))

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                desired_position = desired_agent_positions[num_agent]
                border = crossing(agent.position, desired_position)
                if border in crossings:
                    # Crossed another agent - revert both to prior positions.
                    desired_agent_positions[num_agent] = agent.position
                    num_agent2, _ = crossings[border]
                    desired_agent_positions[num_agent2] = alive_agents[
                        num_agent2].position
                else:
                    crossings[border] = (num_agent, True)

        for bomb_num, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[bomb_num] != bomb.position:
                desired_position = desired_bomb_positions[bomb_num]
                border = crossing(bomb.position, desired_position)
                if border in crossings:
                    # Crossed - revert to prior position.
                    desired_bomb_positions[bomb_num] = bomb.position
                    num, isAgent = crossings[border]
                    if not isAgent:
                        # Crossed bomb - revert that to prior position as well.
                        desired_bomb_positions[num] = curr_bombs[num].position
                else:
                    crossings[border] = (bomb_num, False)

        # Deal with multiple agents or multiple bomb collisions on desired next
        # position by resetting desired position to current position for
        # everyone involved in the collision.
        agent_occupancy = defaultdict(int)
        bomb_occupancy = defaultdict(int)
        for desired_position in desired_agent_positions:
            agent_occupancy[desired_position] += 1
        for desired_position in desired_bomb_positions:
            bomb_occupancy[desired_position] += 1

        # Resolve >=2 agents or >=2 bombs trying to occupy the same space.
        change = True
        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Either another agent is going to this position or more than
                # one bomb is going to this position. In both scenarios, revert
                # to the original position.
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] > 1):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for bomb_num, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[bomb_num]
                curr_position = bomb.position
                if desired_position != curr_position and \
                      (bomb_occupancy[desired_position] > 1 or agent_occupancy[desired_position] > 1):
                    desired_bomb_positions[bomb_num] = curr_position
                    bomb_occupancy[curr_position] += 1
                    change = True

        # Handle kicks.
        bombs_kicked_by = dict()
        delayed_bomb_updates = []
        delayed_agent_updates = []

        # Loop through all bombs to see if they need a good kicking or cause
        # collisions with an agent.
        for bomb_num, bomb in enumerate(curr_bombs):
            desired_position = desired_bomb_positions[bomb_num]

            if agent_occupancy[desired_position] == 0:
                # There was never an agent around to kick or collide.
                continue

            agent_list = [
                (num_agent, agent) for (num_agent, agent) in enumerate(alive_agents) \
                if desired_position == desired_agent_positions[num_agent]]
            if not agent_list:
                # Agents moved from collision.
                continue

            # The agent_list should contain a single element at this point.
            try:
                assert (len(agent_list) == 1)
            except AssertionError as e:
                sys.stdout.write(", ".join([str(b.position) for b in curr_bombs]))
                sys.stdout.write(str(bomb_num))
                sys.stdout.write(", ".join([str(k[0]) for k in agent_list]))
                sys.stdout.write(", ".join([str(k) for k in desired_bomb_positions]))
                sys.stdout.write(str(agent_occupancy))
                sys.stdout.write(str(bomb_occupancy))
                sys.stdout.write(str(tmp_board))
                sys.stdout.write(str(curr_board))
                sys.stdout.write(", ".join([(a.position, actions[a.agent_id]) for a in alive_agents]))
            assert (len(agent_list) == 1)
            num_agent, agent = agent_list[0]

            if desired_position == agent.position:
                # Agent did not move
                if desired_position != bomb.position:
                    # Bomb moved, but agent did not. The bomb should revert
                    # and stop.
                    delayed_bomb_updates.append((bomb_num, bomb.position))
                continue

            # NOTE: At this point, we have that the agent in question tried to
            # move into this position.
            if not agent.can_kick:
                # If we move the agent at this point, then we risk having two
                # agents on a square in future iterations of the loop. So we
                # push this change to the next stage instead.
                delayed_bomb_updates.append((bomb_num, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))
                continue

            # Agent moved and can kick - see if the target for the kick never had anyhing on it
            direction = constants.Action(actions[agent.agent_id])
            target_position = utility.get_next_position(desired_position,
                                                        direction)
            if utility.position_on_board(curr_board, target_position) and \
                       agent_occupancy[target_position] == 0 and \
                       bomb_occupancy[target_position] == 0 and \
                       not utility.position_is_powerup(curr_board, target_position) and \
                       not utility.position_is_wall(curr_board, target_position):
                # Ok to update bomb desired location as we won't iterate over it again here
                # but we can not update bomb_occupancy on target position and need to check it again
                delayed_bomb_updates.append((bomb_num, target_position))
                bombs_kicked_by[bomb_num] = num_agent
                bomb.moving_direction = direction
                # Bombs may still collide and we then need to reverse bomb and agent ..
            else:
                delayed_bomb_updates.append((bomb_num, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))

        for (bomb_num, bomb_position) in delayed_bomb_updates:
            desired_bomb_positions[bomb_num] = bomb_position
            bomb_occupancy[bomb_position] += 1
            change = True

        for (num_agent, agent_position) in delayed_agent_updates:
            desired_agent_positions[num_agent] = agent_position
            agent_occupancy[agent_position] += 1
            change = True

        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] != 0):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position

                # This bomb may be a boomerang, i.e. it was kicked back to the
                # original location it moved from. If it is blocked now, it
                # can't be kicked and the agent needs to move back to stay
                # consistent with other movements.
                if desired_position == curr_position and num_bomb not in bombs_kicked_by:
                    continue

                bomb_occupancy_ = bomb_occupancy[desired_position]
                agent_occupancy_ = agent_occupancy[desired_position]
                if bomb_occupancy_ > 1 or agent_occupancy_ > 1:
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    num_agent = bombs_kicked_by.get(num_bomb)
                    if num_agent is not None:
                        agent = alive_agents[num_agent]
                        desired_agent_positions[num_agent] = agent.position
                        agent_occupancy[agent.position] += 1
                        del bombs_kicked_by[num_bomb]
                    change = True

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] == bomb.position and \
               not num_bomb in bombs_kicked_by:
                # Bomb was not kicked this turn and its desired position is its
                # current location. Stop it just in case it was moving before.
                bomb.stop()
            else:
                # Move bomb to the new position.
                # NOTE: We already set the moving direction up above.
                bomb.position = desired_bomb_positions[num_bomb]

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                agent.move(actions[agent.agent_id])
                if utility.position_is_powerup(curr_board, agent.position):
                    agent.pick_up(
                        constants.Item(curr_board[agent.position]),
                        max_blast_strength=max_blast_strength)
                    self.step_info[agent.agent_id]['item'] = True

        # Explode bombs.
        exploded_map = np.zeros_like(curr_board)
        # The exploded_causes is a dict of explosion position to list of agents
        # that caused that position to blow up.
        exploded_causes = defaultdict(list)
        has_new_explosions = False

        for bomb in curr_bombs:
            bomb.tick()
            if bomb.exploded():
                has_new_explosions = True
            elif curr_board[bomb.position] == constants.Item.Flames.value:
                bomb.fire()
                has_new_explosions = True

        # Chain the explosions.
        while has_new_explosions:
            next_bombs = []
            has_new_explosions = False
            for bomb in curr_bombs:
                if not bomb.exploded():
                    next_bombs.append(bomb)
                    continue

                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all(
                            [r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        exploded_causes[(r, c)].append(bomb.bomber.agent_id)
                        if curr_board[r][c] == constants.Item.Wood.value:
                            break

            curr_bombs = next_bombs
            for bomb in curr_bombs:
                if bomb.in_range(exploded_map):
                    bomb.fire()
                    has_new_explosions = True

        # Update the board's bombs.
        for bomb in curr_bombs:
            curr_board[bomb.position] = constants.Item.Bomb.value

        # Update the board's flames.
        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            position = (row, col)
            flames_dict[position] = characters.Flame(
                position, bomber_ids=exploded_causes[position])

        curr_flames = list(flames_dict.values())
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Kill agents on flames. Otherwise, update position on curr_board.
        # if do_print:
        # print(exploded_causes)
        for agent in alive_agents:
            position = agent.position
            flame = flames_dict.get(position)
            if flame is None:
                # Not on a flame.
                curr_board[agent.position] = utility.agent_value(agent.agent_id)
            else:
                if selfbombing or agent.agent_id not in flame.bomber_ids:
                    agent.die()
                else:
                    curr_board[agent.position] = utility.agent_value(agent.agent_id)

        return curr_board, curr_agents, curr_bombs, curr_items, curr_flames

    def step_grid(self, actions, curr_board, curr_agents, do_print=False):
        print("actions ", actions)
        print("curr board ", curr_board)
        print("curr agents ", curr_agents)
        print("do_print ", do_print)

        import pdb; pdb.set_trace()

        board_size = len(curr_board)
        tmp_board = curr_board.copy()

        self.step_info = {agent.agent_id: {} for agent in curr_agents}

        board_size = len(curr_board)
        tmp_board = curr_board.copy()

        # Tick the flames. Replace any dead ones with passages. If there is an
        # item there, then reveal that item.
        flames_dict = {}
        for flame in curr_flames:
            position = flame.position
            if flame.is_dead():
                item_value = curr_items.get(position)
                if item_value:
                    del curr_items[position]
                else:
                    item_value = constants.Item.Passage.value

                # NOTE: We add this in to deal with the fact that there may be
                # an agent there because of selfbombing.
                if curr_board[position] == constants.Item.Flames.value:
                    curr_board[position] = item_value
            else:
                flame.tick()
                flames_dict[position] = flame

        # Step the living agents and moving bombs.
        # If two agents try to go to the same spot, they should bounce back to
        # their previous spots. This is complicated with one example being when
        # there are three agents all in a row. If the one in the middle tries
        # to go to the left and bounces with the one on the left, and then the
        # one on the right tried to go to the middle one's position, she should
        # also bounce. A way of doing this is to gather all the new positions
        # before taking any actions. Then, if there are disputes, correct those
        # disputes iteratively.
        # Additionally, if two agents try to switch spots by moving into each
        # Figure out desired next position for alive agents
        alive_agents = [agent for agent in curr_agents if agent.is_alive]
        desired_agent_positions = [agent.position for agent in alive_agents]

        for num_agent, agent in enumerate(alive_agents):
            position = agent.position
            # We change the curr_board here as a safeguard. We will later
            # update the agent's new position.
            curr_board[position] = constants.Item.Passage.value
            action = actions[agent.agent_id]

            if action == constants.Action.Stop.value:
                pass
            elif action == constants.Action.Bomb.value:
                position = agent.position
                if not utility.position_is_bomb(curr_bombs, position):
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
            elif utility.is_valid_direction(curr_board, position, action):
                desired_agent_positions[num_agent] = agent.get_next_position(
                    action)

        # Gather desired next positions for moving bombs. Handle kicks later.
        desired_bomb_positions = [bomb.position for bomb in curr_bombs]

        for bomb_num, bomb in enumerate(curr_bombs):
            curr_board[bomb.position] = constants.Item.Passage.value
            if bomb.is_moving():
                desired_position = utility.get_next_position(
                    bomb.position, bomb.moving_direction)
                if utility.position_on_board(curr_board, desired_position) \
                   and not utility.position_is_powerup(curr_board, desired_position) \
                   and not utility.position_is_wall(curr_board, desired_position):
                    desired_bomb_positions[bomb_num] = desired_position

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        crossings = dict()
        def crossing(current, desired):
            current_x, current_y = current
            desired_x, desired_y = desired
            if current_x != desired_x:
                assert current_y == desired_y
                return ('X', min(current_x, desired_x), current_y)
            assert current_x == desired_x
            return ('Y', current_x, min(current_y, desired_y))

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                desired_position = desired_agent_positions[num_agent]
                border = crossing(agent.position, desired_position)
                if border in crossings:
                    # Crossed another agent - revert both to prior positions.
                    desired_agent_positions[num_agent] = agent.position
                    num_agent2, _ = crossings[border]
                    desired_agent_positions[num_agent2] = alive_agents[
                        num_agent2].position
                else:
                    crossings[border] = (num_agent, True)

        for bomb_num, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[bomb_num] != bomb.position:
                desired_position = desired_bomb_positions[bomb_num]
                border = crossing(bomb.position, desired_position)
                if border in crossings:
                    # Crossed - revert to prior position.
                    desired_bomb_positions[bomb_num] = bomb.position
                    num, isAgent = crossings[border]
                    if not isAgent:
                        # Crossed bomb - revert that to prior position as well.
                        desired_bomb_positions[num] = curr_bombs[num].position
                else:
                    crossings[border] = (bomb_num, False)

        # Deal with multiple agents or multiple bomb collisions on desired next
        # position by resetting desired position to current position for
        # everyone involved in the collision.
        agent_occupancy = defaultdict(int)
        bomb_occupancy = defaultdict(int)
        for desired_position in desired_agent_positions:
            agent_occupancy[desired_position] += 1
        for desired_position in desired_bomb_positions:
            bomb_occupancy[desired_position] += 1

        # Resolve >=2 agents or >=2 bombs trying to occupy the same space.
        change = True
        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Either another agent is going to this position or more than
                # one bomb is going to this position. In both scenarios, revert
                # to the original position.
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] > 1):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for bomb_num, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[bomb_num]
                curr_position = bomb.position
                if desired_position != curr_position and \
                      (bomb_occupancy[desired_position] > 1 or agent_occupancy[desired_position] > 1):
                    desired_bomb_positions[bomb_num] = curr_position
                    bomb_occupancy[curr_position] += 1
                    change = True

        # Handle kicks.
        bombs_kicked_by = dict()
        delayed_bomb_updates = []
        delayed_agent_updates = []

        # Loop through all bombs to see if they need a good kicking or cause
        # collisions with an agent.
        for bomb_num, bomb in enumerate(curr_bombs):
            desired_position = desired_bomb_positions[bomb_num]

            if agent_occupancy[desired_position] == 0:
                # There was never an agent around to kick or collide.
                continue

            agent_list = [
                (num_agent, agent) for (num_agent, agent) in enumerate(alive_agents) \
                if desired_position == desired_agent_positions[num_agent]]
            if not agent_list:
                # Agents moved from collision.
                continue

            # The agent_list should contain a single element at this point.
            try:
                assert (len(agent_list) == 1)
            except AssertionError as e:
                sys.stdout.write(", ".join([str(b.position) for b in curr_bombs]))
                sys.stdout.write(str(bomb_num))
                sys.stdout.write(", ".join([str(k[0]) for k in agent_list]))
                sys.stdout.write(", ".join([str(k) for k in desired_bomb_positions]))
                sys.stdout.write(str(agent_occupancy))
                sys.stdout.write(str(bomb_occupancy))
                sys.stdout.write(str(tmp_board))
                sys.stdout.write(str(curr_board))
                sys.stdout.write(", ".join([(a.position, actions[a.agent_id]) for a in alive_agents]))
            assert (len(agent_list) == 1)
            num_agent, agent = agent_list[0]

            if desired_position == agent.position:
                # Agent did not move
                if desired_position != bomb.position:
                    # Bomb moved, but agent did not. The bomb should revert
                    # and stop.
                    delayed_bomb_updates.append((bomb_num, bomb.position))
                continue

            # NOTE: At this point, we have that the agent in question tried to
            # move into this position.
            if not agent.can_kick:
                # If we move the agent at this point, then we risk having two
                # agents on a square in future iterations of the loop. So we
                # push this change to the next stage instead.
                delayed_bomb_updates.append((bomb_num, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))
                continue

            # Agent moved and can kick - see if the target for the kick never had anyhing on it
            direction = constants.Action(actions[agent.agent_id])
            target_position = utility.get_next_position(desired_position,
                                                        direction)
            if utility.position_on_board(curr_board, target_position) and \
                       agent_occupancy[target_position] == 0 and \
                       bomb_occupancy[target_position] == 0 and \
                       not utility.position_is_powerup(curr_board, target_position) and \
                       not utility.position_is_wall(curr_board, target_position):
                # Ok to update bomb desired location as we won't iterate over it again here
                # but we can not update bomb_occupancy on target position and need to check it again
                delayed_bomb_updates.append((bomb_num, target_position))
                bombs_kicked_by[bomb_num] = num_agent
                bomb.moving_direction = direction
                # Bombs may still collide and we then need to reverse bomb and agent ..
            else:
                delayed_bomb_updates.append((bomb_num, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))

        for (bomb_num, bomb_position) in delayed_bomb_updates:
            desired_bomb_positions[bomb_num] = bomb_position
            bomb_occupancy[bomb_position] += 1
            change = True

        for (num_agent, agent_position) in delayed_agent_updates:
            desired_agent_positions[num_agent] = agent_position
            agent_occupancy[agent_position] += 1
            change = True

        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] != 0):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position

                # This bomb may be a boomerang, i.e. it was kicked back to the
                # original location it moved from. If it is blocked now, it
                # can't be kicked and the agent needs to move back to stay
                # consistent with other movements.
                if desired_position == curr_position and num_bomb not in bombs_kicked_by:
                    continue

                bomb_occupancy_ = bomb_occupancy[desired_position]
                agent_occupancy_ = agent_occupancy[desired_position]
                if bomb_occupancy_ > 1 or agent_occupancy_ > 1:
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    num_agent = bombs_kicked_by.get(num_bomb)
                    if num_agent is not None:
                        agent = alive_agents[num_agent]
                        desired_agent_positions[num_agent] = agent.position
                        agent_occupancy[agent.position] += 1
                        del bombs_kicked_by[num_bomb]
                    change = True

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] == bomb.position and \
               not num_bomb in bombs_kicked_by:
                # Bomb was not kicked this turn and its desired position is its
                # current location. Stop it just in case it was moving before.
                bomb.stop()
            else:
                # Move bomb to the new position.
                # NOTE: We already set the moving direction up above.
                bomb.position = desired_bomb_positions[num_bomb]

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                agent.move(actions[agent.agent_id])
                if utility.position_is_powerup(curr_board, agent.position):
                    agent.pick_up(
                        constants.Item(curr_board[agent.position]),
                        max_blast_strength=max_blast_strength)
                    self.step_info[agent.agent_id]['item'] = True

        # Explode bombs.
        exploded_map = np.zeros_like(curr_board)
        # The exploded_causes is a dict of explosion position to list of agents
        # that caused that position to blow up.
        exploded_causes = defaultdict(list)
        has_new_explosions = False

        for bomb in curr_bombs:
            bomb.tick()
            if bomb.exploded():
                has_new_explosions = True
            elif curr_board[bomb.position] == constants.Item.Flames.value:
                bomb.fire()
                has_new_explosions = True

        # Chain the explosions.
        while has_new_explosions:
            next_bombs = []
            has_new_explosions = False
            for bomb in curr_bombs:
                if not bomb.exploded():
                    next_bombs.append(bomb)
                    continue

                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all(
                            [r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        exploded_causes[(r, c)].append(bomb.bomber.agent_id)
                        if curr_board[r][c] == constants.Item.Wood.value:
                            break

            curr_bombs = next_bombs
            for bomb in curr_bombs:
                if bomb.in_range(exploded_map):
                    bomb.fire()
                    has_new_explosions = True

        # Update the board's bombs.
        for bomb in curr_bombs:
            curr_board[bomb.position] = constants.Item.Bomb.value

        # Update the board's flames.
        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            position = (row, col)
            flames_dict[position] = characters.Flame(
                position, bomber_ids=exploded_causes[position])

        curr_flames = list(flames_dict.values())
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Kill agents on flames. Otherwise, update position on curr_board.
        # if do_print:
        # print("SELFBOMBING: ", selfbombing)c
        # print(exploded_causes)
        for agent in alive_agents:
            position = agent.position
            flame = flames_dict.get(position)
            if flame is None:
                # Not on a flame.
                curr_board[agent.position] = utility.agent_value(agent.agent_id)
            else:
                if selfbombing or agent.agent_id not in flame.bomber_ids:
                    agent.die()
                else:
                    curr_board[agent.position] = utility.agent_value(agent.agent_id)

        return curr_board, curr_agents, curr_bombs, curr_items, curr_flames


        return curr_board

    def get_observations(self, curr_board, agents, bombs,
                         is_partially_observable, agent_view_size,
                         max_steps, step_count=None):
        """Gets the observations as an np.array of the visible squares.

        The agent gets to choose whether it keeps the fogged part in memory.
        """
        board_size = len(curr_board)

        def make_bomb_maps(position):
            blast_strengths = np.zeros((board_size, board_size))
            life = np.zeros((board_size, board_size))

            for bomb in bombs:
                x, y = bomb.position
                if not is_partially_observable \
                   or in_view_range(position, x, y):
                    blast_strengths[(x, y)] = bomb.blast_strength
                    life[(x, y)] = bomb.life
            return blast_strengths, life

        def in_view_range(position, vrow, vcol):
            row, col = position
            return all([
                row >= vrow - agent_view_size, row <= vrow + agent_view_size,
                col >= vcol - agent_view_size, col <= vcol + agent_view_size
            ])

        attrs = [
            'position', 'blast_strength', 'can_kick', 'teammate', 'ammo',
            'enemies', 'is_alive'
        ]
        alive_agents = [utility.agent_value(agent.agent_id)
                        for agent in agents if agent.is_alive]

        observations = []
        for agent in agents:
            agent_obs = {'alive': alive_agents}
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
            if step_count is not None:
                agent_obs['step'] = 1.0 * step_count / max_steps

            for attr in attrs:
                assert hasattr(agent, attr)
                agent_obs[attr] = getattr(agent, attr)
            observations.append(agent_obs)

        return observations

    def get_observations_grid(self, curr_board, agents,
                              max_steps, step_count=None):
        # TODO: do we want the step in the obs or not?
        """Gets the observations as an np.array of the visible squares."""

        board_size = len(curr_board)
        attrs = ['position', 'goal_position', 'step']

        board = curr_board
        observations = []
        agent_obs = {'board': board}

        for attr in attrs:
            assert hasattr(agents[0], attr)
            agent_obs[attr] = getattr(agents[0], attr)
        observations.append(agent_obs)

        return observations

    @staticmethod
    def get_done(agents, step_count, max_steps, game_type, training_agents,
                 all_agents=False, agent_pos=None, goal_pos=None):

        if game_type == constants.GameType.Grid:
            if agent_pos == goal_pos or step_count >= max_steps:
                # the agent reached the goal
                return True
            else:
                # the agent has not yet reached the goal
                return False
        else:
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
    def get_done_grid(agent_pos, goal_pos, step_count, max_steps):
        if agent_pos == goal_pos or step_count >= max_steps:
            # the agent reached the goal
            return True
        else:
            # the agent has not yet reached the goal
            return False

    @staticmethod
    def get_info(done, rewards, game_type, agents, training_agents=None):
        if type(done) == list:
            done = all(done)

        alive = [agent for agent in agents if agent.is_alive]
        if game_type == constants.GameType.FFA:
            alive = [agent for agent in agents if agent.is_alive]
            if done:
                if len(alive) == 0:
                    return {
                        'result': constants.Result.Tie,
                        'alive': [agent.agent_id for agent in alive]
                    }
                elif len(alive) > 1:
                    if training_agents is not None and not any([
                        agent.agent_id in training_agents
                        for agent in alive
                    ]):
                        return {
                            'result': constants.Result.Loss,
                            'alive': [agent.agent_id for agent in alive]
                        }
                    else:
                        return {
                            'result': constants.Result.Tie,
                            'alive': [agent.agent_id for agent in alive]
                        }
                else:
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
            if rewards == [-1] * 4:
                return {
                    'result': constants.Result.Tie,
                    'alive': [agent.agent_id for agent in alive]
                }
            else:
                return {
                    'result': constants.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1],
                    'alive': [agent.agent_id for agent in alive]
                }
        else:
            return {
                'result': constants.Result.Incomplete,
            }

    @staticmethod
    def get_info_grid(done, agent_loc, goal_loc):
        # TODO: if done is not list remove below lines; o/w keep them
        if type(done) == list:
            print(" !!! type of done is list ", done, type(done))
            done = all(done)

        if done:
            if agent_loc == goal_loc:
                return {
                    'result': constants.Result.Win
                }
            else:
                return {
                    'result': constants.Result.Loss
                }
        else:
            return {
                'result': constants.Result.Incomplete
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
                return [2 * int(agent.is_alive) - 1 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -1.
                return [-1] * 4
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
                return [-1] * 4
            else:
                # No team has yet won or lost.
                return [0] * 4

    @staticmethod
    def get_rewards_grid(agent_pos, goal_pos):
        if agent_pos == goal_pos:
            # the agent reached the goal
            return [1]
        else:
            # the agent has not yet reached the goal
            return [-0.1]
