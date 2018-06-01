from collections import defaultdict
import copy
import random
import time

import numpy as np
from pommerman import utility, constants, characters, forward_model, agents


verbose = False

class State():

    def am_I_alive(self):
        for i, agent in enumerate(self.curr_agents):
            if (agent.agent_id + 10 == self.self_agent.value):
                return agent.is_alive

    def __init__(self, obs, init=False, bombing_agents={}, board_size=11):
        self._game_mode = constants.GameType.FFA
        self.move = None

        self._board_size = board_size
        self._obs = obs
        self._my_position = tuple(obs['position'])
        self._board = np.array(obs['board'])
        self._bomb_life = np.array(self._obs['bomb_life'])
        self._teammate = obs['teammate']
        self._enemies = [constants.Item(e) for e in obs['enemies']]
        self._ammo = int(obs['ammo'])
        self.fm = forward_model.ForwardModel()

        self.self_agent = self.find_self_agent(self._obs)

        agents_id = [
            constants.Item.Agent0, constants.Item.Agent1, constants.Item.Agent2,
            constants.Item.Agent3
        ]

        self._agents = [
            characters.Bomber(aid.value, "FFA") for aid in agents_id
        ]  # remember to modifiy if it is team or radio mode

        self.bombing_agents = copy.deepcopy(bombing_agents)

        self.score = 0
        if init:
            self.curr_flames = self.convert_flames(
                self._board)  # determine by confirming the map
            self.curr_bombs = self.convert_bombs(
                np.array(obs['bomb_blast_strength']),
                np.array(obs['bomb_life']))
            self.curr_items = self.convert_items(self._board)
            self.curr_agents = self.convert_agents(self._board)
            self.last_items = self.curr_items
            if (bombing_agents != {}):
                self.curr_bombs = self.convert_bombs_two(
                    np.array(self._obs['bomb_blast_strength']), self._bomb_life,
                    bombing_agents)

    def advance_game_on_copy(self, action):
        board = copy.deepcopy(self._board)
        curr_flames = self.convert_flames(board)
        curr_items = self.convert_items(board)
        curr_agents = self.convert_agents(board)

        bombing_agents = copy.deepcopy(self.bombing_agents)
        actions = [-1 for i in range(4)]
        self_agent_value = self.self_agent if type(
            self.self_agent) == int else self.self_agent.value
        self.self_agent_value = self_agent_value
        actions[self_agent_value - 10] = action

        agents_to_bomb = []

        # create the info for the enemies and give them info

        for i, agent in enumerate(self.curr_agents):
            if (agent.is_alive):
                if (agent.agent_id + 10 != self_agent_value):
                    copy_obs = copy.deepcopy(self._obs)
                    # modify enemies
                    agent_idx = None

                    if verbose:
                        print(copy_obs)
                    for j, enemy in enumerate(copy_obs['enemies']):
                        enemyId = enemy if type(enemy) == int else enemy.value
                        if verbose:
                            print(agent.agent_id + 10, enemyId)

                        if agent.agent_id + 10 == enemyId:
                            agent_idx = j
                            break

                    agent_val = copy_obs['enemies'][agent_idx] if type(
                        copy_obs['enemies'][agent_idx]) == int else copy_obs[
                            'enemies'][agent_idx].value
                    del copy_obs['enemies'][agent_idx]
                    copy_obs['enemies'].append(self.self_agent)
                    # modify my position
                    my_position = np.where(self._board == agent.agent_id + 10)
                    copy_obs['position'] = (my_position[0][0],
                                            my_position[1][0])
                    # fuse_everything_in_new_obs
                    copy_obs['ammo'] = 1
                    # actions.append(agent.act(copy_obs)) # REFRACTION --> place action according to the agent id
                    agent_action = agent.act(copy_obs, [
                        constants.Action.Up, constants.Action.Stop,
                        constants.Action.Down, constants.Action.Left,
                        constants.Action.Right, 5
                    ])
                    actions[agent_val - 10] = agent_action
                    # they are responsible for dropping a bomb
                    if (agent_action == 5):
                        agents_to_bomb.append(((my_position[0][0],
                                                my_position[1][0]),
                                               agent.agent_id))

        # calc the bombs for this board
        curr_bombs = self.convert_bombs_two(
            np.array(self._obs['bomb_blast_strength']), self._bomb_life,
            bombing_agents)
        temp_board, temp_curr_agent, temp_curr_bombs, temp_curr_items, temp_curr_flames = self.fm.step(
            actions, board, curr_agents, curr_bombs, curr_items, curr_flames)

        # if an enemy or player is going to bomb, add it
        for a in agents_to_bomb:
            bombing_agents[a[0]] = a[1]
        if action == 5:
            bombing_agents[(self._my_position[0],
                            self._my_position[1])] = self_agent_value - 10

        return temp_board, temp_curr_agent, temp_curr_bombs, temp_curr_items, temp_curr_flames, bombing_agents

    def get_all_possible_states(self):
        list_of_states = []
        moves = [
            constants.Action.Stop, constants.Action.Up, constants.Action.Down,
            constants.Action.Left, constants.Action.Right, 5
        ]
        # check if move will land me on top of a bomb
        #unsafe_directions = self._directions_in_range_of_bomb(self._board, self._my_position, self.curr_bombs)
        # if unsafe_directions:
        #    if(len(unsafe_directions) != 4):
        #        for i in unsafe_directions:
        #print("get all possible states, removing unsafe move", i)
        #            moves.remove(i)
        # #if I am on a bomb, remove stop
        # if self._bomb_life[self._my_position[0]][self._my_position[1]] > 0:
        #     if constants.Action.Stop in moves:
        #         moves.remove(constants.Action.Stop)
        lost_all_moves = False
        if len(moves) == 0:
            lost_all_moves = True
            # input("FKING HELL")
            moves = [
                constants.Action.Up, constants.Action.Down,
                constants.Action.Left, constants.Action.Right,
                constants.Action.Stop, 5
            ]
        for move in moves:
            if move == 5 or utility.is_valid_direction(self._board,
                                                       self._my_position, move):
                # check if position is passible
                check_pos = None
                if move == constants.Action.Up:
                    check_pos = (self._my_position[0] - 1, self._my_position[1])
                elif move == constants.Action.Down:
                    check_pos = (self._my_position[0] + 1, self._my_position[1])
                elif move == constants.Action.Left:
                    check_pos = (self._my_position[0], self._my_position[1] - 1)
                elif move == constants.Action.Right:
                    check_pos = (self._my_position[0], self._my_position[1] + 1)
                if check_pos != None:
                    if not utility.position_is_passable(self._board, check_pos,
                                                        self._enemies):
                        # if i am blocked by a bomb, try kicking
                        if self._obs['can_kick']:
                            if move == constants.Action.Up:
                                if self._board[self._my_position[0] -
                                               1][self._my_position[1]] == 3:
                                    if self._my_position[0] - 2 >= 0:
                                        if self._board[self._my_position[0] -
                                                       2][self.
                                                          _my_position[1]] != 0:
                                            # print("removing non passable move", move)
                                            continue
                                    else:
                                        # print("removing non passable move", move)
                                        continue
                            elif move == constants.Action.Down:
                                if self._board[self._my_position[0] +
                                               1][self._my_position[1]] == 3:
                                    if self._my_position[0] + 2 < self._board_size:
                                        if self._board[self._my_position[0] +
                                                       2][self.
                                                          _my_position[1]] != 0:
                                            # print("removing non passable move", move)
                                            continue
                                    else:
                                        # print("removing non passable move", move)
                                        continue
                            elif move == constants.Action.Left:
                                if self._board[self._my_position[0]][
                                        self._my_position[1] - 1] == 3:
                                    if self._my_position[1] - 2 >= 0:
                                        if self._board[self.
                                                       _my_position[0]][self.
                                                                        _my_position[1]
                                                                        -
                                                                        2] != 0:
                                            # print("removing non passable move", move)
                                            continue
                                    else:
                                        # print("removing non passable move", move)
                                        continue
                            elif move == constants.Action.Right:
                                if self._board[self._my_position[0]][
                                        self._my_position[1] + 1] == 3:
                                    if self._my_position[1] + 2 < self._board_size:
                                        if self._board[self.
                                                       _my_position[0]][self.
                                                                        _my_position[1]
                                                                        +
                                                                        2] != 0:
                                            # print("removing non passable move", move)
                                            continue
                                    else:
                                        # print("removing non passable move", move)
                                        continue
                            else:
                                # print("removing non passable move", move)
                                continue
                        else:
                            # print("removing non passable move", move)
                            continue
                # check to see if its a safe dir
                if move == 5 and self._ammo == 0:
                    # print("bombing without a bomb, skip")
                    # can not bomb with no ammo
                    continue

                # if I am on a bomb, lets not bomb
                if move == 5 and self._my_position in self.bombing_agents:
                    # print("bombing while on bomb, skip")
                    continue

                temp_board, temp_curr_agent, temp_curr_bombs, temp_curr_items, temp_curr_flames, bombing_agents = self.advance_game_on_copy(
                    move)

                temp_obs = self.fm.get_observations(
                    temp_board, temp_curr_agent, temp_curr_bombs, False,
                    self._board_size, 800)[self.self_agent_value - 10]
                temp_obs['ammo'] = self._ammo
                if move == 5:
                    bombing_agents[(
                        self._my_position[0],
                        self._my_position[1])] = self.self_agent_value - 10
                    temp_obs['ammo'] = self._ammo - 1

                temp_obs['enemies'] = self._enemies

                temp_state = State(temp_obs, True, board_size=self._board_size)
                temp_state.bombing_agents = bombing_agents
                temp_state.move = move

                temp_state.score = temp_state.get_score()
                temp_state.score -= 0.1

                # IF THE SCORE IS NEGATIVE, WE DONT WANT THIS STATE
                # IF THE AGENT IS DEAD, NEGATIVE
                if not temp_state.am_I_alive:
                    temp_state.score -= 100

                if lost_all_moves == True:
                    temp_state.score -= 200

                list_of_states.append(temp_state)
        return list_of_states

    def advance_game(self, action):
        actions = [-1 for i in range(4)]

        self_agent = self.self_agent
        actions[self_agent.value - 10] = action

        for i, agent in enumerate(self.curr_agents):
            my_position = np.where(self._board == agent.agent_id + 10)
            agent.is_alive = my_position[0]
            if (agent.is_alive):
                if (agent.agent_id + 10 != self_agent.value):
                    copy_obs = copy.deepcopy(self._obs)
                    # modify enemies
                    agent_idx = None
                    for j, enemy in enumerate(copy_obs['enemies']):
                        if agent.agent_id + 10 == enemy.value:
                            agent_idx = j
                            break

                    agent_val = copy_obs['enemies'][agent_idx].value
                    del copy_obs['enemies'][agent_idx]
                    copy_obs['enemies'].append(self_agent)
                    # modify my position
                    copy_obs['position'] = (my_position[0][0],
                                            my_position[1][0])
                    # fuse_everything_in_new_obs
                    copy_obs['ammo'] = 1
                    # actions.append(agent.act(copy_obs)) # REFRACTION --> place action according to the agent id
                    agent_action = agent.act(copy_obs, [
                        constants.Action.Stop, constants.Action.Up,
                        constants.Action.Down, constants.Action.Left,
                        constants.Action.Right, 5
                    ])
                    actions[agent_val - 10] = agent_action
                    # they are responsible for dropping a bomb
                    if (agent_action == 5):
                        self.bombing_agents[(
                            my_position[0][0],
                            my_position[1][0])] = agent.agent_id

        self.last_items = self.curr_items

        self._board, self.curr_agents, self.curr_bombs, self.curr_items, self.curr_flames = \
            self.fm.step(actions, self._board, self.curr_agents,
                         self.curr_bombs, self.curr_items, self.curr_flames)
        self.update_obs()
        self.score -= 0.1

    def update_obs(self):
        self._obs = self.fm.get_observations(
            self._board, self.curr_agents, self.curr_bombs, False,
            self._board_size, 800)[self.self_agent.value - 10]
        self._obs['enemies'] = self._enemies

    def copy_from(self, source_node):
        self._obs = copy.deepcopy(source_node._obs)
        self._board = copy.deepcopy(source_node.curr_agents)
        self.curr_bombs = copy.deepcopy(source_node.curr_bombs)
        self.curr_items = copy.deepcopy(source_node.curr_items)
        self.curr_flames = copy.deepcopy(source_node.curr_flames)
        self.score = source_node.score
        self.last_items = copy.deepcopy(source_node.last_items)

    def get_score(self):
        score = 0
        # self_agent = self.self_agent
        self_agent_value = self.self_agent if type(
            self.self_agent) == int else self.self_agent.value
        # if the enemy agent is not alive, then the score increases
        for i, agent in enumerate(self.curr_agents):
            if 10 + i != self_agent_value:
                if not agent.is_alive:
                    score += 5
            else:
                # if we are dead, fk
                if not agent.is_alive:
                    score -= score * 0.95

        # if the agent is close to its enemy, then the score goes up
        self_agent_instance = self.curr_agents[self_agent_value - 10]
        for i, agent in enumerate(self.curr_agents):
            if 10 + i == self_agent_value:
                continue
            if not agent.is_alive:
                continue
            tar, tac = agent.position  # target agent row, target agent column
            sar, sac = self_agent_instance.position
            distance = abs(tar - sar) + abs(
                tac - sac)  # (((tar - sar) ** 2 + (tac - sac) ** 2) ** 0.5
            if distance != 0:
                score += (1 / distance * 5) * 5
        # if the agent has eaten good stuff, then score goes up
        if self._obs['position'] in self.last_items:
            val = self.last_items[self._obs['position']]
            # if val != constants.Item.Skull.value:
            score += 5
            # else:
            # score -= 5

        # if I placed bomb near wood, then score goes up
        for (k, v) in self.bombing_agents.items():
            if v is self_agent_value - 10 and self._bomb_life[k[0]][k[1]] == 24:
                for p in [(k[0] - 1, k[1]), (k[0] + 1, k[1]), (k[0], k[1] - 1),
                          (k[0], k[1] + 1)]:
                    if utility.position_on_board(self._board, p) and \
                       self._board[p] == constants.Item.Wood.value:
                        score += 3
        return score

    def find_self_agent(self, obs):
        agents = [
            constants.Item.Agent0, constants.Item.Agent1, constants.Item.Agent2,
            constants.Item.Agent3
        ]
        if type(obs['enemies'][0]) == int:
            agents = [a.value for a in agents]
        for agent in agents:
            if agent not in obs['enemies'] and agent != self._teammate:
                return agent

    def convert_bombs(self, strength_map, life_map):
        ret = []
        locations = np.where(strength_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(strength_map[(r, c)]),
                'bomb_life': int(life_map[(r, c)]),
                'moving_direction': None
            })
        return self.make_bomb_items(ret)

    def convert_bombs_two(self, strength_map, life_map, bombing_agents):
        ret = []
        locations = np.where(strength_map > 0)
        i = 0
        #print("life map \n", life_map)
        #print("bombing locations", locations)
        #print("boming agent keys", bombing_agents.keys())
        for r, c in zip(locations[0], locations[1]):
            b = bombing_agents.get((r, c), -1)
            if b == -1:
                continue
            # TAKE CARE OF TICKED BOMBS, SELECT A RANDOM ENEMY AGENT AS THE BOMBER
            ret.append({
                'position': (r, c),
                'blast_strength': int(strength_map[(r, c)]),
                'bomb_life': int(life_map[(r, c)]),
                'moving_direction': None,
                'agent': b
            })
            i += 1
        return self.make_bomb_items_two(ret)

    def convert_flames(self, board):
        # Assuming that for each flame object, its life span is 2 ticks
        ret = []
        locations = np.where(board == 4)
        for r, c in zip(locations[0], locations[1]):
            ret.append(characters.Flame((r, c)))
        return ret

    def convert_items(self, board):
        ret = {}
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                v = board[r][c]
                if v in [
                        constants.Item.ExtraBomb.value,
                        constants.Item.IncrRange.value,
                        constants.Item.Kick.value
                ]:
                    # constants.Item.Skull.value]:
                    ret[(r, c)] = v
        return ret

    def convert_agents(self, board):
        ret = []
        for aid in range(10, 14):
            locations = np.where(board == aid)
            # agt = agents.AggressiveAgent()
            agt = agents.DummyAgent()
            agt.init_agent(aid, self._game_mode)
            if len(locations[0]) != 0:
                agt.set_start_position((locations[0][0], locations[1][0]))
            else:
                agt.set_start_position((0, 0))
                agt.is_alive = False
            agt.reset(is_alive=agt.is_alive)
            agt.agent_id = aid - 10
            ret.append(agt)
        return ret

    def make_bomb_items(self, ret):
        bomb_obj_list = []

        # FIX THIS LINE OF CODE BELOW
        # IT SEEMS LIKE THE THE VARIABLE IS DECLARED NOT INSTANTIATED
        # this_variable_is_useless = agents.AggressiveAgent()
        this_variable_is_useless = agents.DummyAgent()
        for i in ret:
            bomb_obj_list.append(
                characters.Bomb(this_variable_is_useless, i['position'],
                                i['bomb_life'], i['blast_strength'],
                                i['moving_direction']))

        return bomb_obj_list

    def make_bomb_items_two(self, ret):
        bomb_obj_list = []
        for i in ret:
            this_variable_is_useless_id = i['agent']
            for j, agent in enumerate(self.curr_agents):
                if this_variable_is_useless_id == agent.agent_id:
                    this_variable_is_useless = agent
                    break
            bomb_obj_list.append(
                characters.Bomb(this_variable_is_useless, i['position'],
                                i['bomb_life'], i['blast_strength'],
                                i['moving_direction']))

        return bomb_obj_list

    def look_randomPlay(self):
        # MIGHT WANNA MAKE THIS SMART
        moves = [
            constants.Action.Stop, constants.Action.Up, constants.Action.Down,
            constants.Action.Left, constants.Action.Right
        ]
        # prune unsafe moves, COMMENT FOR STATE AGENT
        #unsafe_directions = self._directions_in_range_of_bomb(self._board, self._my_position, self.curr_bombs)
        # if unsafe_directions:
        #    if(len(unsafe_directions) != 4):
        #        for i in unsafe_directions:
        #            moves.remove(i)
        # check if position is passible

        # remove = []
        # for move in moves:
        #     check_pos = None
        #     if move == constants.Action.Up and utility.is_valid_direction(self._board, self._my_position, move):
        #         check_pos = (self._my_position[0]-1, self._my_position[1])
        #     elif move == constants.Action.Down and utility.is_valid_direction(self._board, self._my_position, move):
        #         check_pos = (self._my_position[0]+1, self._my_position[1])
        #     elif move == constants.Action.Left and utility.is_valid_direction(self._board, self._my_position, move):
        #         check_pos = (self._my_position[0], self._my_position[1]-1)
        #     elif move == constants.Action.Right and utility.is_valid_direction(self._board, self._my_position, move):
        #         check_pos = (self._my_position[0]-1, self._my_position[1]+1)
        #     if check_pos != None:
        #         if not utility.position_is_passable(self._board, check_pos, self._enemies):
        #             remove.append(move)
        # for r in remove:
        #     moves.remove(r)
        moves = self._filter_invalid_directions(self._board, self._my_position,
                                                moves, self._enemies)
        moves = [m.value for m in moves]
        moves += [constants.Action.Bomb.value]
        m = random.choice(moves)
        if (m == 5):
            self._ammo -= 1
        self.advance_game(m)

        # keys_to_pop = []
        # for key in self.bombing_agents.keys():
        #     if self._board[key[0]][key[1]] == 0 or self._board[key[0]][key[1]] == 4:
        #         keys_to_pop.append((key[0],key[1]))
        # for k in keys_to_pop:
        #     self.bombing_agents.pop(k, None)
        # update kicked bombs
        # remove the older bombs
        keys_to_pop = []
        keys_to_add = []
        bomb_life_map = self._bomb_life
        for key in self.bombing_agents.keys():
            if bomb_life_map[key[0]][key[
                    1]] == 0:  # or board[key[0]][key[1]] == 4:
                # check all directions
                # up
                r = key[0] - 1
                c = key[1]
                if (r >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # down
                r = key[0] + 1
                c = key[1]
                if (r < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # left
                r = key[0]
                c = key[1] - 1
                if (c >= 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # right
                r = key[0]
                c = key[1] + 1
                if (c < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                keys_to_pop.append((key[0], key[1]))
        for k in keys_to_pop:
            self.bombing_agents.pop(k, None)
        for k in keys_to_add:
            self.bombing_agents[k[0]] = k[1]
            # print(self.bombing_agents)
            # input("random play updating for kick")

    def non_randomPlay(self, move):
        self.advance_game(move)

        # keys_to_pop = []
        # for key in self.bombing_agents.keys():
        #     if self._board[key[0]][key[1]] == 0 or self._board[key[0]][key[1]] == 4:
        #         keys_to_pop.append((key[0],key[1]))
        # for k in keys_to_pop:
        #     self.bombing_agents.pop(k, None)
        keys_to_pop = []
        keys_to_add = []
        bomb_life_map = self._bomb_life
        for key in self.bombing_agents.keys():
            if bomb_life_map[key[0]][key[
                    1]] == 0:  # or board[key[0]][key[1]] == 4:
                # check all directions
                # up
                r = key[0] - 1
                c = key[1]
                if (r > 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # down
                r = key[0] + 1
                c = key[1]
                if (r < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # left
                r = key[0]
                c = key[1] - 1
                if (c > 0):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                # right
                r = key[0]
                c = key[1] + 1
                if (c < self._board_size):
                    if bomb_life_map[r][c] > 0 and (
                            r, c) not in self.bombing_agents.keys():
                        keys_to_add.append(((r, c), self.bombing_agents[key]))
                keys_to_pop.append((key[0], key[1]))
        for k in keys_to_pop:
            self.bombing_agents.pop(k, None)
        for k in keys_to_add:
            self.bombing_agents[k[0]] = k[1]
            # print(self.bombing_agents)
            # input("non random play updating for kick")

    # def is_winner(self): WRONG IMPLEMENTATION
    #     if (len(self._enemies) < 3):
    #         return True

    def clone(self):
        temp_deep = State(
            self._obs, True, self.bombing_agents, board_size=self._board_size)

        return temp_deep

    def _directions_in_range_of_bomb(self, board, my_position, bombs):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb.position

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right,
                        constants.Action.Left,
                        constants.Action.Up,
                        constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb.blast_strength)
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb.blast_strength)
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb.blast_strength)
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb.blast_strength)
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb.blast_strength)
        return ret

    def _filter_invalid_directions(self, board, my_position, directions,
                                   enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                        board, position, enemies
                    ):  # and not helper_func.position_is_skull(board, position):
                ret.append(direction)
        return ret


class Node:

    def __init__(self, obs=None, init=None, bombing_agents={}, board_size=11):

        self.bombing_agents = bombing_agents
        self._board_size = board_size

        self.childArray = []
        if (obs != None):
            self.state = State(
                obs, init, bombing_agents, board_size=self._board_size)
        else:
            self.state = None
        self.my_move = None

        self.win_score = 0
        self.visit_count = 0
        self.parent = None
        self.move_count = 0
        self.score = 0

    def clone(self):
        temp_deep = Node()
        temp_deep.childArray = copy.deepcopy(self.childArray)
        temp_deep.state = self.state.clone()
        temp_deep.my_move = copy.deepcopy(self.my_move)
        temp_deep.win_score = copy.deepcopy(self.win_score)
        temp_deep.visit_count = copy.deepcopy(self.visit_count)
        temp_deep.parent = self.parent
        temp_deep.move_count = copy.deepcopy(self.move_count)
        temp_deep.bombing_agents = copy.deepcopy(self.bombing_agents)
        return temp_deep

    def increment_move_count(self, parent_move_count):
        self.move_count = parent_move_count + 1

    def get_move_count(self):
        return self.move_count

    def get_random_child_node(self):
        if (len(self.childArray) == 0):
            ##print("child array 0")
            input("broken we are in tree get_random_child_node()")
        # dont pick a child with a negative score
        pickingChildren = []
        for c in self.childArray:
            #print("c.move", c.my_move, c.score)
            if c.score >= 0:
                pickingChildren.append(c)
        if len(pickingChildren) == 0:
            return self.childArray[random.randint(0, len(self.childArray) - 1)]
        random_index = random.randint(0, len(pickingChildren) - 1)
        return pickingChildren[random_index]

    def get_visit_count(self):
        return self.visit_count

    def get_win_score(self):
        return self.win_score

    def get_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state


class Tree:

    def __init__(self, obs, init, bombing_agents, board_size):
        # Node root;
        self.root = Node(obs, init, bombing_agents, board_size)

    def get_root_node(self):
        return self.root

    def set_root(self, newRoot):
        self.root = newRoot
