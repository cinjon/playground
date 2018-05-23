# Pommerman

### Game Overview:
* Pommerman is a play on Bomberman. There are three different variants, each of which follow the same basic idea but have their own distinct flavors:
  * FFA: Free For All where four agents enter and one leaves. It tests planning, tactics, and cunning. The board is fully observable.
  * Team (The NIPS '18 Competition environment): 2v2 where two teams of agents enter and one team wins. It tests planning, and tactics, and cooperation. The board is partially observable.
  * Team Radio: Like team in that a it's a 2v2 game. Differences are that the agents each have a radio that they can use to convey 2 words from a dictionary of size 8 each step.

### Directory Overview:

* agents: Baseline agents will reside here in addition to being available in the Docker directory. 
* characters.py: Here lies the actors in the game. This includes Agent, Bomb, and Flame.
* configs.py: This configs module contains the setup. Feel free to edit this in your local directory for easy game loading.
* envs (module):
  * utility.py has shared Enums, constants, and common functions to the different environments.
  * v0.py: This environment is the base one that we use. 
  * v1.py: This is a modification of v0.py that collapses the walls in order to end the game more quickly.
  * v2.py: This is a modification of v0.py that adds in communication. It works by having the agents send a message as part of their actions and then includes that message in the next turn of observations.

### Agent Observations:

* Each agent sees:
  * Board: The 11x11 board is a numpy array where each value corresponds to one of the representations below. 
    * Passage = 0
    * Rigid Wall = 1
    * Wooden Wall = 2
    * Bomb = 3
    * Flames = 4
    * Fog = 5: This is only applicable in the partially observed (2v2 Team Radio) setting.
    * Extra Bomb Power-Up = 6: adds ammo.
    * Increase Range Power-Up = 7: increases the blast_strength
    * Can Kick Power-Up = 8: can kick bombs by touching them.
    * AgentDummy = 9
    * Agent0 = 10
    * Agent1 = 11
    * Agent2 = 12
    * Agent3 = 13
  * Position: A tuple of Ints of (X position, Y position)
  * Ammo: An Int representing the amount of ammo this agent has. 
  * Blast Strength: An Int representing the blast strength of this agent's bombs.
  * Can Kick: Whether this agent can kick bombs. This ability is gained by stepping on the can kick power-up.
  * Teammate: One Int in [9, 13].  Which agent is this agent's teammate. In the FFA game, this is the AgentDummy.
  * Enemies: A list of three Ints, each in [9, 13]. Which agents are this agent's enemies. There are three here to be amenable to all variants of the game. When there are only two enemies like in the team competitions, the last Int will be the AgentDummy to reflect the fact that there are only two enemies.
  * Bomb Blast Strength: An 11x11 numpy int array representing the bombs' blast strengths in the agent's view. Everything outside of its view will be fogged out.
  * Bomb Life: An 11x11 numpy int array representing the bombs' life in the agent's view. Everything outside of its view will be fogged out.
  * Message: (Team Radio only) A list of two Ints, each in [0, 8]. The message being relayed from the teammate. Both ints are zero when a teammate is dead or it's the first step. Otherwise they are in [1, 8].

### Agent Actions:

* Each agent's actions are:
  * Movement: a single integer in [0, 5] representing which of the six actions that agent would like to take of the following
    * Stop (0): This action is a pass.
    * Up (1): Move up on the board.
    * Down (2): Move down on the board.
    * Left (3): Move left on the board.
    * Right (4): Move right on the board.
    * Bomb (5): Lay a bomb.
  * Message: (Team Radio only) A list of two Ints in [1, 8]. These represent the message. 
        
### Game Rules:

* Every battle starts on a randomly drawn symmetric 11x11 grid (`board'). There are four agents, one in each of the corners. An agent's teammate (if applicable) will be on the kitty corner.
* The board is randomly constructed before each battle and, besides the agents, contains wood walls and rigid walls. We guarantee that the agents will have an accessible path to each other, possibly through wooden walls.
* Rigid walls are indestructible and impassable.
* Wooden walls can be destroyed by bombs (see below). Until they are destroyed, they are impassable. After they are destroyed, they become either a passage or a power-up.
* In any given turn, an agent can choose from one of six actions:
  * Stop (0): This action is a pass.
  * Up (1): Move up on the board.
  * Left (2): Move left on the board.
  * Down (3): Move down on the board.
  * Right (4): Move right on the board.
  * Bomb (5): Lay a bomb.
  * If there is communication, each agent additionally emits a message on each turn consisting of two words from a dictionary of size eight. These words will be given to its teammate in the next step as part of the observation.
* The agent starts with one bomb ("ammo"). Every time it lays a bomb, its ammo decreases by one. After that bomb explodes, its ammo will increase by one.
* The agent also has a blast strength (starts at three). Every bomb it lays is imbued with the current blast strength, which is how far in the vertical and horizontal directions that bomb will effect.
* A bomb has a life of 10 time steps. After its life expires, it explodes and any wooden walls, agents, power-ups or other bombs in its range (given by the blast strength) are destroyed.
* Power-Ups: Half of the wooden walls have power-ups hidden underneath them that are revealed when they are destroyed. These are:
  * Extra Bomb: Picking this up increases the agent's ammo by one.
  * Increase Range: Picking this up increases the agent's blast strength by one.
  * Can Kick: Picking this up allows an agent to kick bombs. It does this by running into them. They then travel in the direction that the agent was moving at a speed of one unit per time step until they are impeded either by a player, a bomb, or a wall.
* The game ends when both players on one team have been destroyed. The winning team is the one who has remaining members.
* Ties can happen when the game does not end before the max steps or if both teams' last agents are destroyed on the same turn. If this happens in a competition, we will rerun the game once. If it happens again after that, then we will rerun it with collapsing walls until there is a winner. This is a variant where, after a large number of steps, the game board becomes smaller according to a specified cadence. See v1.py for a working example in the code.
* If an agent does not respond in an appropriate time limit (100ms), then we will automatically issue them the Stop action and have them send out the message (0, 0).        
* The game setup does not allow for the agents to share a centralized controller. If, however, some clever participant figured out a way to force this, they will be subsequently disqualified.
