# Rules and Submission

1) Each submission should have a Docker file per agent. For FFA and Team Random, there is one agent; For Team Radio, there will be two agents. Instructions and an example for building Docker containers from trained agents can be found in our repository.

2) The positions for each agent will be randomized modulo that each agent's position will be opposite from its teammate's position.

3) The agents should follow the prescribed convention specified in our example code and expose an "act" endpoint that accepts a dictionary of observations. Because we are using Docker containers and http requests, we do not have any requirements for programming language or framework. There will be ample opportunity to test this on our servers beforehand.

4) If an agent has a bug in its software that causes its container to crash, that will count as a loss for that agent's team.

5) The expected response from the agent will be a single integer in [0, 5] representing which of the six actions that agent would like to take, as well as two more integers in [1, 8] representing the message if applicable.

6) If an agent does not respond in an appropriate time limit (100ms), then we will automatically issue them the Stop action and have them send out the message (0, 0) if applicable.

7) The game setup as described does not allow for the agents to share a centralized controller. If, however, some clever participant figured out a way to force this, they will be subsequently disqualified.

8) Agents submitted by organizers can participate in the competitions but are not eligible for prizes. They will be excluded from consideration in the final standings.

9) Competitions will run according to a double elimination style with two brackets. Each battle will be best of three, with the winner moving on and the loser suffering a defeat. Any draws will be replayed. At the end, we will have a clear top four.
