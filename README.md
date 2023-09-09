# Frozen Lake Environment

#### Created in [Frozen-Lake](https://github.com/RaviAgrawal-1824/Assignment-1-Frozen-Lake) environment.

## Requirements
To run this environment, you need to have the following libraries installed:
- numpy
- matplotlib
- gymnasium

## Description
**Description**: For better understanding of the **Policy** and **Value** Iteration using the Frozen lake environment for both Deterministic and Stochastic of fully observable environments.

### Non-Slippery Environment

![](https://i.imgur.com/RlJjiZM.gif) ![](https://i.imgur.com/1dpekVN.gif)

### Slippery Environment

![](https://i.imgur.com/9dF44vt.gif)

This Frozen Lake environment is solved by Dynamic Programming Method using Reinforcement learning.

## Environment Description
  ### State Space
  - There are 16 states in 4x4 Environment and 64 states in 8x8 Environment
  - Each state has 0 as reward except terminal state
  - Any state may contain lake and the aim of the agent is to reach the Goal in optimal way using policy and value iteration
  - Dictionary contain every state with taking all 4 action and transition probability to take action with reaward getting.
  ### Action Space
  There are 4 actions for every state that agent can take,
  - Left - 0
  - Down - 1
  - Right - 2
  - Up - 3

## Algorithm
Here Dynamic Programming method is used for convergence of policy.
This can also be done by two ways
### Policy Iteration
  - Evaluating Value function for all states
  - Acting greedy toward policy using action value function evaluated using value function
  - Iterated many times upto convergence of policy
### Value Iteration
  - Evaluating Value function for particular state
  - Taking Greedy of all action it can take from that state using Action value function
  - Then converging policy


# MiniGrid Empty Room Environment

#### Created in [MiniGrid-Empty-Environment](https://github.com/Farama-Foundation/MiniGrid) environment.

## Requirements
To run this environment, you need to have the following libraries installed:
- numpy
- matplotlib
- gymnasium
- minigrid

## Description
**Description**: To train agent to reach terminal state by using different algorithms in 6x6 and 8x8 minigrid environment.

![](https://i.imgur.com/3m9a615.gif) ![](https://i.imgur.com/ahGLjM7.gif)

MiniGrid Empty Environment is solved by different algorithms Monte Carlo, SARSA, SARSA Lambda, Q-Learning in Reinforcement Learning.

## Installation
Use this code for intalling some library
- pip install minigrid
- pip install numpy
- pip install matplotlib
- pip install gymnasium

## Environment Description

### State Space
- There are 16 states in MiniGrid-Empty-6x6-v0 environment and each state is represented by (x,y) where x = 1 to 4 and y = 1 to 4 
- And there are 36 states in MiniGrid-Empty-8x8-v0 environment and each cell is represented by (x,y) where x = 1 to 6 and y = 1 to 6
- State space also contain the direction of the agent at that state, the direction are as follows,
  	- 0 = Right 
  	- 1 = Down
  	- 2 = Left
  	- 3 = Up
- Obseravtion contain iamge array which can be used to identify where the agent is in environment.

### Action Space
There are three action agent can take to change state or direction,
- 0 = Turn Left
- 1 = Turn Right
- 2 = Move Forward

### Rewards
Every state has 0 reward except at terminal state.

## Algorithms
Four algorithm are used to converge the policy and take optimal actions,
- Monte-Carlo
- SARSA
- SARSA Lambda
- Q-Learning

## Results

### MiniGrid-Empty-6x6-v0

![](https://i.imgur.com/BDnIvQI.png) ![](https://i.imgur.com/4rQas2b.png)

### MiniGrid-Empty-8x8-v0

![](https://i.imgur.com/378ocDl.png) ![](https://i.imgur.com/aDHMdLL.png)
