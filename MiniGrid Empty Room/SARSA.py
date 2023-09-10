import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')

returns = []
epds = []
steps_to_goal = []
env.reset()
action_value_function = {}
# number_of_episodes = 10
# for t in range(number_of_episodes):
#     # env.reset()
#     done = False
#     while not done : 
#         state = env.agent_pos
#         direction = env.agent_dir
#         action = np.random.randint(0,3)
#         observation, reward, done, truncation, info = env.step(action)
#         state_space = (state[0],state[1],direction)
#         action_value_function[state_space] = np.zeros(3,dtype=float)
# my_keys = list(action_value_function.keys())
# my_keys.sort()
# sorted_action_value_function  = {i:action_value_function[i] for i in my_keys}
# print(sorted_action_value_function)

# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
gamma = 0.99
alpha = 0.5
epsilon = 1.0
number_of_episodes = 70
for i in range(number_of_episodes+20):
    episode = {}
    env.reset()
    # print(obs)
    # policy = {}
    # action_value_function = {(1,1,0):(0,0,0)}
    if np.random.rand() < epsilon and epsilon > 0:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(action_value_function[(1,1,0)])
    done = False
    # print("First Action: ", action)
    Policy = []
    step = 1
    while not done:
        state = env.agent_pos
        direction = env.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in action_value_function:
            action_value_function[state_space] = np.zeros(3)
        observation, reward, done, truncation, info = env.step(action)
        a = state_space
        # print("State1: ",a)
        b = action
        Policy.append(b)
        # print("Action1: ", b)
        state = env.agent_pos
        direction = env.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in action_value_function:
            action_value_function[state_space] = np.zeros(3)
        # print("State2: ", state_space)
        # episode.append((state, state_space, reward, done))
        if np.random.rand() < epsilon and epsilon > 0:
            action = np.random.randint(0,3)
        else:
            action = np.argmax(action_value_function[state_space])
        # print("Action2: ", action)  
        action_value_function[a][b] += alpha*((reward + gamma * action_value_function[state_space][action]) - action_value_function[a][b])
        step+=1
    steps_to_goal.append(step)
    returns.append(reward)
    epds.append(i+1)
    epsilon = (-1*i/number_of_episodes) + 1
    # print(epsilon)
    # print(action_value_function)
    # print("Policy: ", Policy)
    print("No. of episodes Completed: ", i+1)
print(Policy)
env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')
for i in range(10):
    env.reset()
    epd_rew = 0
    for i in range(len(Policy)):
        n_obs,rew,done,trunc,info= env.step(Policy[i])
        time.sleep(0.25)
        epd_rew += rew
        env.render()
env.close()
print("Episode Reward: ",epd_rew)
print("Optimal Policy: ",Policy)
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Algorithm")
plt.title("MiniGrid-Empty-8x8-v0 using SARSA Algorithm")
plt.plot(epds,returns)
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.show()

plt.plot(epds,steps_to_goal)
plt.title("MiniGrid-Empty-6x6-v0 using SARSA Algorithm")
plt.title("MiniGrid-Empty-8x8-v0 using SARSA Algorithm")
plt.xlabel("Number of Episodes")
plt.ylabel("Steps to reach Goal")
plt.show()