import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# env = gym.make('MiniGrid-Empty-6x6-v0')
env = gym.make('MiniGrid-Empty-8x8-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')

action_value_function = {}
number_of_episode = 100
env.reset()
# for t in range(number_of_episode):
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
# # print(sorted_action_value_function)
epsilon = 1.0
episode = []
rewards = []
steps_to_goal = []
number_of_episode = 200
for i in range(number_of_episode + 20):
    env.reset()
    gamma = 0.99
    alpha = 0.1
    done = False
    truncate = False
    policy = []
    state = env.agent_pos
    direction = env.agent_dir
    state_space = (state[0], state[1], direction)
    step = 1
    while not done and not truncate:
        if np.random.uniform(0,1) < epsilon and epsilon > 0:
            action = np.random.randint(0,3)
        else:
            action = np.argmax(action_value_function[state_space])
        if state_space not in action_value_function:
            action_value_function[state_space] = np.zeros(3)
        observation, reward, done, truncate, info = env.step(action)
        # a = state_space
        policy.append(action)
        next_state = env.agent_pos
        next_direction = env.agent_dir
        next_state_space = (next_state[0], next_state[1], next_direction)
        if next_state_space not in action_value_function:
            action_value_function[next_state_space] = np.zeros(3)
        # action_value_function[state_space][action] += alpha * (reward + gamma * (np.max(action_value_function[next_state_space])) - action_value_function[state_space][action])
        max_action_value_function = np.max([action_value_function[next_state_space][a] for a in range(3)])
        action_value_function[state_space][action] = (1-alpha) * action_value_function[state_space][action] + alpha * (reward + gamma * max_action_value_function)
        state_space = next_state_space
        step+=1
    epsilon = (-1*(i+1)/number_of_episode) + 1
    episode.append(i+1)
    rewards.append(reward)
    steps_to_goal.append(step)
    # print("No. of episodes Completed: ", i+1)
my_keys = list(action_value_function.keys())
my_keys.sort()
sorted_action_value_function  = {i:action_value_function[i] for i in my_keys}
# print(sorted_action_value_function)
print(policy)

env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')
for i in range(1):
    env.reset()
    epd_rew = 0
    for i in range(len(policy)):
        n_obs,rew,done,trunc,info= env.step(policy[i])
        # time.sleep(0.25)
        epd_rew += rew
        env.render()
env.close()

# plt.title("MiniGrid-Empty-6x6-v0 using Q-Learning")
plt.title("MiniGrid-Empty-8x8-v0 using Q-Learning")
plt.plot(episode,rewards)
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.show()

# plt.title("MiniGrid-Empty-6x6-v0 using Q-Learning")
plt.title("MiniGrid-Empty-8x8-v0 using Q-Learning")
plt.plot(episode,steps_to_goal)
plt.xlabel("Number of Episodes")
plt.ylabel("Steps to reach Goal")
plt.show()
