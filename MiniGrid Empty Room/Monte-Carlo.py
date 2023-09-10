import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode='human')
# env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
steps = []
returns = []
epds = []
gamma = 0.9
alpha = 0.4
# epsilon = 1.0
number_of_episodes = 150
action_value_function = {}
max_steps = 600
for i in range(number_of_episodes + 20):
    env.reset()
    # state = env.reset()
    episode = []
    policy = []
    epsilon = (-1*i/number_of_episodes) + 1
    done = False
    truncate = False
    step = 1
    while not done and not truncate:
        step += 1
        state = env.unwrapped.agent_pos
        direction = env.unwrapped.agent_dir
        state_space = (state[0], state[1], direction)
        if state_space not in action_value_function:
            action_value_function[state_space] = np.zeros(3,dtype=float)
        if np.random.uniform(0,1) < epsilon and epsilon > 0:
            action = np.random.randint(0,3)
        else:
            action = np.argmax(action_value_function[state_space])
        policy.append(action)
        observation, reward, done, truncate, info = env.step(action)
        if reward < 0:
            reward = 0
        episode.append((state_space, action, reward))
        next_state = env.unwrapped.agent_pos
        next_direction = env.unwrapped.agent_dir
        next_state_space = (state[0], state[1], direction)
        state_space = next_state_space
    returns.append(reward)
    steps.append(step)
    epds.append(i+1)
    # print("Episodes: ", episode)
    # my_keys = list(action_value_function.keys())
    # my_keys.sort()
    # sorted_action_value_function  = {i:action_value_function[i] for i in my_keys}
    # print("Action Value Function: ", sorted_action_value_function)
    G = 0
    for k in reversed(range(len(episode))):
        state_space, action, reward = episode[k]
        G = reward + G * gamma
        # if state_space not in action_value_function:
        #     action_value_function[state_space] = [0,0,0]
        action_value_function[state_space][action] += alpha * (G - action_value_function[state_space][action])
    # epsilon = (-1*(i+1)/number_of_episodes) + 1
    # print("Epsilon: ", epsilon)
    print("No. of episodes Completed: ", i+1)
print("Action Value Function:", action_value_function)
print("Policy: ", policy)

# plt.title("MiniGrid-Empty-8x8-v0 using Monte Carlo Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using Monte Carlo Algorithm")
plt.plot(epds,returns)
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.show()

# plt.title("MiniGrid-Empty-8x8-v0 using Monte Carlo Algorithm")
plt.title("MiniGrid-Empty-6x6-v0 using Monte Carlo Algorithm")
plt.plot(epds,steps)
plt.xlabel("Number of Episodes")
plt.ylabel("Steps to reach Goal")
plt.show()
# env = gym.make('MiniGrid-Empty-8x8-v0')
# env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-8x8-v0',render_mode = 'human')
env = gym.make('MiniGrid-Empty-6x6-v0',render_mode = 'human')

for i in range(1):
    env.reset()
    epd_rew = 0
    for i in range(len(policy)):
        n_obs,rew,done,trunc,info= env.step(policy[i])
        # time.sleep(0.25)
        epd_rew += rew
        env.render()
env.close()