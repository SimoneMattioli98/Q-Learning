import math
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

env = gym.make("CartPole-v1")

def get_win_size(os_low, os_high, buckets):
    win_size = (abs(os_low) + abs(os_high)) / buckets
    return np.array(win_size)

low = env.observation_space.low
high = env.observation_space.high
print(f"Observation space low: {low}")
print(f"Observation space high: {high}")

#limit low and high for inf values (will never reach such values)
low[1] = -20.00
high[1] = 20.00

low[3] = -20.00
high[3] = 20.00


print(f"Observation space low: {low}")
print(f"Observation space high: {high}")

n_buckets = [1, 1, 100, 100]

np_win_size = get_win_size(low, high, n_buckets)
print(f"Win_size: {np_win_size}")


q_table = np.random.uniform(low = 0, high = 1, size = n_buckets + [env.action_space.n])
print(f"Q_table shape: {q_table.shape}")

def discretizer(obs, win_size):
    discrete_obs = obs / win_size
    return tuple(discrete_obs.astype(int))

#Q_learning settings
EPISODES = 70_000
LEARNING_RATE = 0.1
SHOW_EVERY = 500
DISCOUNT = 0.95
SAVE_Q_TABLE_EP = 100
SAVE_GIF_EPISODE = 50_000


#Exploration settings
epsilon = 1
MINIMUM_EPS = 0.0
reward_threshold = 0
REWARD_TARGET = 200
episode_reward = 0
high_reward = 0
prev_reward = 0
REWARD_INCREMENT = 1
START_DECAYING_EPISODE = 10_000
epsilon_decay_value = 0.9995

#other variables 
total_reward = 0
mean_reward = 0
prev_reward = 0
objective_reached = 0
frames = []

#analysis
ep_rewards = []
analysis_ep_reward = {'ep': [], 'avg': [], 'min': [], 'max': []}

plt.figure(figsize=(12, 9))

for episode in range(EPISODES + 1):
    done = False
    discrete_obs = discretizer(env.reset(), np_win_size)
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        if np.random.rand() > epsilon:
            action = np.argmax(q_table[discrete_obs])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_obs, reward, done, _ = env.step(action)
        new_discrete_obs = discretizer(new_obs, np_win_size)
        episode_reward += reward
        
        # if render:
        #     env.render()

        if episode == SAVE_GIF_EPISODE:
            frames.append(env.render(mode="rgb_array"))
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_obs])
            
            current_q = q_table[discrete_obs + (action,)]
            
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            q_table[discrete_obs + (action,)] = new_q

        discrete_obs = new_discrete_obs

    if epsilon > 0.05:
        if epsilon > MINIMUM_EPS and episode_reward >= prev_reward and episode > START_DECAYING_EPISODE:    
            epsilon = math.pow(epsilon_decay_value, episode - START_DECAYING_EPISODE)
           
    

    if episode_reward >= REWARD_TARGET :
        objective_reached = objective_reached + 1


    prev_reward = episode_reward
    
    ep_rewards.append(episode_reward)


    if not episode % SHOW_EVERY: 
        avg_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        analysis_ep_reward['ep'].append(episode)
        analysis_ep_reward['avg'].append(avg_reward)
        analysis_ep_reward['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        analysis_ep_reward['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        # plt.subplot(111)
        # plt.axvline(x=START_DECAYING_EPISODE, label="start decay ep", c = "red")
        # plt.axhline(y=REWARD_TARGET, label="target", c = "purple")
        # plt.plot(analysis_ep_reward['ep'], analysis_ep_reward['avg'], label = 'avg')
        # plt.plot(analysis_ep_reward['ep'], analysis_ep_reward['min'], label = 'min')
        # plt.plot(analysis_ep_reward['ep'], analysis_ep_reward['max'], label = 'max')
        # plt.xlim(0, EPISODES+1)
        # plt.ylim(0,600)
        # plt.legend(loc = 4)
        # plt.savefig(f"charts/try{episode}.png")
        # plt.clf()

        print(f"EP: {episode} avg: {avg_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])} Objective reached: {objective_reached}")
        total_reward = 0
        objective_reached = 0

    if frames:
        save_frames_as_gif(frames)
    
    
env.close()




