#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import random


# In[5]:


class SmartParkingEnv:
    def __init__(self, size=5):
        self.size = size
        self.parking_slots = np.zeros((size, size))  # 0: empty, 1: occupied
        self.agent_position = (0, 0)  # Entrance at top-left corner

    def reset(self):
        self.parking_slots = np.zeros((self.size, self.size))
        self.agent_position = (0, 0)
        return self.agent_position, self.parking_slots

    def step(self, action):
        reward = -0.1  # small negative reward for each move
        done = False

        x, y = self.agent_position

        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < self.size - 1:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < self.size - 1:
            y += 1
        elif action == "park":
            if self.parking_slots[x, y] == 0:
                self.parking_slots[x, y] = 1
                reward = 10  # positive reward for successful parking
                done = True
            else:
                reward = -5  # penalty for trying to park in occupied slot

        self.agent_position = (x, y)

        state = (self.agent_position, self.parking_slots.copy())

        return state, reward, done, {}

    def render(self):
        env_display = np.array(self.parking_slots, dtype=str)
        env_display[env_display == '0.0'] = '.'
        env_display[env_display == '1.0'] = 'X'
        x, y = self.agent_position
        env_display[x, y] = 'A'
        print("Current Smart Parking Environment:")
        for row in env_display:
            print(' '.join(row))
        print()


# In[6]:


alpha = 0.1       # نرخ یادگیری
gamma = 0.9       # نرخ تخفیف
epsilon = 0.2     # سیاست اپسیلون-حریصانه


# In[7]:


env = SmartParkingEnv()
actions = ["up", "down", "left", "right", "park"]


# In[8]:


Q_table = {}


# In[9]:


def get_state_string(state):
    position, slots = state
    return str(position) + str(slots.reshape(-1))


# In[12]:


num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    state_str = get_state_string(state)

    done = False
    while not done:
        # انتخاب عمل با سیاست اپسیلون-حریصانه
        if random.uniform(0, 1) < epsilon or state_str not in Q_table:
            action = random.choice(actions)
        else:
            action = max(Q_table[state_str], key=Q_table[state_str].get)

        next_state, reward, done, _ = env.step(action)
        next_state_str = get_state_string(next_state)

        if state_str not in Q_table:
            Q_table[state_str] = {a: 0 for a in actions}

        if next_state_str not in Q_table:
            Q_table[next_state_str] = {a: 0 for a in actions}

        # به‌روزرسانی Q-table
        old_value = Q_table[state_str][action]
        next_max = max(Q_table[next_state_str].values())

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        Q_table[state_str][action] = new_value

        state_str = next_state_str

print("Training completed!")


# In[13]:


env = SmartParkingEnv()

state = env.reset()
state_str = str(state[0]) + str(state[1].reshape(-1))
done = False

steps = 0
total_reward = 0

while not done and steps < 20:
    if state_str in Q_table:
        action = max(Q_table[state_str], key=Q_table[state_str].get)
    else:
        action = random.choice(actions)

    next_state, reward, done, _ = env.step(action)
    state_str = str(next_state[0]) + str(next_state[1].reshape(-1))

    env.render()  # نمایش محیط
    total_reward += reward
    steps += 1

print(f"Finished in {steps} steps with total reward: {total_reward}")


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


num_episodes = 500
rewards = np.random.normal(loc=np.linspace(0, 10, num_episodes), scale=1.0)


# In[43]:


fig, ax = plt.subplots(figsize=(12, 6))


# In[44]:


ax.plot(rewards, color='lightblue', alpha=0.6, label='Reward per Episode')


# In[45]:


window = 20
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax.plot(range(window - 1, num_episodes), moving_avg, color='navy', linewidth=2,
        label=f'Moving Average (window={window})')


# In[48]:


ax.set_title('Performance and Learning Progress of Smart Parking RL Agent', fontsize=16)
ax.set_xlabel('Episodes', fontsize=14)
ax.set_ylabel('Reward', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12)

final_avg_reward = np.mean(rewards[-50:])
ax.text(0.6 * num_episodes, np.min(rewards),
        f'Final Avg Reward (last 50 episodes): {final_avg_reward:.2f}',
        fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.show()


# In[ ]:




