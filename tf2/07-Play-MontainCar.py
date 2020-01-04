# https://github.com/claymcleod/dqn
# https://github.com/llSourcell/deep_q_learning


from collections import deque
import random
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# from gym import envs
# all_envs = envs.registry.all()
# env_ids = [env_spec.id for env_spec in all_envs]
# print(env_ids)

env = gym.make('CartPole-v0')  # Choose game (any in the gym should work)

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential([
	Dense(20, input_shape=(2,) + env.observation_space.shape, activation='relu'),
	# Flatten input so as to have no problems with processing
	Flatten(),
	Dense(18, activation='relu'),
	Dense(10, activation='relu'),
	# Same number of outputs as possible actions
	Dense(env.action_space.n, activation='linear')
])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

D = deque()         # Register where the actions will be stored

observetime = 500   # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7       # Probability of doing a random move

gamma = 0.9         # Discounted future reward. How much we care about steps further in time
mb_size = 50        # Learning minibatch size

# FIRST STEP: Knowing what each action does (Observing)

observation = env.reset()                     # Game begins
# (Formatting issues) Making the observation the first element of a batch of inputs
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        Q = model.predict(state)          # Q-values predictions
        # Move with highest Q-value is the chosen one
        action = np.argmax(Q)
    # See state of the game, reward... after performing the action
    observation_new, reward, done, info = env.step(action)
    # (Formatting issues)
    obs_new = np.expand_dims(observation_new, axis=0)
    # Update the input with the new state of the game
    state_new = np.append(np.expand_dims(
        obs_new, axis=0), state[:, :1, :], axis=1)
    # 'Remember' action and consequence
    D.append((state, action, reward, state_new, done))
    state = state_new         # Update state
    if done:
        env.reset()           # Restart game if it's finished
        # (Formatting issues) Making the observation the first element of a batch of inputs
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
print('Observing Finished')

# SECOND STEP: Learning from the observations (Experience replay)

# Sample some moves
minibatch = random.sample(D, mb_size)

inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, env.action_space.n))

for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]

# Build Bellman equation for the Q function
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)

    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

# Train network to output the Q function
    model.train_on_batch(inputs, targets)
print('Learning Finished')

# THIRD STEP: Play!

observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()                    # Uncomment to see game running
    Q = model.predict(state)
    action = np.argmax(Q)
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))
