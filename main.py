import gym
import numpy as np

from agent import Agent
from model import Model

# Create an agent and prepare the train data
agent = Agent()
data = agent.prepare_data()

# Create and train the model
model = Model([4], 2)
model.train_model(data)


env = gym.make('CartPole-v1')
env.reset()

observation = []
for game_step in range(500):
    # Uncomment the line below to see how the computer plays
    # env.render()
    if len(observation) == 0:
        action = env.action_space.sample()
    else:
        observation = observation.reshape(-1, len(observation))
        # predict the next step
        prediction = model.predict_model(observation)
        action = np.argmax(prediction[0])

    observation, reward, done, info = env.step(action)

    if done:
        break
env.reset()
