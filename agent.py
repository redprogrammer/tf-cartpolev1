import gym


class Agent:
    env = None
    required_min_score = 60
    # Number of games to be trained
    trained_games = 10000
    # Number of game steps to be trained
    game_steps = 500

    def __init__(self, num_of_game_played=10000, num_of_game_step=500):
        self.trained_games = num_of_game_played
        self.game_steps = num_of_game_step
        self.env = gym.make('CartPole-v1')
        self.env.reset()

    def prepare_data(self):
        training_data = [[], []]
        for game_index in range(self.trained_games):
            current_score = 0
            game_memory = []
            prev_obs = []
            for step_index in range(self.game_steps):
                # Get random sample value for the next step,
                # for cartpole game this value must be 0 or 1
                # 0 pushes cart left and 1 pushes cart right
                action = self.env.action_space.sample()
                # Test the action value for this game step
                # observation returns an array like this [Cart Position, Cart Velocity, Pole Angle Pole, Velocity At Tip]
                # reward returns the value 1
                # done value returns boolean, game is finished or not
                observation, reward, done, info = self.env.step(action)

                if len(prev_obs) > 0:
                    if action == 1:
                        game_memory.append([prev_obs, [0, 1]])
                    elif action == 0:
                        game_memory.append([prev_obs, [1, 0]])

                prev_obs = observation
                current_score += reward
                if done:
                    break

            # if the score is enough, save it for the training
            if current_score >= self.required_min_score:
                for data in game_memory:
                    training_data[0].append(data[0])
                    training_data[1].append(data[1])

            self.env.reset()

        return training_data
