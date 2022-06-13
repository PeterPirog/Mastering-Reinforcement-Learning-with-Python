import gym
#env = gym.make('CartPole-v0')
env = gym.make('BipedalWalkerHardcore-v3')
env.reset()
for _ in range(1000):
    #env.render()
    print(env.step(env.action_space.sample())) # take a random action
env.close()