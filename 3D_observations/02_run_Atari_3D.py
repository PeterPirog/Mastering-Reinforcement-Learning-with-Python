
import gym

env = gym.make("SpaceInvaders-v0") #,render_mode='human'
obs = env.reset()
while True:
    #action = agent.compute_action(obs)
    obs, reward, done, _ = env.step(0)
    #env.render()
    if done:
        break
env.close()