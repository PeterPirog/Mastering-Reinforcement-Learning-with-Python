from ray.rllib.agents.ppo.ppo import PPOTrainer
import gym

config = {
    "env": "CartPole-v0",
    "framework": "tf2",
    "evaluation_interval": 2,
    "evaluation_duration": 20,
}

agent = PPOTrainer(config=config)
agent.restore("/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/3D_observations/cartpole/PPO/PPO_CartPole-v0_23a86_00000_0_2022-06-30_16-24-39/checkpoint_000002/checkpoint-2")
agent.load_checkpoint()

env = gym.make("CartPole-v0")
obs = env.reset()
while True:
    action = agent.compute_action(obs)
    obs, reward, done, _ = env.step(action)
    #env.render()
    if done:
        break
env.close()
