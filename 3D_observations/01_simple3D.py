from ray import tune
import gym
from ray.rllib.agents.ppo.ppo import (
    DEFAULT_CONFIG,
    PPOTrainer as trainer)

config={
                       "env": "SpaceInvaders-v0",
                       "framework": "tf2",
                       "num_gpus": 0,
                       "num_workers": 53,
                       "evaluation_interval": 2,
                       # "evaluation_num_episodes":20,
                       "evaluation_duration": 20,
                   }
# https://docs.ray.io/en/latest/tune/api_docs/execution.html
results = tune.run("PPO",
                   config=config,
                   stop={"training_iteration": 1},
                   local_dir="cartpole",
                   checkpoint_freq=2)

# Get the last checkpoint from the above training run.
checkpoint = results.get_last_checkpoint()

algo = trainer(config=config)
algo.restore(checkpoint)

# Create the env to do inference in.
env = gym.make("SpaceInvaders-v0")
obs = env.reset()

num_episodes = 0
episode_reward = 0.0

while num_episodes < 100:
    # Compute an action (`a`).
    a = algo.compute_single_action(
        observation=obs,
        #explore=args.explore_during_inference,
        policy_id="default_policy",  # <- default value
    )
    # Send the computed action `a` to the env.
    obs, reward, done, _ = env.step(a)
    episode_reward += reward
    # Is the episode `done`? -> Reset.
    if done:
        print(f"Episode done: Total reward = {episode_reward}")
        obs = env.reset()
        num_episodes += 1
        episode_reward = 0.0

#ray.shutdown()

# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/3D_observations/cartpole/PPO
# # kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
