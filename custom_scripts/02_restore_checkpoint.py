# https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api
# https://docs.ray.io/en/latest/rllib/package_ref/trainer.html#trainer-base-class-ray-rllib-agents-trainer-trainer
import ray
import tensorflow as tf
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 63
config["env"] = "CartPole-v0"
#config["lr"] = tune.grid_search([0.01, 0.001, 0.0001])

stop_criteria = {"episode_reward_mean": 200}
log_dir = "/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/custom_scripts/cartpole"
checkpoint_path="/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/custom_scripts/cartpole/PPOTrainer_2022-07-07_18-46-18/PPOTrainer_CartPole-v0_16ad8_00000_0_2022-07-07_18-46-19/checkpoint_000020/checkpoint-20"

trainer = ppo.PPOTrainer(config=config)
trainer.restore(checkpoint_path)

env = gym.make("CartPole-v0")
obs = env.reset()
episode_reward = 0

while True:
    action = trainer.compute_action(obs)
    obs, reward, done, _ = env.step(action)

    episode_reward += reward
    print(f"{obs},{episode_reward},{done}")
    if done:
        break
env.close()

print(dir(trainer))
#agent.export_model("my_weights.h5")

# Get weights of the default local policy
print(trainer.get_policy().get_weights())

#policy = trainer.get_policy()
#policy.model.base_model.summary()

#model=policy.model.base_model
model=trainer.get_policy().model.base_model
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

