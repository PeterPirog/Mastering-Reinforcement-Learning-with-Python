# https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api
# https://docs.ray.io/en/latest/rllib/package_ref/trainer.html#trainer-base-class-ray-rllib-agents-trainer-trainer
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
from pprint import pprint

import gym
import ray
import ray.rllib.agents.ppo as ppo

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 63
config["framework"] = "tf2"
config["env"] = "CartPole-v0"
#config["lr"] = tune.grid_search([0.01, 0.001, 0.0001])
config["model"]["vf_share_layers"]=True

#LSTM
#config["model"]["use_lstm"] = True
#config["model"]["max_seq_len"] = 3
#config["model"]["lstm_cell_size"] =256
#config["model"]["lstm_use_prev_action"] =False
#config["model"]["lstm_use_prev_reward"] = False


pprint(config)


checkpoint_path="/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/custom_scripts/cartpole/PPOTrainer_2022-07-08_03-56-42/PPOTrainer_CartPole-v0_fa384_00000_0_2022-07-08_03-56-42/checkpoint_000032/checkpoint-32"

from tensorflow.keras.utils import plot_model
trainer = ppo.PPOTrainer(config=config)
trainer.restore(checkpoint_path)
model=trainer.get_policy().model.base_model
# typical tensorflow visualization
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,rankdir="TB",expand_nested=False,show_layer_activations=True)



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

#print(dir(trainer))
#agent.export_model("my_weights.h5")

# Get weights of the default local policy
#print(trainer.get_policy().get_weights())

#policy = trainer.get_policy()
#policy.model.base_model.summary()

#model=policy.model.base_model


