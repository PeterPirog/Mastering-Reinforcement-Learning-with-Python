# https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from pprint import pprint

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 63
config["env"] = "CartPole-v0"
config["framework"] = "tf2"
config["model"]["vf_share_layers"] = True
# config["lr"] = tune.grid_search([0.01, 0.001, 0.0001])

#LSTM
#config["model"]["use_lstm"] = True
#config["model"]["max_seq_len"] = 3
#config["model"]["lstm_cell_size"] =256
#config["model"]["lstm_use_prev_action"] =False
#config["model"]["lstm_use_prev_reward"] = False

# ATTENTION
#config["model"]["use_attention"] = True



pprint(config)

stop_criteria = {"episode_reward_mean": 200}
log_dir = "/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/custom_scripts/cartpole"

# tune.run() allows setting a custom log directory (other than ``~/ray-results``)
# and automatically saving the trained agent
analysis = ray.tune.run(
    ppo.PPOTrainer,
    config=config,
    local_dir=log_dir,
    stop=stop_criteria,
    checkpoint_at_end=True)

# list of lists: one list per checkpoint; each checkpoint list contains
# 1st the path, 2nd the metric value
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")
print(f"checkpoints: {checkpoints}")

# or simply get the last checkpoint (with highest "training_iteration")
last_checkpoint = analysis.get_last_checkpoint()
print(f"last_checkpoint1: {last_checkpoint}")
# if there are multiple trials, select a specific trial or automatically
# choose the best one according to a given metric
last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="max"
)
print(f"last_checkpoint2: {last_checkpoint}")
