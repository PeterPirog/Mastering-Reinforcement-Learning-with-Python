import ray
from ray import tune
#from inventory_env import InventoryEnv
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler

import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete, Dict
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()


class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class BanditEnv(gym.Env):
    def __init__(self, probabilities, number_of_draws=50):
        self.probabilities = probabilities
        self.number_of_draws = number_of_draws
        self.max_avail_actions = len(self.probabilities)
        self.action_space = Discrete(self.max_avail_actions)
        self.action_mask = None
        self.current_obs = None

        self.observation_space = Dict(
            {
                "action_mask": Box(0, 1, (self.max_avail_actions,), dtype=int),
                "observations": Box(0, 1, (2, 2, 2,), dtype=np.float32),
            }
        )

        self.reset()

    def reset(self):
        self.current_draw = 0
        self.done = False

        # line below is constant for this example only
        self.__update_current_obs()
        self.__update_mask__()  # update mask for this step

        self.observation = {
            "action_mask": self.action_mask,
            "observations": self.current_obs,
        }

        return self.observation

    def step(self, action):
        val = np.random.uniform(low=0.0, high=1.0)
        if val <= self.probabilities[action]:
            reward = 1.0
        else:
            reward = 0.0

        info = {}
        self.current_draw += 1
        if self.current_draw == self.number_of_draws:
            self.done = True

        self.__update_current_obs()
        self.__update_mask__()  # update mask for this step

        self.observation = {
            "action_mask": self.action_mask,
            "observations": self.current_obs
        }

        return self.observation, reward, self.done, info

    def __update_mask__(self):
        self.current_obs
        self.action_mask = np.array([1.0, 1.0, 1.0], dtype=int)

    def __update_current_obs(self):
        self.current_obs = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]], dtype=np.float32)



if __name__ == "__main__":
    def env_creator(env_config={}):
        return BanditEnv(probabilities=env_config["probabilities"],
                         number_of_draws=env_config["number_of_draws"])  # return an env instance


    register_env("my_env", env_creator)
    ModelCatalog.register_custom_model("pa_model", ActionMaskModel)


    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean', # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1)

    ray.init()
    tune.run(
        "PPO",
        stop={"timesteps_total": 1e6},
        scheduler=asha_scheduler,
        num_samples=10,
        local_dir="/home/ppirog/ray_results/bipedal",
        config={
            "env": "my_env",  #
            # configuration for environment
            "env_config": {"probabilities": [0.4, 0.6, 0.8],
                           "number_of_draws": 100},
            "framework": "tf2",
            "num_gpus": 0,
            "num_workers": 60,
            "model": {"custom_model": "pa_model", },
            "evaluation_interval": 1,
            "evaluation_num_episodes": 2,
            "rollout_fragment_length": 2000,
            "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
            #"use_gae": tune.choice([True, False]),
            "train_batch_size": tune.choice([5000, 10000, 20000, 40000]),
            "sgd_minibatch_size": tune.choice([128, 1024, 4096, 8192]),
            "num_sgd_iter": tune.choice([5, 10, 30]),
            "vf_loss_coeff": tune.choice([0.1, 1, 10]),
            "vf_share_layers": tune.choice([True, False]),
            "entropy_coeff": tune.choice([0, 0.1, 1]),
            "clip_param": tune.choice([0.05, 0.1, 0.3, 0.5]),
            "vf_clip_param": tune.choice([1, 5, 10]),
            "grad_clip": tune.choice([None, 0.01, 0.1, 1]),
            "kl_target": tune.choice([0.005, 0.01, 0.05]),

        },
        #local_dir="output_dir",  # directory to save results
        checkpoint_freq=2,  # frequency between checkpoints
        keep_checkpoints_num=6,
    )
    """
    tune.run("PPO",
             # algorithm specific configuration
             config={
                    "env": "my_env",  #
                     # configuration for environment
                     "env_config": {"probabilities": [0.4, 0.6, 0.8],
                                    "number_of_draws": 100},
                     "framework": "tf2",
                     "num_gpus": 1,
                     "num_workers": 2,
                     "model": {"custom_model": "pa_model", },
                     "evaluation_interval": 1,
                     "evaluation_num_episodes": 2
                     },
             local_dir="output_dir",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=6,
             )
             """
# this sript is OK

# sudo tensorboard --logdir=/home/ppirog/ray_results/bipedal/PPO --bind_all --port=12301
# kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
# http://hpc.if.uz.zgora.pl:12301/