import gym
from gym.spaces import Box, Dict
import numpy as np
import argparse
import pprint
from ray import tune
import ray
from ray.rllib.agents.ppo.ppo import (
    DEFAULT_CONFIG,
    PPOTrainer as trainer)


class MountainCar(gym.Env):
    def __init__(self, env_config={}):
        self.wrapped = gym.make("MountainCar-v0")
        self.action_space = self.wrapped.action_space
        self.t = 0
        self.reward_fun = env_config.get("reward_fun")
        self.lesson = env_config.get("lesson")
        self.use_action_masking = env_config.get("use_action_masking", True)
        self.action_mask = None
        self.reset()
        if self.use_action_masking:
            self.observation_space = Dict(
                {
                    "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                    "actual_obs": self.wrapped.observation_space,
                }
            )
        else:
            self.observation_space = self.wrapped.observation_space
        print('self.observation_space["actual_obs"]:',self.observation_space["actual_obs"])

    def _get_obs(self):
        raw_obs = np.array(self.wrapped.unwrapped.state)
        if self.use_action_masking:
            self.update_avail_actions()
            obs = {
                "action_mask": self.action_mask,
                "actual_obs": raw_obs,
            }
        else:
            obs = raw_obs
        return obs

    def reset(self):
        self.wrapped.reset()
        self.t = 0
        self.wrapped.unwrapped.state = self._get_init_conditions()
        obs = self._get_obs()
        return obs

    def _get_init_conditions(self):
        if self.lesson == 0:
            low = 0.1
            high = 0.4
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 1:
            low = -0.4
            high = 0.1
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 2:
            low = -0.6
            high = -0.4
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 3:
            low = -0.6
            high = -0.1
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 4 or self.lesson is None:
            low = -0.6
            high = -0.4
            velocity = 0
        else:
            raise ValueError
        obs = (self.wrapped.np_random.uniform(low=low, high=high), velocity)
        return obs

    def set_lesson(self, lesson):
        self.lesson = lesson

    def step(self, action):
        self.t += 1
        state, reward, done, info = self.wrapped.step(action)
        if self.reward_fun == "custom_reward":
            position, velocity = state
            reward += (abs(position + 0.5) ** 2) * (position > -0.5)
        obs = self._get_obs()
        if self.t >= 200:
            done = True
        return obs, reward, done, info

    def update_avail_actions(self):
        self.action_mask = np.array([1.0] * self.action_space.n)
        pos, vel = self.wrapped.unwrapped.state
        # 0: left, 1: no action, 2: right
        if (pos < -0.3) and (pos > -0.8) and (vel < 0) and (vel > -0.05):
            self.action_mask[1] = 0
            self.action_mask[2] = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        help='Gym env name.')
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    config_update = {
        "env": "BipedalWalkerHardcore-v3",  # "BipedalWalkerHardcore-v3" args.env
        "framework": "tf2",
        "num_gpus": 0,
        "num_workers": 53,
        "evaluation_num_workers": 10,
        "evaluation_interval": 1,
        "lr": 1e-4,
    }

    config.update(config_update)
    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(config)

    ray.init()
    tune.run(trainer,
             stop={"timesteps_total": 2000000},
             config=config
             )

    # this sript is OK
    #  /root/ray_results/PPOTrainer_2022-05-08_07-25-31
    # sudo tensorboard --logdir=/root/ray_results/PPOTrainer_2022-05-10_17-55-00 --bind_all --port=12301
    # kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
    # http://hpc.if.uz.zgora.pl:12301/

"""
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler

env = MountainCar()
print(env.reset())
print(env.action_space)
print(env.step(0))

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='episode_reward_mean',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    brackets=1)

ray.init()
tune.run(
    "PPO",
    config={
        # "env": MountainCar,
        "env": "MountainCar-v0",
        # "env": "BipedalWalkerHardcore-v3",
        # "rollout_fragment_length": 40,
        "num_gpus": 0,
        "num_workers": 63,
        "framework": "tf2",
        "lr": 1e-3,
    }
)


    stop={"timesteps_total": 1e6},
    scheduler=asha_scheduler,
    num_samples=5,
    config={
        "env": MountainCar,
        "rollout_fragment_length": 40,
        "num_gpus": 0,
        "num_workers": 10,
        "framework":"tf2",
        "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
        "use_gae": tune.choice([True, False]),
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

)
"""
