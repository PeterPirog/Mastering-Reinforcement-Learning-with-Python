from pprint import pprint
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.utils.typing import TrainerConfigDict
from ray.tune.schedulers import ASHAScheduler

# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#optuna-tune-suggest-optuna-optunasearch
from ray.tune.suggest.optuna import OptunaSearch

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',  # training_iteration time_total_s
    # metric='episode_reward_mean', # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    # mode='max',
    max_t=100,  # 100 2000
    grace_period=10,
    reduction_factor=3,
    brackets=1)

# https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7
algo = OptunaSearch()


class MyPPO(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        ppo_config = PPOConfig().to_dict()
        ppo_config["num_units"] = None
        ppo_config["num_layers"] = None

        # ppo_config["fcnet_activation"] = None
        # pprint(ppo_config, width=1)

        return ppo_config

    def validate_config(self, config: TrainerConfigDict) -> None:
        # use hyperparameters for the nn model
        # if config["fcnet_activation"] is not None:
        #    config["model"]["fcnet_activation"] = self.fcnet_activation
        #    # del config["fcnet_activation"]

        if config["num_layers"] is not None and config["num_units"] is not None:
            self.num_layers = config["num_layers"]
            self.num_units = config["num_units"]
            config["model"]["fcnet_hiddens"] = [self.num_units] * self.num_layers
        super().validate_config(config)


tune.register_trainable("MyPPO", MyPPO)

search_space = {
    # Config common settings https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
    "env": "BipedalWalkerHardcore-v3",  # BipedalWalkerHardcore-v3 LunarLander-v2
    "disable_env_checking": True,
    "rollout_fragment_length": 2000,
    "num_gpus": 0,
    "num_workers": 8,
    "batch_mode": "complete_episodes",
    "framework": "tf2",
    "preprocessor_pref": "deepmind",

    "horizon": None,
    "lr": tune.choice([0.01, 0.001, 0.0001, 0.00001]),  # tune.choice([0.01, 0.001, 0.0001, 0.00001]), tune.qloguniform(1e-4, 1e-1, 5e-5)
    "gamma": tune.quniform(0.9, 0.999, 0.001),
    # "sgd_stepsize": tune.quniform(5e-6, 0.003, 5e-6),

    "train_batch_size": tune.choice([5000, 10000, 20000, 40000]),

    # https://docs.ray.io/en/latest/rllib/rllib-models.html#default-model-config-settings
    "num_layers": tune.randint(1, 4),
    "num_units": tune.randint(8, 257),
    "model": {
        "fcnet_activation": tune.choice(["tanh", "silu"]),
        "vf_share_layers": tune.choice([True, False]),
    },

    # PPO hyperparameters https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#proximal-policy-optimization-ppo
    "lr_schedule": None,
    "entropy_coeff_schedule":None,
    "use_critic": True,
    "use_gae": True,
    "lambda": tune.quniform(0.9, 1.0, 0.01),
    "sgd_minibatch_size": tune.choice([128, 1024, 4096]),  # 128
    "num_sgd_iter": tune.choice([3, 10, 30]),
    "shuffle_sequences": True,
    "vf_loss_coeff": tune.quniform(0.5, 1.0, 0.05),
    "entropy_coeff": tune.quniform(0, 0.05, 0.005),
    "clip_param": tune.quniform(0.1, 0.3, 0.05),
    "vf_clip_param": tune.randint(1, 11),
    "grad_clip": None,
    "kl_target": tune.quniform(0.003, 0.03, 0.001),

}

ray.init()
analysis = tune.run(
    run_or_experiment="MyPPO",
    stop={"timesteps_total": 1e6},
    scheduler=asha_scheduler,
    metric='episode_reward_mean',
    # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    mode='max',
    search_alg=algo,
    num_samples=100,
    local_dir="/home/ppirog/ray_results/lunarlander",
    config=search_space,
)
print("Best hyperparameters found were: ", analysis.best_config)

"""
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.utils.typing import TrainerConfigDict
from ray.tune.suggest.optuna import OptunaSearch



# You can ignore this. Collab does not have enough CPUs to run with the default settings. This is a workaround.
ray.shutdown()
ray.init(num_cpus=10)


class MyPPO(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        ppo_config = PPOConfig().to_dict()
        ppo_config["num_units"] = None
        ppo_config["num_layers"] = None
        return ppo_config

    def validate_config(self, config: TrainerConfigDict) -> None:
        if config["num_layers"] is not None and config["num_units"] is not None:
            self.num_layers = config["num_layers"]
            self.num_units = config["num_units"]
            config["model"]["fcnet_hiddens"] = [self.num_units] * self.num_layers
        super().validate_config(config)


tune.register_trainable("MyPPO", MyPPO)

search_space = {
    "env": "LunarLander-v2",
    "num_layers": tune.randint(1, 4),
    "num_units": tune.randint(8, 257),
    "lr": tune.uniform(5e-6, 3e-3),
}

analysis = tune.run(
    config=search_space,
    search_alg=OptunaSearch(),
    run_or_experiment="MyPPO",
    metric='episode_reward_mean',
    mode='max',
    stop={"episodes_total": 100},
    num_samples=5,
)

print("Best hyperparameters found were: ", analysis.best_config)

ray.shutdown()


"""
