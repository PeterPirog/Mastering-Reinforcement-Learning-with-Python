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