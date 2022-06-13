import random

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer as Trainer
from ray.tune.schedulers import PopulationBasedTraining,ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

if __name__ == "__main__":

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore,
    )
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        #metric='episode_reward_mean',
        #mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1)

    analysis = tune.run(
        "PPO",
        name="pbt_humanoid_test",
        search_alg=HyperOptSearch(),
        scheduler=asha_scheduler, #pbt
        num_samples=1,
        metric="episode_reward_mean",
        mode="max",
        config={
            "env": "BipedalWalkerHardcore-v3", #"Humanoid-v1",
            "kl_coeff": 1.0,
            "num_workers": 60,
            "num_gpus": 0, # number of GPUs to use
            "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([128, 512, 2048]),
            "train_batch_size": tune.choice([10000, 20000, 40000]),
        },
    )

    print("best hyperparameters: ", analysis.best_config)

    # this sript is OK

    # sudo tensorboard --logdir=/root/ray_results/pbt_humanoid_test --bind_all --port=12301
    # kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
    # http://hpc.if.uz.zgora.pl:12301/