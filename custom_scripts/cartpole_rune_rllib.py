from datetime import datetime
from glob import glob
import os

from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DQNTrainer as Trainer
from ray.rllib.rollout import rollout
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import tensorflow as tf


env_id = "CartPole-v0"


def on_train_result(info):
    info["result"]["objective"] = info["result"]["episode_reward_mean"]


def get_agent(trial):
    agent = Trainer(config=trial.config)
    agent.restore(trial._checkpoint.value)
    return agent


def get_best_trials(trials, metric):
    return sorted(trials, key=lambda trial: trial.last_result[metric], reverse=True)


def remove_checkpoints(trials):
    for trial in trials:
        for path in glob(os.path.join(trial._checkpoint.value + "*")):
            os.remove(path)
        os.rmdir(os.path.dirname(trial._checkpoint.value))
        trial.clear_checkpoint()


def run_trials(num_cpus=os.cpu_count(), num_gpus=0, logdir=None):
    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    hyperopt = HyperOptSearch(
        {
            "gamma": (1 - hp.loguniform("_gamma", np.log(1e-4), np.log(1e-1))) / 1,
            "lr": hp.loguniform("lr", np.log(1e-6), np.log(1e-3)),
            "num_atoms": hp.choice("num_atoms", [1, 51]),
            "noisy": hp.choice("noisy", [False, True]),
            "hiddens": hp.choice(
                "hiddens",
                [
                    [scope.int(64 * (2 ** hp.quniform("_layer_1_1", 0, 3, 1)))],
                    [
                        scope.int(64 * (2 ** hp.quniform("_layer_2_1", 0, 3, 1))),
                        scope.int(64 * (2 ** hp.quniform("_layer_2_2", 0, 3, 1))),
                    ],
                ],
            ),
        },
        max_concurrent=32,
        reward_attr="objective",
    )

    hyperband = AsyncHyperBandScheduler(
        time_attr="training_iteration", reward_attr="objective", max_t=20
    )

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    return tune.run(
        DQNAgent,
        name=now,
        num_samples=4,
        search_alg=hyperopt,
        scheduler=hyperband,
        stop={},
        resources_per_trial={"cpu": 1, "gpu": np.clip(num_gpus / num_cpus, 0, 1)},
        config={
            "env": env_id,
            "num_gpus": np.clip(num_gpus / num_cpus, 0, 1),
            "callbacks": {"on_train_result": tune.function(on_train_result)},
        },
        local_dir=logdir,
        checkpoint_at_end=True,
    )


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    trials = run_trials(
        num_gpus=tf.contrib.eager.num_gpus(),
        logdir=os.path.join(dir_path, "ray_results"),
    )

    best_trials = get_best_trials(trials, "objective")
    best_trial = best_trials[0]
    agent = get_agent(best_trial)
    print(
        "best score: {}, config: {}, checkpoint: {}".format(
            best_trial.last_result["objective"],
            best_trial.config,
            best_trial._checkpoint.value,
        )
    )
    remove_checkpoints(best_trials[1:])

    rollout(agent, env_id, 1000, no_render=True)


if __name__ == "__main__":
    main()