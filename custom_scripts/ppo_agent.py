import argparse
import pprint
from ray import tune
import ray
from ray.rllib.agents.ppo.ppo import (
    DEFAULT_CONFIG,
    PPOTrainer as trainer)

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
        "lr":tune.choice([1e-2,1e-3,1e-4,1e-5]),
        "use_gae":tune.choice([True,False])
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
