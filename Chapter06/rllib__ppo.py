import pprint
from ray import tune
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo import PPOTrainer as Trainer
# https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py


if __name__ == '__main__':
    config = DEFAULT_CONFIG.copy()


    pp = pprint.PrettyPrinter(indent=4)

    #config['env'] = "CartPole-v0"
    config['env'] = "BipedalWalkerHardcore-v3"
    config['framework'] = "tf2"
    #distributional training
    #config['num_atoms'] = 51 # default 1
    #config['v_min'] = 0.0 # default -10.0
    #config['v_max'] = 200.0 # default 10.0

    config['num_gpus'] = 0
    config['num_workers'] = 53
    config['evaluation_num_workers'] = 10
    config['evaluation_interval'] = 1
    #config['log_level'] = 'WARN' # 'INFO', 'DEBUG'
    pp.pprint(config)
    tune.run(Trainer, config=config)

    # this sript is OK
    #  /root/ray_results/PPOTrainer_2022-05-08_07-25-31
    # tensorboard --logdir=/root/ray_results/PPOTrainer_2022-05-08_07-25-31 --bind_all --port=12301
    # kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
    # http://hpc.if.uz.zgora.pl:12301/
