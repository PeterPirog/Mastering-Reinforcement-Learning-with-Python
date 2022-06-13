import pprint
from ray import tune
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn import DQNTrainer as Trainer
# https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py


if __name__ == '__main__':
    config = DEFAULT_CONFIG.copy()


    pp = pprint.PrettyPrinter(indent=4)

    config['env'] = "CartPole-v0"
    #distributional training
    config['num_atoms'] = 51 # default 1
    config['v_min'] = 0.0 # default -10.0
    config['v_max'] = 200.0 # default 10.0

    config['num_gpus'] = 0
    config['num_workers'] = 53
    config['evaluation_num_workers'] = 10
    config['evaluation_interval'] = 1
    config['learning_starts'] = 5000
    pp.pprint(config)
    tune.run(Trainer, config=config)

    # this sript is OK
