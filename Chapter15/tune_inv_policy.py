import ray
from ray import tune
from inventory_env import InventoryEnv
from ray.tune.schedulers import PopulationBasedTraining,ASHAScheduler

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
    stop={"timesteps_total": 1e6},
    scheduler=asha_scheduler,
    num_samples=100,
    local_dir="/home/ppirog/ray_results",
    config={
        "env": InventoryEnv,
        #"rollout_fragment_length": 40, # maximum episode len
        "num_gpus": 0,
        "num_workers": 7,
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
        
        
        "lr": 0.0001,
        "use_gae":True,
        "train_batch_size": 10000,
        "sgd_minibatch_size": 1024,
        "num_sgd_iter": 30,
        "vf_loss_coeff": 1,
        "vf_share_layers": True,
        "entropy_coeff": 0,
        "clip_param": 0.5,
        "vf_clip_param": 10,
        "grad_clip": 0.01,
        "kl_target": 0.005,
"""


# this sript is OK

# sudo tensorboard --logdir=/home/ppirog/ray_results/PPO --bind_all --port=12301
# kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
# http://hpc.if.uz.zgora.pl:12301/