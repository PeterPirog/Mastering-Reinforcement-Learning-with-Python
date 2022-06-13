import ray
from ray import tune
#from inventory_env import InventoryEnv
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#optuna-tune-suggest-optuna-optunasearch
from ray.tune.suggest.optuna import OptunaSearch

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    #metric='episode_reward_mean', # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    #mode='max',
    max_t=30, #100
    grace_period=10,
    reduction_factor=3,
    brackets=1)

# https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7
algo = OptunaSearch()

ray.init()
analysis=tune.run(
    "PPO",
    stop={"timesteps_total": 1e6},
    scheduler=asha_scheduler,
    metric='episode_reward_mean', # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    mode='max',
    search_alg=algo,
    num_samples=100,
    local_dir="/home/ppirog/ray_results/lunarlander",
    config={
        "env": "LunarLander-v2", #BipedalWalkerHardcore-v3
        "disable_env_checking":True,
        "rollout_fragment_length": 2000,
        "num_gpus": 0,
        "num_workers": 8,
        "batch_mode":"complete_episodes",
        "framework": "tf2",

        #"lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
        "lr": tune.choice([0.01, 0.001, 0.0001, 0.00001]),


        "train_batch_size": tune.choice([5000, 10000, 20000, 40000]),


        # https://docs.ray.io/en/latest/rllib/rllib-models.html#default-model-config-settings

        "model":{
    #        "vf_share_layers": tune.choice([True, False]),
            "fcnet_hiddens": tune.choice([[256, 256], [64, 64], [128, 128]]),
        },
        # PPO hyperparameters https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#proximal-policy-optimization-ppo
        "lr_schedule":None,
        "use_critic":True, # True
        #"use_gae": tune.choice([True, False]), # True
        "lambda":tune.choice([0.9,1.0]),
        "kl_coeff":0.2,
        "sgd_minibatch_size": tune.choice([128, 1024, 4096]), # 128
        "num_sgd_iter": tune.choice([3, 10, 30]),
        "shuffle_sequences":True,
        "vf_loss_coeff": tune.choice([0.5, 0.8, 1.0]), # 1
        "entropy_coeff": tune.choice([0, 0.01, 0.05]), # 0.0
        "entropy_coeff_schedule":None,
        "clip_param": tune.choice([0.1, 0.2, 0.3]), # 0.3
        "vf_clip_param": tune.choice([1, 5, 10]), # 10
        "grad_clip": tune.choice([None, 0.01, 0.1, 1]), # None
        "kl_target": tune.choice([0.005, 0.01, 0.05]), # 0.01

    },

)
print("Best hyperparameters found were: ", analysis.best_config)

# this sript is OK

# sudo tensorboard --logdir=/home/ppirog/ray_results/bipedal/PPO --bind_all --port=12301
# sudo tensorboard --logdir=/home/ppirog/ray_results/lunarlander/PPO --bind_all --port=12301
# kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
# http://hpc.if.uz.zgora.pl:12301/