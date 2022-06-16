import ray
from ray import tune
# from inventory_env import InventoryEnv
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#optuna-tune-suggest-optuna-optunasearch
from ray.tune.suggest.optuna import OptunaSearch

asha_scheduler = ASHAScheduler(
    time_attr='time_total_s', # training_iteration
    # metric='episode_reward_mean', # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    # mode='max',
    max_t=900,  # 100
    grace_period=10,
    reduction_factor=3,
    brackets=1)

# https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7
algo = OptunaSearch()

ray.init()
analysis = tune.run(
    "PPO",
    stop={"timesteps_total": 1e6},
    scheduler=asha_scheduler,
    metric='episode_reward_mean',
    # episode_reward_mean # maximum episode len episode_reward_min reward  episode_reward_mean episode_len_mean
    mode='max',
    search_alg=algo,
    num_samples=100,
    local_dir="/home/ppirog/ray_results/lunarlander",
    config={
        # Config common settings https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
        "env": "LunarLander-v2",  # BipedalWalkerHardcore-v3
        "disable_env_checking": True,
        "rollout_fragment_length": 2000,
        "num_gpus": 0,
        "num_workers": 8,
        "batch_mode": "complete_episodes",
        "framework": "tf2",
        "preprocessor_pref": "deepmind",

        "horizon": None,
        "lr": tune.qloguniform(1e-4, 1e-1, 5e-5),  # tune.choice([0.01, 0.001, 0.0001, 0.00001]),
        "gamma": tune.quniform(0.9, 0.999, 0.001),
        #"sgd_stepsize": tune.quniform(5e-6, 0.003, 5e-6),

        "train_batch_size": tune.choice([5000, 10000, 20000, 40000]),

        # https://docs.ray.io/en/latest/rllib/rllib-models.html#default-model-config-settings

        "model": {
            "vf_share_layers": tune.choice([True, False]),
            "fcnet_hiddens": tune.choice([[32, 32], [64, 64], [128, 128]]),
            #"fcnet_hiddens": tune.sample_from(lambda: [tune.choice(range(8,257)).sample()]*tune.choice(range(1,4)).sample()),
            "fcnet_activation": "tanh",
        },
        # PPO hyperparameters https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#proximal-policy-optimization-ppo
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
        "grad_clip":None,
        "kl_target": tune.quniform(0.003, 0.03, 0.001),

    },

)
print("Best hyperparameters found were: ", analysis.best_config)

# How to define ranges for hyperparameter optimization: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#site-title


# this sript is OK

# sudo tensorboard --logdir=/home/ppirog/ray_results/bipedal/PPO --bind_all --port=12301
# sudo tensorboard --logdir=/home/ppirog/ray_results/lunarlander/PPO --bind_all --port=12301
# kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
# http://hpc.if.uz.zgora.pl:12301/
# Current best trial: e3ad3ac5 with episode_reward_mean=126.40451759732242 and parameters={'num_workers': 8, 'num_envs_per_worker': 1, 'create_env_on_driver': False, 'rollout_fragment_length': 625, 'batch_mode': 'complete_episodes', 'gamma': 0.993, 'lr': 0.0003954038829555581, 'train_batch_size': 5000, 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, 'fcnet_hiddens': [128, 128], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}, 'optimizer': {}, 'horizon': None, 'soft_horizon': False, 'no_done_at_end': False, 'env': 'LunarLander-v2', 'observation_space': None, 'action_space': None, 'env_config': {}, 'remote_worker_envs': False, 'remote_env_batch_wait_ms': 0, 'env_task_fn': None, 'render_env': False, 'record_env': False, 'clip_rewards': None, 'normalize_actions': True, 'clip_actions': False, 'preprocessor_pref': 'deepmind', 'log_level': 'WARN', 'callbacks': <class 'ray.rllib.agents.callbacks.DefaultCallbacks'>, 'ignore_worker_failures': False, 'log_sys_usage': True, 'fake_sampler': False, 'framework': 'tf2', 'eager_tracing': False, 'eager_max_retraces': 20, 'explore': True, 'exploration_config': {'type': 'StochasticSampling'}, 'evaluation_interval': None, 'evaluation_duration': 10, 'evaluation_duration_unit': 'episodes', 'evaluation_parallel_to_training': False, 'in_evaluation': False, 'evaluation_config': {'num_workers': 8, 'num_envs_per_worker': 1, 'create_env_on_driver': False, 'rollout_fragment_length': 625, 'batch_mode': 'complete_episodes', 'gamma': 0.993, 'lr': 0.0003954038829555581, 'train_batch_size': 5000, 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, 'fcnet_hiddens': [128, 128], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}, 'optimizer': {}, 'horizon': None, 'soft_horizon': False, 'no_done_at_end': False, 'env': 'LunarLander-v2', 'observation_space': None, 'action_space': None, 'env_config': {}, 'remote_worker_envs': False, 'remote_env_batch_wait_ms': 0, 'env_task_fn': None, 'render_env': False, 'record_env': False, 'clip_rewards': None, 'normalize_actions': True, 'clip_actions': False, 'preprocessor_pref': 'deepmind', 'log_level': 'WARN', 'callbacks': <class 'ray.rllib.agents.callbacks.DefaultCallbacks'>, 'ignore_worker_failures': False, 'log_sys_usage': True, 'fake_sampler': False, 'framework': 'tf2', 'eager_tracing': False, 'eager_max_retraces': 20, 'explore': True, 'exploration_config': {'type': 'StochasticSampling'}, 'evaluation_interval': None, 'evaluation_duration': 10, 'evaluation_duration_unit': 'episodes', 'evaluation_parallel_to_training': False, 'in_evaluation': False, 'evaluation_config': {}, 'evaluation_num_workers': 0, 'custom_eval_function': None, 'always_attach_evaluation_results': False, 'keep_per_episode_custom_metrics': False, 'sample_async': False, 'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 'observation_filter': 'NoFilter', 'synchronize_filters': True, 'tf_session_args': {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}, 'local_tf_session_args': {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}, 'compress_observations': False, 'metrics_episode_collection_timeout_s': 180, 'metrics_num_episodes_for_smoothing': 100, 'min_time_s_per_reporting': None, 'min_train_timesteps_per_reporting': None, 'min_sample_timesteps_per_reporting': 0, 'seed': None, 'extra_python_environs_for_driver': {}, 'extra_python_environs_for_worker': {}, 'num_gpus': 0, '_fake_gpus': False, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0, 'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1, 'placement_strategy': 'PACK', 'input': 'sampler', 'input_config': {}, 'actions_in_input_normalized': False, 'input_evaluation': ['is', 'wis'], 'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None, 'output_config': {}, 'output_compress_columns': ['obs', 'new_obs'], 'output_max_file_size': 67108864, 'multiagent': {'policies': {'default_policy': PolicySpec(policy_class=<class 'ray.rllib.policy.tf_policy_template.PPOTFPolicy'>, observation_space=Box([-inf -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf], (8,), float32), action_space=Discrete(4), config={})}, 'policy_map_capacity': 100, 'policy_map_cache': None, 'policy_mapping_fn': None, 'policies_to_train': None, 'observation_fn': None, 'replay_mode': 'independent', 'count_steps_by': 'env_steps'}, 'logger_config': None, '_tf_policy_handles_more_than_one_loss': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, '_disable_execution_plan_api': False, 'disable_env_checking': True, 'simple_optimizer': True, 'monitor': -1, 'evaluation_num_episodes': -1, 'metrics_smoothing_episodes': -1, 'timesteps_per_iteration': 0, 'min_iter_time_s': -1, 'collect_metrics_timeout': -1, 'use_critic': True, 'use_gae': True, 'lambda': 0.92, 'kl_coeff': 0.2, 'sgd_minibatch_size': 4096, 'shuffle_sequences': True, 'num_sgd_iter': 30, 'lr_schedule': None, 'vf_loss_coeff': 0.9, 'entropy_coeff': 0.025, 'entropy_coeff_schedule': None, 'clip_param': 0.2, 'vf_clip_param': 10, 'grad_clip': None, 'kl_target': 0.03, 'vf_share_layers': -1}, 'evaluation_num_workers': 0, 'custom_eval_function': None, 'always_attach_evaluation_results': False, 'keep_per_episode_custom_metrics': False, 'sample_async': False, 'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 'observation_filter': 'NoFilter', 'synchronize_filters': True, 'tf_session_args': {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}, 'local_tf_session_args': {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}, 'compress_observations': False, 'metrics_episode_collection_timeout_s': 180, 'metrics_num_episodes_for_smoothing': 100, 'min_time_s_per_reporting': None, 'min_train_timesteps_per_reporting': None, 'min_sample_timesteps_per_reporting': 0, 'seed': None, 'extra_python_environs_for_driver': {}, 'extra_python_environs_for_worker': {}, 'num_gpus': 0, '_fake_gpus': False, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0, 'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1, 'placement_strategy': 'PACK', 'input': 'sampler', 'input_config': {}, 'actions_in_input_normalized': False, 'input_evaluation': ['is', 'wis'], 'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None, 'output_config': {}, 'output_compress_columns': ['obs', 'new_obs'], 'output_max_file_size': 67108864, 'multiagent': {'policies': {'default_policy': PolicySpec(policy_class=<class 'ray.rllib.policy.tf_policy_template.PPOTFPolicy'>, observation_space=Box([-inf -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf], (8,), float32), action_space=Discrete(4), config={})}, 'policy_map_capacity': 100, 'policy_map_cache': None, 'policy_mapping_fn': None, 'policies_to_train': None, 'observation_fn': None, 'replay_mode': 'independent', 'count_steps_by': 'env_steps'}, 'logger_config': None, '_tf_policy_handles_more_than_one_loss': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, '_disable_execution_plan_api': False, 'disable_env_checking': True, 'simple_optimizer': True, 'monitor': -1, 'evaluation_num_episodes': -1, 'metrics_smoothing_episodes': -1, 'timesteps_per_iteration': 0, 'min_iter_time_s': -1, 'collect_metrics_timeout': -1, 'use_critic': True, 'use_gae': True, 'lambda': 0.92, 'kl_coeff': 0.2, 'sgd_minibatch_size': 4096, 'shuffle_sequences': True, 'num_sgd_iter': 30, 'lr_schedule': None, 'vf_loss_coeff': 0.9, 'entropy_coeff': 0.025, 'entropy_coeff_schedule': None, 'clip_param': 0.2, 'vf_clip_param': 10, 'grad_clip': None, 'kl_target': 0.03, 'vf_share_layers': -1}
"""
"lr_schedule": None,
"use_critic": True,  # True
# "use_gae": tune.choice([True, False]), # True
"lambda": tune.choice([0.9, 1.0]),
"kl_coeff": 0.2,
"sgd_minibatch_size": tune.choice([128, 1024, 4096]),  # 128
"num_sgd_iter": tune.choice([3, 10, 30]),
"shuffle_sequences": True,
"vf_loss_coeff": tune.choice([0.5, 0.8, 1.0]),  # 1
"entropy_coeff": tune.choice([0, 0.01, 0.05]),  # 0.0
"entropy_coeff_schedule": None,
"clip_param": tune.choice([0.1, 0.2, 0.3]),  # 0.3
"vf_clip_param": tune.choice([1, 5, 10]),  # 10
"grad_clip": tune.choice([None, 0.01, 0.1, 1]),  # None
"kl_target": tune.choice([0.005, 0.01, 0.05]),  # 0.01
"""

# https://discuss.ray.io/t/how-to-define-fcnet-hiddens-size-and-number-of-layers-in-rllib-tune/6504/4
