from ray import tune

#https://docs.ray.io/en/latest/tune/api_docs/execution.html
tune.run("PPO",
         config={
             "env":"CartPole-v0",
             "framework": "tf2",
             "evaluation_interval":2,
             #"evaluation_num_episodes":20,
             "evaluation_duration":20,
         },
         local_dir="cartpole",
         checkpoint_freq=2)

# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/3D_observations/cartpole_v1/PPO
