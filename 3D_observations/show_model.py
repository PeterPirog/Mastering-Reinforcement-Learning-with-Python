
import tensorflow as tf
model_path="/home/ppirog/projects/Mastering-Reinforcement-Learning-with-Python/3D_observations/cartpole/PPO/PPO_CartPole-v0_2890c_00000_0_2022-06-30_16-03-19/checkpoint_000002"
new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()
