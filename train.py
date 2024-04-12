import gymnasium as gym

from sbx import SAC

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=110, progress_bar=True)
