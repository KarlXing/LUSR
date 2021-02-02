import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override

import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
from torch.distributions.kl import kl_divergence

import gym
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--deterministic-sample', default=False, action='store_true')
parser.add_argument('--env', default="CarRacing-v0", type=str)
parser.add_argument('--num-episodes', default=100, type=int)
parser.add_argument('--model-path', default='./', type=str)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--save-path', default='./', type=str)
parser.add_argument('--action-repeat', default=4, type=int)
args = parser.parse_args()


######## obs preprocess ###########
def process_obs(obs): # a single frame (96, 96, 3) for CarRacing
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    obs = np.transpose(obs, (2,0,1))
    return torch.from_numpy(obs).unsqueeze(0)

######## models ##########
class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu


class MyModel(nn.Module):
    def __init__(self, deterministic_sample=False, latent_size=16):
        nn.Module.__init__(self)

        self.main = Encoder(latent_size=latent_size)
        self.critic = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.deterministic_sample = deterministic_sample

    def forward(self, x):
        with torch.no_grad():
            features = self.main(x)
            actor_features = self.actor(features)
            alpha = self.alpha_head(actor_features)+1
            beta = self.beta_head(actor_features)+1
        dist = Beta(alpha, beta)
        if not self.deterministic_sample:
            action = dist.sample().squeeze().numpy()
        else:
            action = dist.mean.squeeze().numpy()
        action[0] = action[0]*2-1
        return action


########### Do Evaluation #################
def main():
    results = []
    env = gym.make(args.env)
    model = MyModel(args.deterministic_sample, args.latent_size)
    weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    for i in range(args.num_episodes):
        rewards, done, obs = 0, False, env.reset()
        obs = process_obs(obs)
        while not done:
            action = model(obs)
            for _ in range(args.action_repeat):
                obs, reward, done, info = env.step(action)
                rewards += reward
                if done:
                    break
            if args.render:
                env.render()
            obs = process_obs(obs)
        results.append(rewards)

    print('Evaluate %d episodes and achieved %f scores' % (args.num_episodes, np.mean(results)))
    file_name = "%s_%d_%s" % (args.env, args.num_episodes, args.model_path)
    torch.save(results, os.path.join(args.save_path, file_name))

if __name__ == '__main__':
    main()