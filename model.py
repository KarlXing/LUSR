import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
from utils import reparameterize


# Models for carracing games
class Encoder(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size) 

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3, flatten_size=1024):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x


# DisentangledVAE here is actually Cycle-Consistent VAE, disentangled stands for the disentanglement between domain-general and domain-specifc embeddings 
class DisentangledVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, img_channel = 3, flatten_size = 1024):
        super(DisentangledVAE, self).__init__()
        self.encoder = Encoder(class_latent_size, content_latent_size, img_channel, flatten_size)
        self.decoder = Decoder(class_latent_size + content_latent_size, img_channel, flatten_size)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x


# Models for CARLA autonomous driving
class CarlaEncoder(nn.Module):
    def __init__(self, class_latent_size = 16, content_latent_size = 32, input_channel = 3, flatten_size = 9216):
        super(CarlaEncoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size) 

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class CarlaDecoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3):
        super(CarlaDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, 9216)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (-1,256,6,6))
        x = self.main(x)
        return x


class CarlaDisentangledVAE(nn.Module):
    def __init__(self, class_latent_size = 16, content_latent_size = 32, img_channel = 3, flatten_size=9216):
        super(CarlaDisentangledVAE, self).__init__()
        self.encoder = CarlaEncoder(class_latent_size, content_latent_size, img_channel, flatten_size)
        self.decoder = CarlaDecoder(class_latent_size + content_latent_size, img_channel)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x


class CarlaLatentPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[64,64]):
        super(CarlaLatentPolicy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        actor_features = self.actor(x)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        self.dist = Beta(alpha, beta)
        if action is None:
            action = self.dist.sample()
        else:
            action = (action+1)/2
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x)
        return action*2-1, action_log_prob, value.squeeze(-1), entropy


class CarlaSimpleEncoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(CarlaSimpleEncoder, self).__init__()
        self.latent_size = latent_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(9216, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu


class CarlaImgPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[400,300]):
        super(CarlaImgPolicy, self).__init__()
        self.main_actor = CarlaSimpleEncoder(latent_size = input_dim-1)
        self.main_critic = CarlaSimpleEncoder(latent_size = input_dim-1)
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(layer_init(nn.Linear(hidden_layer[-1], 1), gain=1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        speed = x[:, -1:]
        x = x[:, :-1].view(-1, 3,128,128)  # image size in carla driving task is 128x128
        x1 = self.main_actor(x)
        x1 = torch.cat([x1, speed], dim=1)

        x2 = self.main_critic(x)
        x2 = torch.cat([x2, speed], dim=1)

        actor_features = self.actor(x1)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        self.dist = Beta(alpha, beta)
        if action is None:
            action = self.dist.sample()
        else:
            action = (action+1)/2
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x2)
        return action*2-1, action_log_prob, value.squeeze(-1), entropy
