import torch
import torch.nn as nn
import gym
from gym import spaces
import gym_carla
import carla
import numpy as np
import argparse
import random 
import rlcodebase
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import Config, Logger
from torch.utils.tensorboard import SummaryWriter
from model import CarlaLatentPolicy, CarlaImgPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--weather', default=0, type=int)
parser.add_argument('--action-repeat', default=1, type=int)
parser.add_argument('--model-path', default='./', type=str)
parser.add_argument('--use-encoder', default=False, action='store_true')
parser.add_argument('--encoder-path', default='./', type=str)
parser.add_argument('--latent-size', default=16, type=int, help='dimension of latent state embedding')
parser.add_argument('--port', default=2000, type=int)
parser.add_argument('--num-eval', default=10, type=int)
parser.add_argument('--save-obs', default=False, action='store_true')
parser.add_argument('--save-obs-path', default='./obs', type=str)
args = parser.parse_args()

weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters(50, 0, 0, 0.35, 0, -40)]
weather = weathers[args.weather]
start_point = (75, -10, 2.25)
end_point = (5, -242, 2.25)

params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': args.port,  # connection port
    'town': 'Town07',  # which town to simulate
    # 'task_mode': 'random',  # removed
    'max_time_episode': 800,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 16,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 5,  # desired speed (m/s)
    'max_ego_spawn_times': 1,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': True,  # whether to output PIXOR observation
    'start_point': start_point,
    'end_point': end_point,
    'weather': weather,
    'ip': 'localhost'
}


class VecGymCarla:
    def __init__(self, env, action_repeat, encoder = None):
        self.env = env
        self.action_repeat = action_repeat
        self.encoder = encoder
        self.action_space = self.env.action_space
        if self.encoder:
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(16+1,), dtype=np.float)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3*128*128+1,), dtype=np.uint8)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        self.episodic_return = 0
        self.episodic_len = 0

    def step(self, action):
        action = np.clip(action, -1, 1)
        action = np.squeeze(action) * self.env.action_space.high
        cum_r = 0
        i = {'episodic_return': None}
        for _ in range(self.action_repeat):
            s,r,d,_ = self.env.step(action)
            cum_r += r
            self.episodic_return += r
            self.episodic_len += 1
            if d:
                s = self.env.reset()
                i = {'episodic_return': self.episodic_return}
                print('Done: ', self.episodic_return, self.episodic_len)
                self.episodic_return, self.episodic_len = 0, 0
                break
        s, cum_r, d, i = self.process_state(s), [cum_r], [d], [i]
        return s, cum_r, d, i

    def reset(self):
        s = self.env.reset()
        self.episodic_return = 0
        return self.process_state(s)

    def process_state(self, s):
        if self.encoder is None:
            obs = np.transpose(s['camera'], (2,0,1)).reshape(-1)
            speed = s['state'][2]
            state = np.append(obs, speed)
            state = np.expand_dims(state, axis=0)
        else:
            obs = np.transpose(s['camera'], (2,0,1))
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float().to(self.device)
            with torch.no_grad():
                obs = self.encoder(obs).cpu().squeeze().numpy()
            speed = s['state'][2]
            state = np.expand_dims(np.append(obs, speed), axis=0)
        
        return state


class Encoder(nn.Module):
    def __init__(self, latent_size = 16, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(9216, latent_size)

    def forward(self, x):
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu



def main():
    # prepare env
    encoder = None
    if args.use_encoder:
        encoder = Encoder()
        weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in encoder.state_dict().keys():
                del weights[k]
        encoder.load_state_dict(weights)

    carla_env = gym.make('carla-v0', params=params)
    env = VecGymCarla(carla_env, args.action_repeat, encoder)

    # prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.use_encoder:
        Model = CarlaLatentPolicy
        input_dim = args.latent_size+1  # 16+1 in paper
    else:
        Model = CarlaImgPolicy
        input_dim = args.latent_size+1  # 128+1 in paper (16 is too small)
    model = Model(input_dim, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model = model.to(device)

    res = []
    state = env.reset()
    while(len(res) < args.num_eval):
        action, _, _, _ = model(torch.from_numpy(state).float().to(device))
        state, _, done, info = env.step(action.cpu().numpy())
        for i in info:
            if i['episodic_return'] is not None:
                res.append(i['episodic_return'])
                print(i['episodic_return'])

    print("Average Score", np.mean(res))

if __name__ == '__main__':
    main()
