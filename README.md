# LUSR 

This is the official implementation of **Domain Adaptation In Reinforcement Learning Via Latent Unified State Representation**. The work is accepted by AAAI 2021.  Here is the project website https://sites.google.com/view/lusr and paper link https://arxiv.org/abs/2102.05714.

### Demo
1. Carracing games  
![carracing](assets/carracing.gif)   

2. CARLA Autonomousd Driving  
![carla](assets/carla.gif)


### Requirement
Python 3.7  
PyTorch 1.4  
CUDA 10.1  
gym-carla (from https://github.com/KarlXing/gym-carla.git)  
rlcodebase (from https://github.com/KarlXing/RLCodebase.git)  
CARLA 0.9.6 
gym (from https://github.com/KarlXing/gym.git)  
rllib 0.8.4  

### Installation 
```
conda create -n lusr python=3.7
conda activate lusr

# install gym_carla (for carla autonomous driving task)
git clone https://github.com/KarlXing/gym-carla.git
cd gym-carla
pip install -r requirements.txt
pip install -e .
cd ..

# install rlcodebase (for carla autonomous driving task)
git clone https://github.com/KarlXing/RLCodebase.git
cd RLCodebase
pip install -e .
pip install -r requirements.txt
cd ..

# install rllib  (for carracing game)
pip install ray[rllib]==0.8.4

# install gym (for carracing game)
git clone https://github.com/KarlXing/gym.git
cd gym
pip install -e .
cd ..

git clone https://github.com/KarlXing/LUSR.git
cd LUSR

```

### Data
Data used to train LUSR in carracing and carla tasks is stored in [google drive](https://drive.google.com/drive/folders/1Tx3zd_Qfk3fQAKXnuSEFHIv5vjrF2xIJ?usp=sharing). Download and unzip them to directory `LUSR/data`. 


### Examples

#### Carracing games
1. train lusr for carracing games
```
python train_lusr.py --data-dir data/carracing_data/  --data-tag car  --num-splitted 10 --beta 10 --class-latent-size 8  --content-latent-size 16  --flatten-size 1024   --num-epochs 2  
```

2. ppo training on carracing games
```
xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- python main_carracing.py  --encoder-path checkpoints/encoder.pt   --train-epochs 1000
```
The default path to save log information in rllib experiments is `~/ray_results/`. You could also customize the path by setting the environment variable `TUNE_RESULT_DIR`. For example, run `export TUNE_RESULT_DIR='./'` before running the training command above. 


3. evaluate the trained policy on carracing games  
```
xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- python evaluate_carracing.py --model-path checkpoints/policy.pt  --num-episodes 100  --env CarRacing-v0
```
Besides the original Carracing game `CarRacing-v0`, other carracing variants are named as `CarRacing2-v0, CarRacing3-v0 ... CarRacing8-v0`. `CarRacing2-v0` to `CarRacing7-v0` have different visual observations compared with `CarRacing-v0`. `CarRacing8-v0` has a limited number of tracks and was evaluated for track generalization.


#### CARLA autonomous driving

1. train lusr for carla autonoumous driving
```
python train_lusr.py --data-dir data/carla_data/  --data-tag weather  --num-splitted 1  --beta 10 --class-latent-size 8  --content-latent-size 16  --flatten-size 9216   --num-epochs 50  --carla-model  
```

2. ppo training on carla autonomous driving  

We need an active CARLA server to run ppo training on carla driving task. Please refer https://github.com/KarlXing/gym-carla/blob/master/README.md to install and lanuch the CARLA server. Besides that, we used town07 as the map. So please go to https://github.com/carla-simulator/carla/releases/tag/0.9.6 to download the map of town07 first and refer https://carla.readthedocs.io/en/latest/tuto_A_create_standalone/ to import town07 into the CARLA package.

Once the environment is ready, you can run
```
xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- python main_carla.py  --weather 1 --use-encoder --encoder-path checkpoints/encoder.pt --max-steps 50000
```

3. evaluate the trained policy on CARLA driving tasks
```
xvfb-run -a -s '-screen 0 1400x900x24 +extension RANDR' -- python evaluate_carla.py  --weather 1  --model-path checkpoints/policy.pt  --use-encoder  --encoder-path checkpoints/encoder.pt  --num-eval 10
```
You can set weather as 0,1 or 2 to evaluate the domain apdatation performance.



### Special Thanks
Thank Inhan Kim for pointing out a typo in equation 3. It has been fixed in the latest arxiv version of the paper.
