import os 
print("config: ", os.getcwd())
from experiments import *

# from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim


W = width = image_width = 100
H = height = image_height = 100 
near=2.0
far=6.0
NUM_SAMPLES = nC = 12
BATCH_SIZE = 5
POS_ENCODE_DIMS = 16 # 
EPOCHS = 100
rand=True
num_layers=8
num_pos = H * W * NUM_SAMPLES # 320000
encode_dims = 16


transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1),
    transforms.Resize((100,100),antialias=True)
])


# writer_real = SummaryWriter(save_files['nerf_real_dir'])
# writer_nerf = SummaryWriter(save_files['nerf_model_dir'])
checkpoint = f"{save_files['weights_dir']}/{file['experiment_facts']['checkpoints']['checkpoint_nerf']}"