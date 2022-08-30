from config import *
import os
import json
from skimage import io
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from typing import List, Tuple, Dict, Any
from utils import *

class FashionDataset(Dataset):
    def __init__(self,model:str="dennis",data:str="train",type="image"): # type = ["image","normal"."depth"]
        dataset_folder = file['dataset_facts']['folder_path']
        self.model = model
        self.root_dir = f"{dataset_folder}/{self.model}"
        self.image_dir = f"{self.root_dir}/{data}"
        self.data = data
        self.type = type
        self.size = len(os.listdir(self.image_dir)) // 3 if self.data == 'test' else len(os.listdir(self.image_dir))
        if self.type == "image":
            self.image_list = [f"r_{index}.png" for index in range(self.size)]
        elif self.type == "normal":
            self.image_list = [f"r_{index}_normal_0000.png" for index in range(self.size)]
        elif self.type == "depth":
            self.image_list = [f"r_{index}_depth_0000.png" for index in range(self.size)]
        self.json_path = f"{self.root_dir}/transforms_{data}.json"
        self.json_file = json.load(open(self.json_path))
        self.camera_angle_x =  self.json_file['camera_angle_x']

    def __len__(self):
        return len(self.image_list) if len(self.image_list) != 0 else []

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.image_list[index])
        if self.type == 'depth':
            image = io.imread(img_path)[:,:,0]
        else:
            image = io.imread(img_path)
        image = transform_norm(image).permute(1,2,0)
        rotation_matrix = np.array(self.json_file['frames'][index]['transform_matrix'])
        return image,rotation_matrix
        

class FashionPipeline(Dataset):
    def __init__(self,dataset,nC=8,near=2.0,far=6.0,rand=True):
        self.images = [dataset.__getitem__(index)[0] for index in range(len(dataset))]
        self.poses = [dataset.__getitem__(index)[1] for index in range(len(dataset))]
        # self.focal_length = dataset.camera_angle_x
        self.focal_length = np.array(138.8888789)
        self.nC = nC 
        self.near = near 
        self.far = far 
        self.rand = rand

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        pose = self.poses[index]
        image = self.images[index]
        rays_flat,t_vals = map_fn(pose,self.focal_length,width,height,self.near,self.far,self.nC,rand=self.rand)
        return image,rays_flat,t_vals