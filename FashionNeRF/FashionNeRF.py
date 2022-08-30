import os
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
# Hello
from config import *
from datasets import FashionDataset, FashionPipeline
from utils import *
from model import *
import torch
import matplotlib.pyplot as plts
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader,ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# argparse
model_3d = "dennis3"
transform = transforms.ToTensor()
train_dataset = FashionDataset(model=model_3d,data="test",type="depth")
train_dataset.__getitem__(0)
train_pipeline = FashionPipeline(train_dataset,nC=NUM_SAMPLES,near=near,far=far,rand=True)
train_pipeline.__getitem__(0)
train_pipeline_loader = DataLoader(train_pipeline,batch_size=BATCH_SIZE,shuffle=True)
image,rays_flat,t_vals = iter(train_pipeline_loader).next()
torch.cuda.empty_cache()

device = 'cpu'
net = Net(num_layers,num_pos,encode_dims).to(device)
image,rays_flat,t_vals = iter(train_pipeline_loader).next()
rays_flat = rays_flat.to(device,torch.float32)
t_vals = t_vals.to(device,torch.float32)
image = image.to(device)

#  tensorboard --logdir=tensorboard 
step = 0
LEARNING_RATE = 2e-4
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
net.train()
do_train = True
if do_train:
    for epoch in range(EPOCHS):
        running_loss = 0
        for batch_idx,data in enumerate(train_pipeline_loader):
            image,rays_flat,t_vals = data
            rays_flat = rays_flat.to(device,torch.float32)
            t_vals = t_vals.to(device,torch.float32)
            image = image.to(device)
            rgb,_ = render_rgb_depth(net,rays_flat,t_vals,device,rand=rand,train=True)
            optimizer.zero_grad()
            loss = criterion(image,rgb.permute(0,3,1,2))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx%2 == 0:
                print(f"[{epoch+1},{batch_idx+1:5d}] loss: {running_loss/85:.5f}")
                with torch.no_grad():
                    rgb, _ = render_rgb_depth(net,rays_flat,t_vals,device,rand=rand,train=True)
                    img_grid_real = torchvision.utils.make_grid(image,normalize=True)
                    img_grid_rgb = torchvision.utils.make_grid(rgb.permute(0,3,1,2)[:4],normalize=True)
                    # writer_nerf.add_scalar('training loss',loss,epoch*len(train_pipeline_loader)+batch_idx)
                    # writer_real.add_image('Real',img_grid_real,global_step=step)
                    # writer_nerf.add_image('Nerf',img_grid_rgb,global_step=step)
                step += 1
    checkpoint_state = {'state_dict':net.state_dict(),'optimizer':optimizer.state_dict()}    
    save_checkpoint(checkpoint_state)
    # To load: checkpoint_state = torch.load(checkpoint)
    print("Finished traning")