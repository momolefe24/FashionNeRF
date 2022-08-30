from config import *
import numpy as np
import torch 
import torch.nn as nn

def get_meshgrid(width,height):
    x = np.linspace(0,width-1,width)
    y = np.linspace(0,height-1,height)
    return np.meshgrid(x,y)

def get_camera_vector(width,height,zc=0.5,focal_length=0.5):
    x,y = get_meshgrid(width,height)
    x_camera = (x-width*zc)/focal_length
    y_camera = (y-height*zc)/focal_length
    ones_like = np.ones(x_camera.shape)
    camera_vector = np.stack([x_camera,-y_camera,-ones_like],axis=-1)
    return camera_vector

def extend_dimension(camera_vector):
    return camera_vector[...,None,:]


def get_rays(width,height,camera2world,focal_length=0.5):
    camera_vector = get_camera_vector(width,height,focal_length=focal_length)
    camera_vector = extend_dimension(camera_vector)
    rotation = camera2world[:3,:3]
    translation = camera2world[:3,-1]
    world_coordinates = camera_vector * rotation # (100,100,1,3) * (3,3) => (100,100,3,3)
    ray_d = np.sum(world_coordinates,axis=-1) 
    ray_d = ray_d/np.linalg.norm(ray_d,axis=-1,keepdims=True)
    ray_o = np.broadcast_to(translation,ray_d.shape)# (100,100,3)
    return ray_o,ray_d


def encode_position(x,POS_ENCODE_DIMS=16):
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [np.sin,np.cos]:
            positions.append(fn((2.0 ** i) * x))
    return np.concatenate(positions,axis=-1)

def render_flat_rays(ray_o,ray_d,near,far,nC,rand=True): #nC is number of samples
    t_vals = np.linspace(near,far,nC)
    if rand:
        noise_shape = list(ray_o.shape[:-1]) + [nC]
        noise = (np.random.uniform(size=noise_shape)* (far - near)/nC)
        t_vals = t_vals + noise
    
    rays = ray_o[...,None,:] + (ray_d[...,None,:] * t_vals[...,None])
    rays_flat = np.reshape(rays,[-1,3]) # (10000,3)
    rays_flat = encode_position(rays_flat)
    return (rays_flat,t_vals)


def map_fn(pose,focal_length,image_width,image_height,near,far,nC,rand=True):
    ray_o,ray_d = get_rays(image_width,image_height,pose,focal_length=focal_length)
    rays_flat,t_vals = render_flat_rays(ray_o,ray_d,near,far,nC,rand=rand)
    return rays_flat,t_vals



def render_rgb_depth(model,rays_flat,t_vals,device,rand=True,train=True):
    predictions = model(rays_flat)
    predictions = predictions.reshape((BATCH_SIZE, H, W, NUM_SAMPLES, 4))
    
    # Slice the predictions into rgb and sigma
    rgb = torch.sigmoid(predictions[..., :-1])
    sigma_a = nn.ReLU()(predictions[..., -1])
    
    # get the distance of adjacent intervals
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    if rand:
        delta = torch.cat([delta,torch.broadcast_to(torch.tensor([1e10]),(BATCH_SIZE,H,W,1)).to(device)],axis=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta)
    else:
        delta = torch.cat([delta,torch.broadcast_to(torch.tensor([1e10]),(BATCH_SIZE,1)).to(device)],axis=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta[:,None,None,:])
    
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon,-1)
    weights = alpha * transmittance
    rgb = torch.sum(weights[...,None] * rgb,axis=-2)
    
    if rand:
        depth_map = torch.sum(weights * t_vals,axis=-1)
    else:
        depth_map = torch.sum(weights * t_vals[:,None,None],axis=-1)
    return rgb,depth_map   

def save_checkpoint(state):
    print("=> Saving checkpoint")
    torch.save(state,checkpoint)
    
def load_checkpoint(model,optimizer,checkpoint_state):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint_state['state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])

