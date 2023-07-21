import pandas as pd
import torch
from torch import nn, optim
import numpy as np
import bpy
import math



#input area
position = [-1.5,-1]


joint_01 = bpy.data.objects["Empty"]
joint_02 = bpy.data.objects["Empty.001"]
joint_03 = bpy.data.objects["Empty.002"]

device = torch.device('cuda:0')

IkPrdt = torch.load(r"C:\Users\Nilesh\Desktop\blender_models\weights\ikprdt_360deg_v2.pth")
IkPrdt.to(device)

position = torch.tensor(position, dtype=torch.float32).to(device)
rotation = IkPrdt(position)
rotation = rotation.cpu().detach().numpy().tolist()


joint_01.rotation_euler[1] = math.radians(90 - rotation[0])
joint_02.rotation_euler[1] = math.radians(180 - rotation[1])
joint_03.rotation_euler[1] = math.radians(180 - rotation[2])