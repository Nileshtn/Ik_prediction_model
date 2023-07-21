import pandas as pd
import torch
import numpy as np

# set device
device = torch.device('cuda:0')

#load model
IkPrdt = torch.load(r'weights\ikprdt_360deg_v2.pth')
IkPrdt.to(device)
IkPrdt.eval()

#expected_out = [119.25600449339308,107.3134478716153,96.68822118363587]

x = [1,1]

test_data = torch.tensor(x, dtype=torch.float32).to(device)

output = IkPrdt(test_data)
output = output.cpu().detach().numpy().tolist()

print(output)