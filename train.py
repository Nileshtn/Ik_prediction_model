import pandas as pd
import torch
from torch import nn, optim
import numpy as np

#data prepration

data_df= pd.read_csv(r"ik_data\ik_data.csv",index_col=False)
location_lable = data_df.iloc[:,1:3]
angle_input = data_df.iloc[:,3:]

input_array = location_lable.values
lable_array = angle_input.values
device = torch.device('cuda:0')
#just converting it to tensor
input_tensor= torch.tensor(input_array, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(lable_array, dtype=torch.float32).to(device)

dataset = torch.utils.data.TensorDataset(input_tensor, labels_tensor)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=450, shuffle=True)



#building the network
IkPrdt = nn.Sequential(nn.Linear(2,20),
                       nn.ReLU(),
                       nn.Linear(20,80),
                       nn.ReLU(),
                       nn.Linear(80,60),
                       nn.ReLU(),
                       nn.Linear(60,30),
                       nn.ReLU(),
                       nn.Linear(30,3)
                      )

criterion = nn.HuberLoss()
optimizer = optim.SGD(IkPrdt.parameters(), lr=0.01)
#uncommand if you want to use the trained model
#IkPrdt = torch.load(r"C:\Users\Nilesh\Desktop\blender_models\weights\ikprdt_360deg_v3.pth")
IkPrdt.to(device)


#training loop
count = 0
try:
    while True:
        running_loss = 0
        for inputs_tensor, labels_tensor in train_loader:
            optimizer.zero_grad()

            output = IkPrdt(inputs_tensor)
            loss  = criterion(output, labels_tensor)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            print (running_loss)
        if running_loss < 1.5:
            count +=1
            if count == 30:
                break

except KeyboardInterrupt: # too stop the training with ctrl + c
    pass


#save the model
path = r'weights\ikprdtv2.pth'

torch.save(IkPrdt, path)

print("trained and saved")







   