import os
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()
# use_cuda = 0
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#Reading data from csv file and dividing into inner and outer channels 
data = np.genfromtxt("antihydrogen_1-10000.csv")
#Ground truth 
target= data[:,:1]
target= np.expand_dims(target, axis=0)
target= torch.from_numpy(target)
target = target.float()
print(target[0][0].shape)

# For axial (z)
InnerZ= data[:,1:448]
OuterZ= data[:,694:1141]
# For azimuthal Phi
InnerPhi= data[:,448:694]
OuterPhi= data[:,1141:1431]
#concatenating the two arrays
Z= np.append(InnerZ, OuterZ, axis=1)
Phi= np.append(InnerPhi, OuterPhi, axis=1)
#converting to torch tensor

Z_= np.array([Z])

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__() 
		self.conv1= nn.Conv1d(100,8, 7, stride=2)
		self.conv2= nn.Conv1d(8, 16, 3, stride=2)
		self.conv3= nn.Conv1d(16, 32, 3, stride=2)
		self.conv4= nn.Conv1d(32, 64, 3, stride=2)
		self.conv5= nn.Conv1d(64, 128, 3, stride=2)
		self.fc1 = nn.Linear(128, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 100)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv3(x))
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv4(x))
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv5(x))
		x = x.view(1,-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net=Net().to(device)
params = list(net.parameters())
net.zero_grad()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
for iterations in range(0, 10000):
	avg_loss=0
	for i in range(0, Z_.shape[1], 100):
		z=np.array([Z_[0][i:100+i]])
		# z= np.array([z])
		z= torch.from_numpy(z)
		z = z.float()
		z = z.to(device)
		output= net.forward(z)
		criteria= nn.MSELoss()
		target_= np.array(target[0][i:100+i])
		target_= np.transpose(target_)
		label= torch.from_numpy(target_)
		label= label.to(device)
		loss= criteria(output, label)
		avg_loss= avg_loss+loss
		optimizer.zero_grad() 
		loss.backward()
		optimizer.step() 
	print(iterations+1, avg_loss.item())
	if(iterations%10 ==0):
		torch.save(net, "/home/sine/ruchika/model.pt")