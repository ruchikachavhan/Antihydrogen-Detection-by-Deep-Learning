import os
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def return_tensor(X, i):
	X_= np.expand_dims([X], axis=0)
	z=np.array([X_[0][i:100+i]])
	z= torch.from_numpy(z)
	z = z.float()
	z = z.to(device)
	return z

use_cuda = torch.cuda.is_available()
# use_cuda = 0
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__() 
		self.conv1= nn.Conv1d(100,8, 7, stride=2)
		self.conv2= nn.Conv1d(8, 16, 3, stride=2)
		self.conv3= nn.Conv1d(16, 32, 3, stride=2)
		self.conv4= nn.Conv1d(32, 64, 3, stride=2)
		self.conv5= nn.Conv1d(64, 128, 3, stride=2)
		

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
		return x

class Net(nn.Module):
	def __init__(self):
	super(Net, self).__init__()
	self.fc1 = nn.Linear(256, 64)
	self.fc2 = nn.Linear(64, 32)
	self.fc3 = nn.Linear(32, 100)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		
conv_net=ConvNet().to(device)
forward_net= Net().to(device)
params1 = list(conv_net.parameters())
conv_net.zero_grad()
params2 = list(forward_net.parameters())
forward_net.zero_grad()
params_total= []
params_total.append(params1)
params_total.append(params2)

#Reading data from csv file and dividing into inner and outer channels 
data = np.genfromtxt("antihydrogen_1-10000.csv")
#Ground truth 
target= data[:,:1]
InnerZ= data[:,1:448]
OuterZ= data[:,694:1141]
InnerPhi= data[:,448:694]
OuterPhi= data[:,1141:1431]
Z= np.append(InnerZ, OuterZ, axis=1)
Phi= np.append(InnerPhi, OuterPhi, axis=1)


optimizer = optim.Adam(params_total, lr=0.0001)
def train(Z, Phi, target):
	for iterations in range(0, 10000):
	avg_loss=0
	for i in range(0, Z_.shape[1], 100):
		z= return_tensor(Z, i)
		phi= return_tensor(Phi, i)
		Conv_Net_output_Z= ConvNet.forward(z)
		Conv_Net_output_Phi= ConvNet.forward(phi)
		FirstLayer= np.append(Conv_Net_output_Z, Conv_Net_output_Phi, axis =1)
		output= forward_net.forward(FirstLayer)
		criteria= nn.MSELoss()
		label= return_tensor(target, i)
		loss= criteria(output, label)
		avg_loss= avg_loss+loss
		optimizer.zero_grad() 
		loss.backward()
		optimizer.step() 
	print(iterations+1, avg_loss.item())
	if(iterations%10 ==0):
		name= "model_"+ str(iterations)+ ".pt"
		torch.save(net, "/home/sine/ruchika/"+ name)
