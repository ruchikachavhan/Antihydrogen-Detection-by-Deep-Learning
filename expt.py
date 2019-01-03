import os
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def return_tensor(X, i):
	z=np.array([X[i:1000+i]])
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
		self.conv1= nn.Conv1d(1000,8, 7, stride=1)
		self.conv2= nn.Conv1d(8, 16, 3, stride=1)
		self.conv3= nn.Conv1d(16, 32, 3, stride=1)
		self.conv4= nn.Conv1d(32, 64, 3, stride=1)
		self.conv5= nn.Conv1d(64, 128, 3, stride=1)
		self.conv6= nn.Conv1d(128, 256, 3, stride=1)
		self.conv7= nn.Conv1d(256, 512, 3, stride=1)
		

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
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv6(x))
		x = F.max_pool1d(x, 2)
		x = F.relu(self.conv7(x))
		x = F.max_pool1d(x, 2)
		x = x.view(1,-1)
		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(3072, 2000)
		self.fc2 = nn.Linear(2000, 640)
		self.fc3 = nn.Linear(640, 300)
		self.fc4 = nn.Linear(300, 1000)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

conv_net=ConvNet().to(device)
forward_net= Net().to(device)
params1 = list(conv_net.parameters())
conv_net.zero_grad()
params2 = list(forward_net.parameters())
forward_net.zero_grad()
params_total= np.append(params1, params2)
optimizer = optim.SGD(params_total, lr=0.1)

def train(Z, Phi, target):
	for iterations in range(0, 2001):
		avg_loss=0
		for i in range(0, Z.shape[0], 1000):
			z= return_tensor(Z, i)
			phi= return_tensor(Phi, i)
			Conv_Net_output_Z= conv_net.forward(z)
			Conv_Net_output_Phi= conv_net.forward(phi)
			FirstLayer= torch.cat((Conv_Net_output_Z, Conv_Net_output_Phi), 1)
			output= forward_net.forward(FirstLayer)
			criteria= nn.MSELoss()
			l=np.array([target[i:1000+i]])
			l= torch.from_numpy(l[0])
			l = l.float()
			l= l.t()
			l= l.to(device)
			loss= criteria(output, l)
			avg_loss= avg_loss+loss
			optimizer.zero_grad() 
			loss.backward()
			optimizer.step() 
		print(iterations+1, avg_loss.item())
		if(iterations%1000 ==0):
			name_conv= "model_conv_"+ str(iterations)+ ".pt"
			name_forward = "model_forward_"+ str(iterations)+ ".pt"
			torch.save(conv_net, "/home/sine/ruchika/"+ name_conv )
			torch.save(forward_net, "/home/sine/ruchika/"+ name_forward )

#Reading data from csv file and dividing into inner and outer channels 
data = np.genfromtxt("antihydrogen_1-10000.csv")
#Ground truth 
target= data[:,:1]
# target= np.transpose(target)
print("target", target.shape)
InnerZ= data[:,1:448]
OuterZ= data[:,694:1141]
InnerPhi= data[:,448:694]
OuterPhi= data[:,1141:1431]
Z= np.append(InnerZ, OuterZ, axis=1)
Phi= np.append(InnerPhi, OuterPhi, axis=1)

train(Z, Phi, target)


