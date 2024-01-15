from torch import nn
import torch
import matplotlib.pyplot as plt

class SmallNet(nn.Module):
	def __init__(self):
		super(SmallNet, self).__init__()
		# Try (384,512)

		self.relu = nn.ReLU()
		
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=(3, 3))
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3, 3))
		self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))
		self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 3))
		self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3, 3))
		self.maxpool5 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3, 3))
		self.maxpool6 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.conv7 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3, 3))
		self.maxpool7 = nn.MaxPool2d(kernel_size=(2, 2))
		
		self.fc1 = nn.Linear(in_features=512, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=128)
		self.fc3 = nn.Linear(in_features=128, out_features=1)

		self.m = nn.Sigmoid()
		
	def forward(self,x):
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		
		x = self.conv4(x)
		x = self.relu(x)
		x = self.maxpool4(x)
		
		x = self.conv5(x)
		x = self.relu(x)
		x = self.maxpool5(x)
		
		x = self.conv6(x)
		x = self.relu(x)
		x = self.maxpool6(x)
		
		x = self.conv7(x)
		x = self.relu(x)
		x = self.maxpool7(x)
		
		x = torch.flatten(x, start_dim=1)

		print(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return torch.flatten(self.m(x))
	
def test(testloader, model, criterion, device):
	model.eval()
	epoch_loss = 0
	total_correct = 0
	total_samples = 0
	num_batch_count = 0

	for (image, label) in testloader:
		image = image.to(device)
		label = label.to(device)

		out = model(image)
		loss = criterion(out, label.float())

		predicted = torch.round(out)

		epoch_loss += loss.item()
		total_correct += (predicted == label).sum().item()
		total_samples += label.size(0)
		num_batch_count +=1

	loss = epoch_loss / num_batch_count
	acc = total_correct / total_samples * 100
	model.train()
	return loss, acc

def save_loss(path, train_loss, test_loss):

	plt.figure(figsize=(8, 6))
	plt.plot(list(range(len(train_loss))), test_loss, marker='.', linestyle='-', label='Test Loss')
	plt.title('Test and Training Loss Curves')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.plot(list(range(len(train_loss))), train_loss, marker='.', linestyle='-', label='Training Loss')
	plt.grid(True)
	plt.legend() 
	plt.savefig(path + '/loss_curves.png')
	plt.clf()

def save_acc(path, train_acc, test_acc):
	plt.figure(figsize=(8, 6))
	plt.plot(list(range(len(test_acc))), test_acc, marker='.', linestyle='-', label='Validation Accuracy')
	plt.title('Validation and Training Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.plot(list(range(len(train_acc))), train_acc, marker='.', linestyle='-', label='Training Accuracy')
	plt.grid(True)
	plt.legend() 
	plt.savefig(path + '/acc_curves.png')

