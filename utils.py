from torch import nn
import torch
import matplotlib.pyplot as plt
	
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
		loss = criterion(out, label)

		predicted = torch.argmax(out, axis=1)

		epoch_loss += loss.item()
		total_correct += (predicted == torch.argmax(label, axis=1)).sum().item()
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
	plt.close()

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
	plt.close()

def save_multi_plot(path, dict, ylabel, title):
	plt.figure(figsize=(8, 6))
	plt.title(title)
	plt.xlabel('Outer Updates')
	plt.ylabel(ylabel)
	plt.grid(True)

	for key in dict.keys():
		list = dict[key]
		x = [elem[1] for elem in list]
		y = [elem[0] for elem in list]
		plt.plot(x,y, marker='.', linestyle='-', label=key)

	plt.legend()
	plt.savefig(path )
	plt.clf()



