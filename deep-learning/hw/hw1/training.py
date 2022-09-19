import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# Load the data from the csv files


def load_tensor(filename: str):
	arr = np.genfromtxt(filename, delimiter=',')
	return torch.from_numpy(arr).float()

def load_data(type: str):
	data = load_tensor(f'./data/{type}_data.csv')
	labels = load_tensor(f'./data/{type}_labels.csv').float()
	return data, labels

training_data, training_labels = load_data('training')
validation_data, validation_labels = load_data('validation')
testing_data, testing_labels = load_data('testing')

figure, axis = plt.subplots(5, 2)

print("Training data", training_data.size())

class MLP2(torch.nn.Module):
	def __init__(self):
		super(MLP2, self).__init__()
		self.layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(2, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Sigmoid(),
		)


		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=1.0)
			if module.bias is not None:
				module.bias.data.zero_()

	def forward(self, x):
		return self.layers(x)

class MLP(torch.nn.Module):
	def __init__(self, hidden = 3):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(2, hidden),
			nn.ReLU(),
			nn.Linear(hidden, 1),
			nn.Sigmoid(),
		)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=1.0)
			if module.bias is not None:
				module.bias.data.zero_()

	def forward(self, x):
		return self.layers(x)

class MLP3(torch.nn.Module):
	def __init__(self):
		super(MLP3, self).__init__()
		self.layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(2, 32),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(32, 1),
			nn.Sigmoid(),
		)


		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=1.0)
			if module.bias is not None:
				module.bias.data.zero_()

	def forward(self, x):
		return self.layers(x)


def compute_accuracy(pred: torch.Tensor, labels: torch.Tensor):
	num_correct = pred.squeeze(1).round().eq(labels).sum().item()
	return num_correct / labels.size(0)

def train(model, loss_fn, opt_fn, epochs, plot_row: int):
	optimizer = opt_fn(model.parameters(), lr = 0.01)

	model.eval()

	y_pred = model(validation_data)
	accuracy = compute_accuracy(y_pred, validation_labels)

	print('Test accuracy before:' , accuracy)

	losses = np.array([])
	accuracies = np.array([])

	for epoch in range(epochs):
		model.train()

		outputs: torch.Tensor = model(training_data)

		rounded_outputs = outputs.squeeze(1)

		loss = loss_fn(rounded_outputs, training_labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		validation_outputs = model(validation_data)
		accuracy = compute_accuracy(validation_outputs, validation_labels)

		losses = np.append(losses, loss.item())
		accuracies = np.append(accuracies, accuracy)

		print(f'Epoch {epoch+1, 2}, loss = {round(loss.item(), 3)}, validation accuracy = {round(accuracy, 2)}')

	axis[plot_row][0].plot(losses)
	axis[plot_row][0].set_title("Training loss")

	axis[plot_row][1].plot(accuracies)
	axis[plot_row][1].set_title("Validation accuracy")

	model.eval()
	y_pred = model(testing_data)
	accuracy = compute_accuracy(y_pred, testing_labels)
	print('Test accuracy after:' , accuracy)



print("MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 10 epochs.")
model1 = MLP()
train(model1, nn.MSELoss(), torch.optim.SGD, 10, 0)

print("MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 500 epochs.")
model2 = MLP()
train(model2, nn.MSELoss(), torch.optim.SGD, 500, 1)

print("MLP with one hidden layer, 128 perceptrons, BCloss, Adam optimizer, no dropout, 100 epochs.")
model3 = MLP(hidden=128)
train(model3, nn.BCELoss(), torch.optim.Adam, 100, 2)

print("MLP with three hidden layers, (32, 64, 32) perceptrons, Cross Entropy loss, Adam optimizer, no dropout, 100 epochs.")
model4 = MLP2()
train(model4, nn.BCELoss(), torch.optim.Adam, 100, 3)

print("MLP with three hidden layers, (32, 64, 32) perceptrons, Cross Entropy loss, Adam optimizer, 20% dropout, 100 epochs.")
model5 = MLP3()
train(model5, nn.BCELoss(), torch.optim.Adam, 100, 4)

plt.show()

