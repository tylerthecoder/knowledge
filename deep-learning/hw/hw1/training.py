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
	labels = load_tensor(f'./data/{type}_labels.csv')
	return data, labels

training_data, training_labels = load_data('training')
validation_data, validation_labels = load_data('validation')
testing_data, testing_labels = load_data('testing')

figure, axis = plt.subplots(5, 2)


class MLP(torch.nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(2, 3),
			nn.ReLU(),
			nn.Linear(3, 2),
		)
	def forward(self, x):
		return self.layers(x)


print("LP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 10 epochs.")
model = MLP()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.eval()
y_pred = model(testing_data)
testing_accuracy = (y_pred.argmax(1) == testing_labels).float().mean()
print('Test accuracy before:' , testing_accuracy.item())

losses = np.array([])
accuracies = np.array([])


for epoch in range(0, 10): # 5 epochs at maximum
	model.train()

	# Print epoch
	print(f'Starting epoch {epoch+1}')

	outputs = model(training_data)

	loss = loss_function(outputs, training_labels.long())

	loss.backward()

	optimizer.step()

	model.eval()

	validation_outputs = model(validation_data)

	validation_accuracy = (validation_outputs.argmax(1) == validation_labels).float().mean()

	losses = np.append(losses, loss.item())
	accuracies = np.append(accuracies, validation_accuracy.item())

	print(f'Epoch {epoch+1, 2}, loss = {round(loss.item(), 3)}, validation accuracy = {round(validation_accuracy.item(), 2)}')

axis[0][0].plot(losses)
axis[0][0].set_title("Q1 training loss")

axis[0][1].plot(accuracies)
axis[0][1].set_title("Q1 validation accuracy")


plt.show()


model.eval()
y_pred = model(testing_data)
testing_accuracy = (y_pred.argmax(1) == testing_labels).float().mean()
print('Test accuracy after:' , testing_accuracy.item())




