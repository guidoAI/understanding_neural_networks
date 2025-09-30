import torch
import numpy as np
from matplotlib import pyplot as plt

# create an MLP neural network class with one hidden layer
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        self.a = self.hidden(x)
        self.h = self.activation(self.a)
        x = self.output(self.h)
        return x
    
# create a dataset
def create_dataset(n_samples=100, range=3):
    x = np.linspace(-range, range, n_samples)
    y = x**2
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

# train the model
def train_model(model, x_train, y_train, n_epochs=1000, learning_rate=0.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return model

x,y = create_dataset()
model = MLP(input_size=1, hidden_size=10, output_size=1)
model = train_model(model, x, y)

model.eval()
# first run it on the training data set:
with torch.no_grad():
    y_train_pred = model(x).numpy()

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), 'b.', label='Original data')
plt.plot(x.numpy(), y_train_pred, 'r-', label='Model prediction')
plt.title('MLP Regression on Quadratic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

xx, yy = create_dataset(n_samples=1000, range=10)
with torch.no_grad():
    y_pred = model(xx).numpy()
    # get the hidden activations and outputs
    hidden_activations = model.a.numpy()
    hidden_outputs = model.h.numpy()

# get the weights and biases of the output layer:
weights = model.output.weight.data.numpy()
bias = model.output.bias.data.numpy()
    
plt.figure(figsize=(12, 8))
# plot the original data and the model's predictions
plt.plot(x.numpy(), y.numpy(), 'b.', label='Original data')
plt.plot(xx.numpy(), y_pred, 'r-', label='Model prediction')
# show per hidden neuron the hidden outputs:
for i in range(hidden_outputs.shape[1]):
    plt.plot(xx.numpy(), weights[0][i] * hidden_outputs[:, i], '--', label=f'Neuron {i+1} output')
plt.plot(xx.numpy(), bias * np.ones_like(xx.numpy()), 'k:', label='Bias term')
plt.title('MLP Regression with Hidden Neuron Outputs')
plt.xlabel('x')
plt.ylabel('y / Neuron Outputs')
plt.legend()
plt.grid()
plt.show()

print('Done')
