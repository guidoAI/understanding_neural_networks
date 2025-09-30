import numpy as np
import torch
from matplotlib import pyplot as plt

# generate a multiplication dataset:
def generate_dataset(graphics = False, range = 10.0, step = 1.0):
    # make a torch range from -10 to 10 with step size 1.0:
    x1 = torch.arange(-range, range+1.0, step)
    x2 = torch.arange(-range, range+1.0, step)

    # create a meshgrid from x1 and x2:
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')

    # flatten x1 and x2:
    x1 = x1.flatten()
    x2 = x2.flatten()

    # stack x1 and x2 to create a 2D input tensor:
    X = torch.stack((x1, x2), dim=1)

    # compute the multiplication of x1 and x2 to create the target tensor:
    t = (x1 * x2).unsqueeze(1)

    # convert to numpy arrays:
    x = X.numpy()
    y = t.numpy()

    if graphics:
        # show a three-dimensiontal surface plot of the dataset:
        fig = plt.figure()
        plt.title('Multiplication Dataset')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        plt.show()

    return X, t, x, y

if __name__ == '__main__':
    X, t, x, y = generate_dataset(graphics=True, range = 10.0, step = 1.0)

    # Create a simple neural network model
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5000

    # Training loop
    loss = []
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()

        outputs = model(X)
        
        l = criterion(outputs, t)
        l.backward()
        optimizer.step()

        loss.append(l.item())
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {l.item():.4f}')
    
    # Plot the loss curve
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()


    # Evaluate the model outside of the training range, and in between training points:
    X_test, t_test, x_test, y_test = generate_dataset(graphics=False, range = 10.0, step = 0.5)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    # Plot the predictions vs the true values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c='k', marker='o', label='True Values')
    ax.scatter(x_test[:, 0], x_test[:, 1], predictions, c='r', marker='^', label='Predictions')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.legend()
    plt.show()


    # Evaluate the model outside of the training range, and in between training points:
    X_test, t_test, x_test, y_test = generate_dataset(graphics=False, range = 20.0, step = 0.5)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    # Plot the predictions vs the true values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c='k', marker='o', label='True Values')
    ax.scatter(x_test[:, 0], x_test[:, 1], predictions, c='r', marker='^', label='Predictions')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.legend()
    plt.show()

    print("Done")