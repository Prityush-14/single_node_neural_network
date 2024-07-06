import math
import numpy as np

class Node:
    def __init__(self):
        self.w = np.random.randn(2) * 0.01
        self.b = 0

epochs = 1000
lr = 0.01

input_x = np.array([
    [2, 2.5],
    [8, 3.2],
    [5, 3.0],
    [1, 2.2],
    [3, 2.8],
    [7, 3.5],
    [6, 3.1],
    [4, 2.7],
    [9, 3.8],
    [3, 2.6]
])

input_y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

node = Node()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X):
    return sigmoid(np.dot(X, node.w) + node.b)

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

for i in range(epochs):
    # Forward pass
    y_pred = forward(input_x)
    
    # Compute loss
    loss = compute_loss(input_y, y_pred)
    
    # Backward pass
    dz = y_pred - input_y
    dw = np.dot(input_x.T, dz) / len(input_x)
    db = np.mean(dz)
    
    # Update parameters
    node.w -= lr * dw
    node.b -= lr * db
    
    if (i + 1) % 50 == 0:
        print(f'Epoch is {i+1}')
        print(f'Loss is {loss}')
    
    # Evaluation
    predictions = (forward(input_x) > 0.5).astype(int)
    accuracy = np.mean(predictions == input_y) * 100
    print(f"Accuracy: {accuracy}%")