# Simple Neural Network Classifier

This project implements a basic neural network classifier using a single neuron (perceptron) to classify 2D input data. The classifier is trained using gradient descent and the sigmoid activation function.

## Features

- Single neuron implementation
- Binary classification
- Gradient descent optimization
- Sigmoid activation function
- Loss computation using binary cross-entropy
- Accuracy evaluation

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Prityush-14/single_node_neural_network
   ```

2. Install the required packages:
   ```
   pip install numpy
   ```

## Usage

Run the script:

```
python main.py
```

The script will train the neural network for 1000 epochs and print the loss every 50 epochs, along with the final accuracy.

## Code Structure

- `Node` class: Represents a single neuron with weights and bias
- `sigmoid` function: Implements the sigmoid activation function
- `forward` function: Performs the forward pass of the neural network
- `compute_loss` function: Calculates the binary cross-entropy loss
- Main training loop: Implements gradient descent and updates the neuron's parameters

## Dataset

The code uses a small built-in dataset with 10 samples, each having 2 features. The target variable is binary (0 or 1). The dataset is generated with the help of Claude 3.5 Sonnet as well as the readme file.

## Customization

You can modify the following parameters to experiment with the model:

- `epochs`: Number of training iterations
- `lr`: Learning rate for gradient descent
- `input_x` and `input_y`: Input features and target variables

## License

This project is open-source and available under the MIT License.

## Author

Prityush Bansal