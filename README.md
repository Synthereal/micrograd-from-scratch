# micrograd from scratch

An Automatic Gradient Engine (Autograd) that implements backpropagation (reverse-mode autodifferentiation) over a Directed Acyclic Graph (DAG), and a small API based on PyTorch.

Based on the tutorial from Andrej Karpathy: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=iX6GS6R8l2tItcOI)

### Setup

This project uses [**uv**](https://github.com/astral-sh/uv) to manage packages and virtual environments.

```bash
# Navigate to current directory
cd micrograd-from-scratch

# Sync dependencies and initialize the environment
uv sync
```

### Usage

Check out the `notebooks/` directory for detailed lecture notes and step-by-step code walkthroughs. To test the engine locally with a sample gradient descent loop, run:

```bash
uv run python src/test/test.py
```

### Features

- Scalar-valued Autograd: Supports fundamental operations including addition, multiplication, power, and activation functions like tanh and exp.
- Neural Network Library: Provides `Neuron`, `Layer`, and `MLP` classes for building and training modular neural networks.

### Project Structure

- `notebooks/micrograd_notebook.ipynb`: Notes on fundamentals of backpropagation and building an autograd engine from scratch, focusing on the Value class and manual gradient calculations.
- `notebooks/mlp_implementation.ipynb`: Builds upon the autograd engine to implement a full neural network library, covering the construction of neurons, layers, and a multi-layer perceptron (MLP) for binary classification.

- `src/micrograd/engine.py`: Core `Value` class and backpropagation logic.
- `src/micrograd/nn.py`: Neural network implementation including `Neuron`, `Layer`, and `MLP`.
- `src/test/test.py`: Local test suite for verifying model convergence.

### Bye Bye!
![Agu Seal](https://media1.tenor.com/m/bvGwM-rC37YAAAAd/aguhiyori-agu-seal.gif)
