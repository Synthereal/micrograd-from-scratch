````markdown
# micrograd

---

An Automatic Gradient Engine (Autograd) that implements backpropagation (reverse-mode autodifferentiation) over a Directed Acyclic Graph (DAG), and a small API based on PyTorch.

Based on the tutorial from Andrej Karpathy: ["The spelled-out intro to neural networks and backpropagation: building micrograd"](https://youtu.be/VMj-3S1tku0?si=iX6GS6R8l2tItcOI)

## 🚀 Features

- **Scalar-valued Autograd**: Supports fundamental operations including addition, multiplication, power, and activation functions like tanh and exp.
- **Neural Network Library**: Provides `Neuron`, `Layer`, and `MLP` classes for building and training modular neural networks.
- **Topological Sorting**: Automatically handles the ordering of operations to ensure gradients flow correctly during the backward pass.

## 🛠 Setup & Installation

This project uses [**uv**](https://github.com/astral-sh/uv) to manage packages and virtual environments.

```bash
# Clone the repository
git clone [https://github.com/your-username/micrograd-from-scratch.git](https://github.com/your-username/micrograd-from-scratch.git)
cd micrograd-from-scratch

# Sync dependencies and initialize the environment
uv sync
```
````

## 💻 Usage

Checkout the `notebooks/` directory for detailed lecture notes and step-by-step code walkthroughs. To test the engine locally with a sample gradient descent loop, run:

```bash
uv run python src/test/test.py

```

## 📁 Project Structure

- `src/micrograd/engine.py`: The core `Value` class and backpropagation logic.
- `src/micrograd/nn.py`: Neural network implementation including `Neuron`, `Layer`, and `MLP`.
- `src/test/test.py`: Local test suite for verifying model convergence.

```

Would you like me to add a "Mathematical Background" section to the README explaining the chain rule logic used in your `_backward` functions?

```
