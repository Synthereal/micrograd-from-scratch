# see mlp_implementation.ipynb for original code and full notes

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import cast
from micrograd.engine import Value
from micrograd.nn import MLP


# Initialize MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1]) # 3 inputs, [2 layers of 4, 1 output]
n(x)

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
] # inputs
ys = [1.0, -1.0, -1.0, 1.0] # desired targets


# Gradient Descent Loop

lr = 0.05 # learning rate

for k in range(200):

  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) # type: ignore

  # backward pass
  for p in n.parameters(): # ALWAYS ZERO YOUR GRADIENTS
    p.grad = 0.0 # reset to constructor
  loss.backward() # type: ignore

  # update
  for p in n.parameters():
    p.data += -lr * p.grad

  # print output
  print("step:", k + 1, " loss:", loss.data) # type: ignore


# Final Verification
print("\nExpected:")
for val in ys:
    print(f"Target: {val}")

print("\nFinal Predictions:")
for val in [n(x) for x in xs]:
    print(f"Target: {val.data:.4f}") # type: ignore