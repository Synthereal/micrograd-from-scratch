# see mlp_implementation.ipynb for original code and full notes

import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    # sum(w * x) + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self):
     return f"Neuron({len(self.w)})"
  

# A layer is a list of neurons that which all neurons connect to all inputs
# i.e. a set of nuerons evaluated independently
class Layer:

  def __init__(self, nin, nout, **kwargs):
    # nin = number of inputs from previous layer
    # nout = number of neurons in this layer
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    params = []
    for neuron in self.neurons:
      ps = neuron.parameters()
      params.extend(ps)
    return params
  
  def __repr__(self):
     return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
  

# MLP: Multi Layer Perceptron
class MLP:

  def __init__(self, nin, nouts):
    # nouts = list of nout
    sz = [nin] + nouts # number of neurons per layer
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    # calls these layers sequentially
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
     return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"