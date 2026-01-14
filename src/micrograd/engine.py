# see mlp_implementation.ipynb for original code and full notes

import math

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    # automatic backpropagation
    # leaf node has no function (i.e. None)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self): # repr = representation
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other) # handle non value cases
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      # local derivative * global derivative
      # global derivative = derivative of final output with respect to out data 
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad

      # set to += to composite gradients
      # when self and other are same object, = will override grad value
      # instead, we accumulate the gradients starting from base of 0

    out._backward = _backward
    return out
  
  def __radd__(self, other): # other + self
    return self + other
  
  def __sub__(self, other): # self - other
    return self + (-other)
  
  def __rsub__(self, other): # other - self
    return other + (-self)
  
  def __neg__(self): # -self
    return self * -1
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out
  
  def __rmul__(self, other): # other * self
    return self * other
  
  def __truediv__(self, other): # self / other
    # a / b
    # a * (1 / b)
    # a * (b ** -1)
    return self * other**-1
  
  def __rtruediv__(self, other): # other / self
    return other * self**-1
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supports in/float powers"
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      # power rule * global grad
      self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out
  
  def tanh(self):
    # composite operation
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward
    return out
  
  def backward(self):
    # automatic backpropagation

    # sort graph into list
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    # find backwards gradients in sorted order    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()