import numpy as np
from time import perf_counter as pc
from sklearn.metrics import accuracy_score

response = input('Choose neuron activation type {sigmoid, tanh}')
response = response.lower()
response = response.rstrip()
img_size = 50
imageUsage = input('Choose y for images or n for simple array input\n')
if imageUsage == 'n':
  X = np.array([[1,0,1,0],[1,0,1,1],[1,0,1,0],[0,1,0,1],[0,1,1,1],[0,0,0,0],[1,1,1,1],[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,1,0],[1,0,0,1],[1,0,1,0],[1,0,1,1]])
  y = np.array([[0,0,1,0,0,0,0,1,1,1,0,1,0,0]])
else:
  X1 = np.load('neuronImageArrays.npy')
shp = X1.shape
X = np.resize(X1, (shp[0], np.prod(shp{:3])))  # 226, 2500
y1 = np.load('neuronImageLabels.npy')
y = np.where(y1 == 'Bll', 1,0 )  # Ball = 1  Box = 0
y = np.atleast_2d(y).T  # transpose to 226, 1

def sigmoid_actvn(x):
  return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
  return x * (1-x)

def tanh_actvn(x):
  return np.tanh(x)

def derivative_tanh(x):
  return 1.0 - np.tanh(x)**2

epoch - int(input('Choose number of iterations for training the neuron (5-10k)'))
lr = float(input('Choose Learning Rate for the iteration training (0.0001 - 0.9)'))
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 1
output_neurons = 1

#weight, bias
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

st = pc()
epochCount = 1
for i in range(epoch):
  #forward propagation
  hidden_layer_input1 = np.dot(X,wh) # dot product weights
  hidden_layer_input = hidden_layer_input1 + bh  # add bias
  if response == 'sigmoid':
    hidden_layer_activations = sigmoid_actvn(hidden_layer_input)
  else:
    hidden_layer_activations = tanh_actvn(hidden_layer_input)

  output_layer_input1 = np.dot(hiddenlayer_activations, wout)  # dot product weights
  output_layer_input = output_layer_input1 + bout  # add bias
  if response == 'simgoid':
    output = sigmoid_actvn(output_layer_input)
  else:
    output = tanh_actvn(output_layer_input)

  #back propagation
  E = y - output # get error from known y values
  if response == 'sigmoid':
    slope_output_layer = derivative_sigmoid(output)
    slope_hidden_layer = derivative_sigmoid(hidden_layer_activations)
  else:
    slope_output_layer = derivative_tanh(output)
    slope_hidden_layer = derivative_tanh(hidden_layer_activations

  d_output = E * slope_output_layer
  Error_at_hidden_layer = d_output.dot(wout.T)  # apply weights to output
  d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer  # large error increases gradient 
  wout += hiddenlayer_activations.T.dot(d_output) * lr
  bout += np.sum(d_output, axis=0, keepdims=True) * lr
  epochCount = epochCount + 1
  if epochCount % 10 == 0:
    print(f'epoch {epochCount}')

et=pc()
print(f'Runtime:  {(et-st)/1e6:.3f} ms')
predY = np.where(output> 0.75, 1, 0)
accuracy = accuracy_score(y, predY)
print(f'Accuracy:  {accuracy:.2f}')
mse = (np.square(predY - y)).mean(axis=None)
print(f'Error: {mse:.2f}')





