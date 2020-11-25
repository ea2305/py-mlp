# Multi Layer Perceptron - Python implementation from scratch

We provide a useful way to implement a simple multi layer perceptron.

### Requirements

* Python >= 3.9 
* numpy [don't install the latest version due unexpected issues]

## Example

This implementation generates a random set of numbers and train the model to obtain a network capable to sum two numbers (yeah, is an overkill implementation, but show the basics.)

```py
# generate a set of random values for input and targets
inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] + i[1]] for i in inputs])

# create mlp
# input, hidden layer and output layer
mlp = Mlp(2, [5], 1)

# train
mlp.training(inputs, targets, 100, 0.1)

# test traing results with two new values
input = np.array([0.3, 0.1])
target = np.array([0.4])

output = mlp.fwdPropagation(input)
print("Network x:{} + y:{} = {}".format(input[0], input[1], output[0]))
```