import numpy as np

def softmax(x):
    numerator= np.exp(x)
    denominator = np.sum(numerator)
    output = numerator / denominator
    return output

def cross_entropy(prediction, target, epsilon=1e-12):
    prediction = np.clip(prediction, epsilon, 1.-epsilon)
    N = prediction.shape[0]
    ce = -np.sum(target*np.log(prediction+1e-9))/N
    return ce

class Sigmoid(object):
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        output = 1 / (1 + np.exp(-x))
        self.out = output

        return output

    def backward(self, d_out):
        dx = (1 - self.out) * self.out * d_out # sigmoid derivative : y(1-y)

        return dx

class LeakyReLU(object):
    def __init__(self):
        self.out = None
    
    def forward(self, x, alpha=0.1): # As lecture slide, alpha = 0.
        self.x = x

        output = np.array([i if i>=0 else i*alpha for i in x])
        self.out = output
        self.alpha = alpha

        return output

    def backward(self, d_out):
        dx = np.array([1 if i>=0 else self.alpha for i in self.x])

        return dx * d_out

class Linear(object):
    def __init__(self, input_size, output_size, bias=True):
        self.x = None
        self.dW = None
        self.db = None
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.weight = np.random.normal(0, 1, (output_size, input_size))

        if bias:
            self.bias = np.random.normal(0, 1, output_size)
        else:
            self.bias = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        output = np.dot(self.weight, x) + self.bias

        return output

    def backward(self, d_out):
        dx = np.dot(self.weight.T, d_out)
        self.dW = np.dot(d_out[np.newaxis].T, self.x[np.newaxis])
        self.db = np.sum(self.bias, axis=0)

        return dx

class Model(object):
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)
        
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            self.activation = LeakyReLU()

    def forward(self, x):
        output = self.layer1.forward(x)
        output = self.activation.forward(output)
        output = self.layer2.forward(output)
        output = self.activation.forward(output)
        
        return output

    def backward(self, d_out):
        d_out = self.activation.backward(d_out)
        d_out = self.layer2.backward(d_out)
        d_out = self.activation.backward(d_out)
        d_out = self.layer1.backward(d_out)

        return d_out

    def update(self, learning_rate=0.1):
        self.layer1.weight = self.layer1.weight - learning_rate * self.layer1.dW
        self.layer1.bias = self.layer1.bias  - learning_rate * self.layer1.db

        self.layer2.weight = self.layer2.weight - learning_rate * self.layer2.dW
        self.layer2.bias = self.layer2.bias - learning_rate * self.layer2.db
