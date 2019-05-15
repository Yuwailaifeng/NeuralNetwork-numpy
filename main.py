from model import *
import matplotlib.pyplot as plt

model = Model(2, 8, 1, activation='sigmoid')

# Test XOR problem!
x = np.array([0, 0], [0, 1], [1, 0], [1, 1])
y = np.array([0, 1, 1, 0])

epoch = 20
train_loss = []

for i in range(epoch):
    result = model.forward(x)
    loss = (np.square(result - y)).mean()
    backprop = model.backward(loss)
    model.update()

    train_loss.append(loss)

# Testing..
print(model.forward(x))

plt.plot(train_loss)
plt.show()