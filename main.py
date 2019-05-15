from model import *
import matplotlib.pyplot as plt

model = Model(2, 4, 1, activation='sigmoid')

# Test XOR problem!
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

epoch = 20
train_loss = []

for i in range(epoch):
    print('Epoch :', i+1)
    for k, x_ in enumerate(x): # Batch size = 1
        result = model.forward(x_)
        loss = (np.square(result - y[k])).mean()
        backprop = model.backward(loss)
        model.update()

    train_loss.append(loss)

# Testing..
print(model.forward(np.array([1,0])))

plt.plot(train_loss)
plt.show()