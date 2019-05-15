from model import *
import matplotlib.pyplot as plt

model = Model(2, 8, 2, activation='sigmoid')

# Test XOR problem!
x = np.array([1, 0])
y = np.array([0, 1])

epoch = 20
train_loss = []

for i in range(epoch):
    result = model.forward(x)
    loss = (np.square(result - y)).mean()
    backprop = model.backward(loss)
    model.update()

    train_loss.append(loss)

print(model.forward(x))

plt.plot(train_loss)
plt.show()