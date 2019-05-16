from model import *
import matplotlib.pyplot as plt

model = Model(1, 10, 1, activation='sigmoid')

# Test AND problem!
#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x = np.array([[0.1], [0.2], [0.3], [0.4]])
y = np.array([0.2, 0.4, 0.6, 0.8])

epoch = 20
train_loss = []

for i in range(epoch):
    loss_ = 0
    for k, x_ in enumerate(x): # Online learning
        result = model.forward(x_)
        loss = (np.square(result - y[k])).mean()
        loss_ += loss
        backprop = model.backward(loss)
        model.update(learning_rate=0.05)

    train_loss.append(loss_/4)

print(model.forward(np.array([0.5])))
plt.plot(train_loss)
plt.show()