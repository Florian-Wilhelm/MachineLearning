# Includes a bias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w, b):
    return X * w  + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)  # MSE

def train(X, Y, iterations, lr):
    w = 0
    b = 50
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Couldn't converge within %d iterations" % iterations)

sns.set()
plt.axis([0, 300, 50, 65])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel("24h cycle", fontsize=15)
plt.ylabel("Moisture (%)", fontsize=15)

X, Y = np.loadtxt("ML_data/moisture.txt", skiprows=0, unpack=True)

w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

plt.plot(X, Y, "bo")
plt.plot(w*X + b, "r-")
plt.show()
