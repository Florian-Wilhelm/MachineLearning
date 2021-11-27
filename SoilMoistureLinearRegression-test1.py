import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w):
    return X * w 

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)  # MSE

def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
    raise Exception("Couldn't converge within %d iterations" % iterations)

sns.set()
plt.axis([0, 300, 50, 65])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel("24h cycle", fontsize=15)
plt.ylabel("Moisture (%)", fontsize=15)

X, Y = np.loadtxt("ML_data/moisture.txt", skiprows=0, unpack=True)

w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % w)

plt.plot(X, Y, "bo")
plt.plot(w*X, "r-")
plt.show()
