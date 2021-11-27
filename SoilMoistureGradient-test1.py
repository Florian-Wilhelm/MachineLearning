import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w, b):
    return X * w + b  # Line Equation

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)  # Mean Squared Error

def gradient(X, Y, w, b):
    w_gradient = (2 * np.average(X*(predict(X, w, b) - Y)))  # w Gradient of the Loss
    b_gradient = (2 * np.average(predict(X, w, b) - Y))  # b Gradient of the Loss
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X,Y,w,b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

sns.set()
# plt.axis([0, 10, 0, 10])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)

X, Y = np.loadtxt("ML_data/test.txt", skiprows=0, unpack=True)

w, b = train(X, Y, iterations=1000, lr=0.0001)
print("\nw=%.10f, b=%.10f" % (w, b))

plt.plot(X, Y, "bo")
plt.plot(w*X + b, "r-")
plt.show()
