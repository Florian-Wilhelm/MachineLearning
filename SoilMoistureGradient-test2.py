import numpy as np

def predict(X, w):
    return np.matmul(X,w)  # np.matmul = matrix multiplication; result is a (n, 1)-matrix

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)  # result is a (n, 1)-matrix

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
    # operation X.T results in a (2, n)-matrix that gets multiplied with a (n, 1)-matrix; results then in a (2, 1) matrix

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w)))
        w -= gradient(X,Y,w) * lr
    return w

x1, x2, y = np.loadtxt("ML_data/moisture-2Inputs.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2))
Y = y.reshape(-1, 1)

w = train(X, Y, iterations=2000, lr=0.0001)
print(w.T)