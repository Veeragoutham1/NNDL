import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w1 = 0.8
w2 = 0.9
bias = 0.25
learning_rate = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(5000):
    for i in range(4):
        z = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(z)
        error = y[i] - result
        w1 += learning_rate * error * x[i][0]
        w2 += learning_rate * error * x[i][1]
        bias += learning_rate * error

print("Final weights:", w1, w2)
print("Final bias:", bias)

for i in range(4):
    z = x[i][0] * w1 + x[i][1] * w2 + bias
    result = sigmoid(z)
    print(f"Input: {x[i]}, Output: {result:.4f}, Expected: {y[i]}")
