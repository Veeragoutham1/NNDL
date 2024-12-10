import numpy as np

# Input data: AND function
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Target outputs for AND

# Initialize weights and bias
w1 = 0.8
w2 = 0.9
bias = 0.25
learning_rate = 0.1

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training loop
for epoch in range(5000):
    for i in range(4):
        # Calculate the weighted input plus bias
        z = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(z)

        # Calculate the error
        error = y[i] - result
        
        # Update weights and bias
        w1 += learning_rate * error * x[i][0]
        w2 += learning_rate * error * x[i][1]
        bias += learning_rate * error

# Testing the trained model
print("Final weights:", w1, w2)
print("Final bias:", bias)

# Show results for all inputs
for i in range(4):
    z = x[i][0] * w1 + x[i][1] * w2 + bias
    result = sigmoid(z)
    print(f"Input: {x[i]}, Output: {result:.4f}, Predicted: {1 if result >= 0.5 else 0}")
