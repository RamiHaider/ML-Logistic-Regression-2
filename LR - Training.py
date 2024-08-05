import pandas as pd
import numpy as np

df = pd.read_csv('train_X.csv')

costy_Cost = 1
best_Cost_Function_Weights = []

# Initialize the weights and bias
w1 = 0.5
w2 = 0.01
w3 = 3
w4 = 1
w5 = 1
w6 = 1
w7 = 1
b = 0

# Define learning rate
learning_rate = 0.0015

# Define the number of epochs
epochs = 100000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(epochs):
    z = (df['Pclass'] * w1) + (df['Sex'] * w2) + (df['Age'] * w3) + (df['SibSp'] * w4) + (df['Parch'] * w5) + (df['Fare'] * w6) + (df['Embarked'] * w7)+ b

    y_hat = sigmoid(z)

    epsilon = 1e-7  # small constant
    loss_function = - (df['Survived'] * np.log(y_hat + epsilon) + (1 - df['Survived']) * np.log(1 - y_hat + epsilon))

    cost_function = np.mean(loss_function)

    # Backpropagation
    dz = y_hat - df['Survived']

    dw1 = np.mean(dz * df['Pclass'])
    dw2 = np.mean(dz * df['Sex'])
    dw3 = np.mean(dz * df['Age'])
    dw4 = np.mean(dz * df['SibSp'])
    dw5 = np.mean(dz * df['Parch'])
    dw6 = np.mean(dz * df['Fare'])
    dw7 = np.mean(dz * df['Embarked'])
    db = np.mean(dz)

    # Update weights and bias
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3
    w4 = w4 - learning_rate * dw4
    w5 = w5 - learning_rate * dw5
    w6 = w6 - learning_rate * dw6
    w7 = w7 - learning_rate * dw7
    b = b - learning_rate * db

    # best_Cost_Function_Weights = []
    if cost_function < costy_Cost:
        costy_Cost = cost_function
        best_Cost_Function_Weights.append(w1)
        best_Cost_Function_Weights.append(w2)
        best_Cost_Function_Weights.append(w3)
        best_Cost_Function_Weights.append(w4)
        best_Cost_Function_Weights.append(w5)
        best_Cost_Function_Weights.append(w6)
        best_Cost_Function_Weights.append(w7)
        best_Cost_Function_Weights.append(b)        


    # Print the epoch number, cost function and updated weights and bias
    print(f'Epoch: {i+1}, Best-Cost-Funct: {costy_Cost}')

print(f"The Best weights are: {best_Cost_Function_Weights[-8:]}, with the best cost_function score at {costy_Cost}")