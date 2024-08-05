import pandas as pd
import numpy as np

df = pd.read_csv('test_X.csv')
df = df.astype({"Survived": float})

# Set the calculated weights and bias
w1 = -0.769
w2 = 2.72
w3 = -0.0231
w4 = -0.317
w5 = -0.1196
w6 = 0.00566
w7 = -0.0526
b = 0.932



z = (df['Pclass'] * w1) + (df['Sex'] * w2) + (df['Age'] * w3) + (df['SibSp'] * w4) + (df['Parch'] * w5) + (df['Fare'] * w6) + (df['Embarked'] * w7) + b



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

predictions = sigmoid(z)

# Now we will go through each number in the list
for i in range(len(predictions)):  # "range(len(numbers))" is a way to count how many items there are in the list
    if predictions[i] > 0.5:  # If the number is bigger than 0.5
        predictions[i] = int(1)  # We replace it with 1
    else:  # If it's not bigger than 0.5
        predictions[i] = int(0)  # We replace it with 0

actuals = df['Survived']

z = [i == j for i,j in zip(predictions,actuals)]

true_Counter = 0
false_Counter = 0

for i in z:
    if i == True:
        true_Counter += 1
    else:
        false_Counter += 1

accuracy = ((true_Counter / (false_Counter + true_Counter)) * 100)

print(f'The model predicted {true_Counter} correct, and {false_Counter} incorrect, thus the prediction is {round(accuracy)}%')