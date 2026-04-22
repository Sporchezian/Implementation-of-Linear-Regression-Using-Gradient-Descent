# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import libraries and load the dataset from the CSV file.

2. Select input (R&D Spend) and output (Profit) values.

3.Normalize the input data and initialize parameters (m, b).

4.Set learning rate, epochs, and apply Gradient Descent loop.

5.Update slope and intercept in each iteration to reduce error using. 

6.Print final values and plot the regression line with data points.

## Program:

Program to implement the linear regression using gradient descent.
Developed by: PORCHEZIAN S
RegisterNumber:  212225040304

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

## Output:

<img width="838" height="636" alt="image" src="https://github.com/user-attachments/assets/7cd64aac-ab8c-4026-aaad-bd92ddf0ad47" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
