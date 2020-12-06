# Linear descent
Simply run train-model.py to train & test-model.py to test

## Gradient Descent
The code will train & test 3 models of gradient descent:
- Standard Gradient Descent
- Stochastic Gradient Descent
- Minibatch Gradient Descent

## Code
- **model.py** - Contains the model code, and each gradient descent algorithm
- utils/**csvreader.py** - reads the csv data into python lists
- utils/**standardscaler.py** - applies *data = (data - mean) / std* for each feature
- **train-model.py** - trains the models
- **test-model.py** - tests the models

## Includes
The **pickle** library is used to serialize the models into a models folder.
