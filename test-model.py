from utils import csvreader
from model import Model
import numpy as np
import pickle

# SSE
def sse(x, y):
    return (y - x)**2

# load csv
x, y = csvreader.open_csv("data/Admission_Predict_Ver1.1.csv")
x = np.array(x)
y = np.array(y)

# load serialized model
standard_model = pickle.load(open("models/standard_model.ser", "rb"))
stochastic_model = pickle.load(open("models/stochastic_model.ser", "rb"))
minibatch_model = pickle.load(open("models/minibatch_model.ser", "rb"))

# Number of correct guesses
standard_sse = 0.0
stochastic_sse = 0.0
minibatch_sse = 0.0

# Real-time prediction
for d, a in zip(x, y):
    standard_sse += sse(standard_model.predict(np.array([d]))[0], a)
    # stochastic_sse += sse(stochastic_model.predict([d])[0], a)
    # minibatch_sse += sse(minibatch_model.predict([d])[0], a)

# Print result
print("Standard SSE: " + str(standard_sse))
print("Stochastic SSE: " + str(stochastic_sse))
print("Minibatch SSE: " + str(minibatch_sse))