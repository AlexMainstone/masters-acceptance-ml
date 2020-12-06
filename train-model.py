from utils import csvreader, standardscaler
import numpy as np
import pickle
import model
import os

# load csv
x, y = csvreader.open_csv("data/Admission_Predict_Ver1.1.csv")
x = standardscaler.scale(np.array(x))
# x = np.array(x)
y = np.array(y)

# Check if models path exists
if not os.path.exists("models"):
    os.mkdir("models")

# Create & fit models, then serialize
print("STANDARD:")
standard_model = model.Model(len(x[0]))
standard_model.standard_fit(x, y)
print(standard_model.predict(x))
pickle.dump(standard_model, open("models/standard_model.ser", "wb"))

print("STOCHASTIC:")
stochastic_model = model.Model(len(x[0]), max_iter=1000)
stochastic_model.stochastic_fit(x, y)
print(stochastic_model.predict(x))
pickle.dump(stochastic_model, open("models/stochastic_model.ser", "wb"))

print("MINIBATCH:")
minibatch_model = model.Model(len(x[0]), max_iter=1000)
minibatch_model.minibatch_fit(x, y)
print(minibatch_model.predict(x))
pickle.dump(minibatch_model, open("models/minibatch_model.ser", "wb"))