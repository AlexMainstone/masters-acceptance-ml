import numpy as np

class Model:
    def __init__(self, features=5, max_iter=2500, alpha=0.01, precision=0.01):
        self.theta = np.zeros(features+1)
        self.alpha = alpha
        self.precision = precision

        self.m = features + 1
        self.max_iter = max_iter

    def cost(self, pred, y):
        return sum((pred-y)**2)
    
    def standard_fit(self, x, y):
        # add a column of 1s
        x = np.array(np.hstack((np.ones((len(x), 1)), x)))
        dt = np.zeros(self.m)

        for i in range(self.max_iter):
            # Predict y
            pred_y = np.dot(x, self.theta)

            # Calculate loss
            loss = pred_y - y

            # Calculate gradient of the cost
            dt = np.dot(x.T, loss)

            # modify theta
            prev_theta = self.theta.copy()
            self.theta -= self.alpha * dt

            # Check precision
            if abs(sum(self.theta - prev_theta)) < self.precision:
                print("PRECISION MET!")
                break

            if i % 100 == 0:
                # Calculate cost
                pred_cost = self.cost(pred_y, y)
                print("ITERATION " + str(i) + ": " + str(pred_cost))

    def predict(self, x):
        return self.m.dot(x.T) + self.b