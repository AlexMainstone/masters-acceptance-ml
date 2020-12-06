import numpy as np

class Model:
    def __init__(self, features=5, max_iter=25000, alpha=0.0005, precision=0.01):
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

        # stores gradient of cost function
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
            # if abs(sum(self.theta - prev_theta)) < self.precision:
                # print("PRECISION MET!")
                # break

            # Print every 100 iterations
            if i % 100 == 0:
                # Calculate cost
                pred_cost = self.cost(pred_y, y)
                print("ITERATION " + str(i) + ": " + str(pred_cost))
    
    def stochastic_fit(self, x, y):
        # add a column of 1s
        x = np.array(np.hstack((np.ones((len(x), 1)), x)))

        # stores gradient of cost function
        dt = np.zeros(self.m)

        # Shuffle dataset
        np.random.shuffle(x)

        for i in range(self.max_iter):
            for xj, yj in zip(x, y):
                # Predict
                pred_y = np.dot(xj, self.theta)

                # Calculate loss
                loss = pred_y - yj

                # Calculate gradient for j
                dt = np.dot(xj.T, loss)

                # Calculate theta
                self.theta -= self.alpha * dt
    
    def minibatch_fit(self, x, y, batch_size=10):
        # add a column of 1s
        x = np.array(np.hstack((np.ones((len(x), 1)), x)))

        # stores gradient of cost function
        dt = np.zeros(self.m)
        
        for i in range(self.max_iter):
            for j in range(int(len(x)/batch_size)):
                # Get subarrays of batch size
                xj  = x[j*batch_size:j+batch_size]
                yj  = y[j*batch_size:j+batch_size]

                # Predicted y
                pred_y = np.dot(xj, self.theta)

                # Calculate loss
                loss = pred_y - yj

                # Calculate batch gradient
                dt = np.dot(xj.T, loss)

                # Calculate theta
                self.theta -= self.alpha * dt

    def predict(self, x):
        # Add column of 1s
        x = np.array(np.hstack((np.ones((len(x), 1)), x)))
        return np.dot(x, self.theta)