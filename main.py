import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

N = 2000 # number of loops
t_n = 20 # end time
h = t_n / N # inverval

dim = 3 # dimension
x_init = np.array([1.0, 1.0, 1.0])

# Runge-Kutta method
def RungeKutta(x, f):
    k1 = f(x)
    k2 = f(x + h * k1 / 2.0)
    k3 = f(x + h * k2 / 2.0)
    k4 = f(x + h * k3)

    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

class LorenzModel:
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r

    def __call__(self, x):
        dx_dt = -self.a * x[0] + self.a * x[1]
        dy_dt = (self.r - x[2]) * x[0] - x[1]
        dz_dt = x[0] * x[1] - self.b * x[2]

        return np.array([dx_dt, dy_dt, dz_dt])

def main():
    data = np.zeros((N + 1, dim))
    model = LorenzModel(10, 8 / 3, 28)

    data[0] = x_init.copy()
    for i in range(N):
        data[i + 1] = RungeKutta(data[i], model)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title("Lorenz model")
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    m = N // 2

    x = data[0:m]
    y = data[1:(m + 1)]
    y_test = data[(m + 1):]

    print('Linear regression')
    lr = LinearRegression()
    lr.fit(x, y)

    y_pred_r_train = lr.predict(x)
    train_mse = mean_squared_error(y, y_pred_r_train)
    print("Training error: {}".format(train_mse))

    y_pred_r_test = lr.predict(data[m:N])
    test_mse = mean_squared_error(y_test, y_pred_r_test)
    print("Test error: {}".format(test_mse))

    print('Ridge regression')
    ridge = Ridge(alpha=0.01, max_iter=1000)
    ridge.fit(x, y)

    y_pred_ridge_train = ridge.predict(x)
    train_mse = mean_squared_error(y, y_pred_ridge_train)
    print("Training error: {}".format(train_mse))

    y_pred_ridge_test = ridge.predict(data[m:N])
    test_mse = mean_squared_error(y_test, y_pred_ridge_test)
    print("Test error: {}".format(test_mse))

    print('LASSO rigression')
    lasso = Lasso(alpha=0.01, max_iter=1000)
    lasso.fit(x, y)

    y_pred_lasso_train = lasso.predict(x)
    train_mse = mean_squared_error(y, y_pred_lasso_train)
    print("Training error: {}".format(train_mse))
    
    y_pred_lasso_test = lasso.predict(data[m:N])
    test_mse = mean_squared_error(y_test, y_pred_lasso_test)
    print("Test error: {}".format(test_mse))

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Lorenz model and predicted training data')
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.plot(y_pred_lasso_train[:, 0], y_pred_lasso_train[:, 1], y_pred_lasso_train[:, 2])
    ax.legend(labels=['Lorenz model', 'LASSO regression'])
    plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Lorenz model and predicted test data')
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.plot(y_pred_lasso_test[:, 0], y_pred_lasso_test[:, 1], y_pred_lasso_test[:, 2])
    ax.legend(labels=['Lorenz model', 'LASSO regression'])
    plt.show()

if __name__ == "__main__":
    main()
