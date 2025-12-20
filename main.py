import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
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
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    shift = 1

    x = data[0:(N + 1 - shift)]
    y = data[shift:N + 1]

    lasso = Lasso(alpha=0.01, max_iter=1000)
    lasso.fit(x, y)

    y_pred = lasso.predict(x)
    mse = mean_squared_error(y, y_pred)

    print(mse)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2])
    plt.show()

if __name__ == "__main__":
    main()
