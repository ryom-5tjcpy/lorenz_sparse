import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

N = 2000 # number of loops
t_n = 20 # end time
h = t_n / N # interval

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
    y = np.zeros([N, dim])

    data[0] = x_init.copy()
    for i in range(N):
        y[i] = model(data[i])
        data[i + 1] = RungeKutta(data[i], model)

if __name__ == "__main__":
    main()
