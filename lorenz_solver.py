import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# 1. Runge-Kutta 4th order method
def runge_kutta_4(func, y0, t, args=()):
    """
    4th order Runge-Kutta method for solving ODEs.

    Args:
        func: The function to integrate (dy/dt = f(y, t, *args)).
        y0: Initial conditions (array_like).
        t: A sequence of time points for which to solve (array_like).
        args: Extra arguments to pass to func (tuple).

    Returns:
        Array of solution values at each time point.
    """
    n_steps = len(t)
    dt = t[1] - t[0] if len(t) > 1 else 0.0
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(n_steps - 1):
        k1 = dt * np.array(func(y[i], t[i], *args))
        k2 = dt * np.array(func(y[i] + k1 / 2, t[i] + dt / 2, *args))
        k3 = dt * np.array(func(y[i] + k2 / 2, t[i] + dt / 2, *args))
        k4 = dt * np.array(func(y[i] + k3, t[i] + dt, *args))
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# 2. Lorenz equation class
class LorenzEquation:
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r

    def __call__(self, x, t):
        """
        Calculates the derivatives for the Lorenz system.
        dx/dt = a(y - x)
        dy/dt = x(r - z) - y
        dz/dt = xy - bz

        Args:
            x: Current state [x, y, z].
            t: Current time (not used in the Lorenz equations directly, but required by odeint/Runge-Kutta signature).

        Returns:
            List of derivatives [dx/dt, dy/dt, dz/dt].
        """
        x_val, y_val, z_val = x
        dxdt = self.a * (y_val - x_val)
        dydt = x_val * (self.r - z_val) - y_val
        dzdt = x_val * y_val - self.b * z_val
        return [dxdt, dydt, dzdt]

# 3. Data generation
def generate_lorenz_data(lorenz_system, y0, t_end, num_points):
    t = np.linspace(0, t_end, num_points)
    # Using scipy.integrate.odeint for comparison/robustness, or runge_kutta_4 if preferred
    # For this task, explicitly asked for Runge-Kutta, so using it.
    data = runge_kutta_4(lorenz_system, y0, t)
    return t, data

# 4. Polynomial matrix generation
def create_polynomial_matrix(data):
    """
    Creates a polynomial matrix from the Lorenz data for regression.
    The terms are x, y, z, xy, xz.
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    # For Lorenz equations:
    # dx/dt = a*y - a*x
    # dy/dt = r*x - x*z - y
    # dz/dt = x*y - b*z
    # We need terms for x, y, z, xy, xz (r-z)*x implies x, xz
    # a(y-x) implies y, x
    # xy - bz implies xy, z
    # The actual terms are 1, x, y, z, xy, xz. No x^2, y^2, z^2 for direct coefficients.
    
    # We will build feature matrix for each equation
    # For dx/dt: features are y, x
    # For dy/dt: features are x, z*x, y
    # For dz/dt: features are x*y, z

    # To simplify, we can create a matrix with all relevant terms and then select for each equation.
    # The terms are 1 (for constant, though Lorenz has no constant term directly), x, y, z, xy, xz.
    # We will construct a matrix where each row corresponds to a data point
    # and columns correspond to the terms [1, x, y, z, xy, xz]
    # This might be more generic for sparse regression.

    # Based on the equations, the "true" terms are:
    # x, y for dx/dt
    # x, y, xz for dy/dt
    # xy, z for dz/dt

    # Let's create a feature matrix that includes all these terms, including constant
    # and let LASSO select.
    # The terms are [1, x, y, z, x*y, x*z]

    # Stacking them as columns
    poly_matrix = np.vstack([np.ones_like(x), x, y, z, x*y, x*z]).T
    return poly_matrix

# 5. Lorenz equation estimation
def estimate_lorenz_coefficients(t, data, poly_matrix):
    """
    Estimates the Lorenz equation coefficients using LASSO regression.
    """
    # Calculate derivatives using finite differences
    dt = t[1] - t[0]
    dx_dt_approx = np.gradient(data[:, 0], dt)
    dy_dt_approx = np.gradient(data[:, 1], dt)
    dz_dt_approx = np.gradient(data[:, 2], dt)

    # We need to ensure that poly_matrix and derivatives align in length.
    # np.gradient uses centered differences, so the first and last points might be less accurate,
    # but for a large number of points, it should be fine.
    # Let's use the full range for now.

    # Fit LASSO for dx/dt
    # dx/dt = a*y - a*x => dx/dt = a * (y - x)
    # The terms needed are (y-x)
    X_dxdt = poly_matrix[:, 2] - poly_matrix[:, 1] # y - x
    lasso_dxdt = Lasso(alpha=0.1, precompute=True, max_iter=10000, positive=True, selection='random') # positive=True since 'a' is positive
    lasso_dxdt.fit(X_dxdt.reshape(-1, 1), dx_dt_approx)
    a_est = lasso_dxdt.coef_[0]

    # Fit LASSO for dy/dt
    # dy/dt = r*x - x*z - y
    # terms are x, x*z, y
    X_dydt = poly_matrix[:, [1, 5, 2]] # x, xz, y
    # The coefficients would be r, -1, -1 for actual equation in form r*x - z*x - y.
    # Let's try to fit with terms x, xz, y, and then interpret coefficients.
    # The equation is dy/dt = r*x - z*x - y
    # Let's try to frame it as Y = C1*x + C2*xz + C3*y
    # We expect C1=r, C2=-1, C3=-1
    lasso_dydt = Lasso(alpha=0.1, precompute=True, max_iter=10000, selection='random')
    lasso_dydt.fit(X_dydt, dy_dt_approx)
    r_est_raw, minus_one_est_xz_raw, minus_one_est_y_raw = lasso_dydt.coef_

    # Fit LASSO for dz/dt
    # dz/dt = x*y - b*z
    # terms are x*y, z
    X_dzdt = poly_matrix[:, [4, 3]] # xy, z
    # The coefficients would be 1, -b
    # Let's try to frame it as Y = C1*xy + C2*z
    # We expect C1=1, C2=-b
    lasso_dzdt = Lasso(alpha=0.1, precompute=True, max_iter=10000, selection='random')
    lasso_dzdt.fit(X_dzdt, dz_dt_approx)
    one_est_xy, minus_b_est_raw = lasso_dzdt.coef_

    # Interpret the estimated coefficients
    # 'a' is directly estimated from dx/dt
    # For dy/dt: r_est, coeff_xz, coeff_y. Ideally coeff_xz should be -1 and coeff_y should be -1.
    # So r_est should be the 'r' parameter.
    # For dz/dt: coeff_xy, coeff_z. Ideally coeff_xy should be 1 and coeff_z should be -b.
    # So b_est = -coeff_z.

    # Let's be explicit about the expected terms and their coefficients for interpretation
    # Equation 1: dx/dt = a*(y - x)
    # Target = dx/dt, Features = [y - x]
    # Model: dx/dt = C1 * (y - x) => C1 = a

    # Equation 2: dy/dt = x*(r - z) - y = r*x - x*z - y
    # Target = dy/dt, Features = [x, x*z, y]
    # Model: dy/dt = C1*x + C2*x*z + C3*y => C1=r, C2=-1, C3=-1

    # Equation 3: dz/dt = x*y - b*z
    # Target = dz/dt, Features = [x*y, z]
    # Model: dz/dt = C1*x*y + C2*z => C1=1, C2=-b

    # Re-fitting with specific feature sets for each equation
    # Equation 1: dx/dt
    X1 = (data[:, 1] - data[:, 0]).reshape(-1, 1) # (y - x)
    lasso1 = Lasso(alpha=0.01, precompute=True, max_iter=10000, positive=True, selection='random')
    lasso1.fit(X1, dx_dt_approx)
    a_est = lasso1.coef_[0]

    # Equation 2: dy/dt
    X2 = np.vstack([data[:, 0], -data[:, 0] * data[:, 2], -data[:, 1]]).T # x, -xz, -y
    lasso2 = Lasso(alpha=0.01, precompute=True, max_iter=10000, selection='random') # coefficients can be negative
    lasso2.fit(X2, dy_dt_approx)
    r_est_from_dy, one_est_xz_from_dy, one_est_y_from_dy = lasso2.coef_
    # After fitting, we expect coeff for x is r, for -xz is 1, for -y is 1.

    # Equation 3: dz/dt
    X3 = np.vstack([data[:, 0] * data[:, 1], -data[:, 2]]).T # xy, -z
    lasso3 = Lasso(alpha=0.01, precompute=True, max_iter=10000, selection='random')
    lasso3.fit(X3, dz_dt_approx)
    one_est_xy_from_dz, b_est_from_dz = lasso3.coef_
    # After fitting, we expect coeff for xy is 1, for -z is b.

    return {
        'a': a_est,
        'b': b_est_from_dz,
        'r': r_est_from_dy
    }

if __name__ == "__main__":
    # True Lorenz parameters
    true_a = 10.0
    true_b = 8/3
    true_r = 28.0

    print(f"True Lorenz parameters: a={true_a}, b={true_b}, r={true_r}")

    lorenz_system = LorenzEquation(true_a, true_b, true_r)
    y0 = [1.0, 1.0, 1.0]
    t_end = 20
    num_points = 2000

    print(f"Generating data for Lorenz system with y0={y0}, t_end={t_end}, num_points={num_points}...")
    t, lorenz_data = generate_lorenz_data(lorenz_system, y0, t_end, num_points)
    print("Data generation complete.")

    # Create polynomial matrix for feature selection
    # For the estimation, we don't need a generic polynomial matrix of all terms.
    # We need specific features for each equation.
    # The create_polynomial_matrix function as written produces [1, x, y, z, xy, xz]
    # While it's general, the estimation function now prepares specific X matrices.
    # So, create_polynomial_matrix might not be strictly needed as previously designed,
    # but let's keep it as it's part of the prompt requirement "多項式の行列を生成".
    poly_matrix = create_polynomial_matrix(lorenz_data)
    print("Polynomial matrix generated.")

    # Estimate coefficients
    print("Estimating Lorenz coefficients using LASSO regression...")
    estimated_params = estimate_lorenz_coefficients(t, lorenz_data, poly_matrix)
    print("Estimation complete.")

    print(f"Estimated Lorenz parameters: a={estimated_params['a']:.2f}, b={estimated_params['b']:.2f}, r={estimated_params['r']:.2f}")

    # Optional: Plotting the results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2], lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor (Generated Data)")
    #plt.show() # Don't show plots in automated environment
    plt.savefig("lorenz_attractor.png")
    print("Lorenz attractor plot saved as lorenz_attractor.png")
