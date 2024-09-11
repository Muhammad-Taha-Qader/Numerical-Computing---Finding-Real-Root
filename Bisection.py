import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Bisection Method Implementation with iteration prints
def bisection_method(func, a, b, tol=1e-5, max_iter=100):
    x = sp.symbols('x')
    # Predefine math functions like log for user input
    local_dict = {'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}
    f = sp.sympify(func, locals=local_dict)  # Convert the input string to a symbolic expression
    f_lambdified = sp.lambdify(x, f, 'numpy')  # Convert symbolic to a numeric function
    
    if f_lambdified(a) * f_lambdified(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints. Bisection Method Fail to find Root")
    
    midpoints = []
    intervals = []
    
    print(f"\n{'Iteration':<10}{'Midpoint':<20}{'Interval':<30}")
    print(f"{'-'*60}")
    
    for i in range(max_iter):
        midpoint = (a + b) / 2
        midpoints.append(midpoint)
        intervals.append((a, b))
        
        print(f"{i+1:<10}{midpoint:<20.10f}[{a:<10.5f}, {b:<10.5f}]")
        
        if abs(f_lambdified(midpoint)) < tol:
            break
        
        if f_lambdified(a) * f_lambdified(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

    return midpoint, midpoints, intervals, f_lambdified

# Plot the function and solution
def plot_function(func, a, b, midpoints, intervals, solution):
    x_vals = np.linspace(a - 1, b + 1, 1000)
    f = sp.sympify(func, locals={'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    f_lambdified = sp.lambdify(sp.symbols('x'), f, 'numpy')
    y_vals = f_lambdified(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'f(x) = {func}')
    plt.axhline(0, color='black',linewidth=0.5)
    
    for i, (a_i, b_i) in enumerate(intervals):
        plt.axvline(a_i, color='blue', linestyle='--', label=f"Iteration {i+1} interval" if i == 0 else "")
        plt.axvline(b_i, color='blue', linestyle='--')
    
    plt.scatter(midpoints, [f_lambdified(m) for m in midpoints], color='red', zorder=5, label='Midpoints')
    plt.scatter(solution, f_lambdified(solution), color='green', zorder=5, label=f'Solution x = {solution:.5f}')
    
    plt.title(f"Bisection Method for f(x) = {func}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
def main():
    # Example: "x**2 - 4*x + log(x) = 0, interval 1<=x<=2"
    func = input("Enter the function (e.g., x**2 - 4*x + log(x)): ")
    interval_a = float(input("Enter the lower bound of the interval (a): "))
    interval_b = float(input("Enter the upper bound of the interval (b): "))
    
    try:
        solution, midpoints, intervals, f_lambdified = bisection_method(func, interval_a, interval_b)
        print(f"\nThe root is approximately at x = {solution:.5f}")
        plot_function(func, interval_a, interval_b, midpoints, intervals, solution)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
