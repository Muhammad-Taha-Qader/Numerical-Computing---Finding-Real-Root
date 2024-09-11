import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Secant Method Implementation
def secant_method(func, a, b, tol=1e-5, max_iter=100):
    x = sp.symbols('x')
    local_dict = {'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan}
    
    # Convert input string to symbolic expression
    f = sp.sympify(func, locals=local_dict)
    
    # Lambdify for numerical evaluation
    f_lambdified = sp.lambdify(x, f, 'numpy')
    
    # Check if the function has different signs at a and b
    if f_lambdified(a) * f_lambdified(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints of the interval [a, b].")
    
    # Initial guesses
    x0 = a
    x1 = b
    
    # Lists to track the values of x and f(x) during iterations
    x_values = [x0, x1]
    f_values = [f_lambdified(x0), f_lambdified(x1)]
    
    print(f"\n{'Iteration':<10}{'x0':<20}{'x1':<20}{'f(x0)':<20}{'f(x1)':<20}")
    print(f"{'-'*80}")
    
    for i in range(max_iter):
        f_x0 = f_lambdified(x0)
        f_x1 = f_lambdified(x1)
        
        # Division by zero Check
        if f_x1 - f_x0 == 0:
            raise ValueError("Division by zero in secant method.")
        
        # Secant iteration: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        x_values.append(x2)
        f_values.append(f_lambdified(x2))
        
        print(f"{i+1:<10}{x0:<20.10f}{x1:<20.10f}{f_x0:<20.10f}{f_x1:<20.10f}")
        
        if abs(f_lambdified(x2)) < tol:
            break
        
        # Update guesses
        x0 = x1
        x1 = x2

    return x2, x_values, f_values, f_lambdified

# Plotting the function and Secant method iteration points
def plot_secant_method(func, x_values, f_values, solution):
    x_vals = np.linspace(min(x_values) - 1, max(x_values) + 1, 1000)
    f = sp.sympify(func, locals={'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan, 'ln': sp.ln})
    f_lambdified = sp.lambdify(sp.symbols('x'), f, 'numpy')
    y_vals = f_lambdified(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'f(x) = {func}')
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.scatter(x_values, f_values, color='red', zorder=5, label='Iteration Points')
    plt.scatter(solution, f_lambdified(solution), color='green', zorder=5, label=f'Solution x = {solution:.5f}')
    
    plt.title(f"Secant Method for f(x) = {func}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # x**2 - 4*x + 4 - ln(x)
    # ln(x-1) + cos(x-1)
    func = input("Enter the function (e.g., exp(x) - 2 - cos(exp(x) - 2)): OR (e.g., x**2 - 4*x + log(x)): ")
    interval_a = float(input("Enter the lower bound of the interval (a): "))
    interval_b = float(input("Enter the upper bound of the interval (b): "))
    
    try:
        solution, x_values, f_values, f_lambdified = secant_method(func, interval_a, interval_b)
        print(f"\nThe root is approximately at x = {solution:.5f}")
        plot_secant_method(func, x_values, f_values, solution)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
