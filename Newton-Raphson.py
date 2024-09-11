import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Newton-Raphson Method Implementation
def newton_raphson_method(func, a, b, tol=1e-5, max_iter=100):
    x = sp.symbols('x')
    local_dict = {'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan}
    
    # Convert input string to symbolic expression and its derivative
    f = sp.sympify(func, locals=local_dict)
    f_prime = sp.diff(f, x)
    
    # Lambdify for numerical evaluation
    f_lambdified = sp.lambdify(x, f, 'numpy')
    f_prime_lambdified = sp.lambdify(x, f_prime, 'numpy')
    
    # Check if the function has different signs at a and b
    if f_lambdified(a) * f_lambdified(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints of the interval [a, b].")
    
    # Initial guess is the midpoint of the interval [a, b]
    x0 = (a + b) / 2
    
    # Lists to track the values of x and f(x) during iterations
    x_values = [x0]
    f_values = [f_lambdified(x0)]
    
    print(f"\n{'Iteration':<10}{'x':<20}{'f(x)':<20}{'f\'(x)':<20}")
    print(f"{'-'*60}")
    
    for i in range(max_iter):
        f_x0 = f_lambdified(x0)
        f_prime_x0 = f_prime_lambdified(x0)
        
        #Division By zero Check
        if f_prime_x0 == 0:
            raise ValueError("Derivative is zero, and thus 'f(x)/0'. Newton Raphson Failed.")

        print(f"{i+1:<10}{x0:<20.10f}{f_x0:<20.10f}{f_prime_x0:<20.10f}")
        
        if abs(f_x0) < tol:
            break
        
        # Newton-Raphson iteration: x_{n+1} = x_n - f(x_n) / f'(x_n)
        x1 = x0 - f_x0 / f_prime_x0
        x_values.append(x1)
        f_values.append(f_lambdified(x1))
        
        # Check convergence
        if abs(x1 - x0) < tol:
            break
        
        x0 = x1

    return x0, x_values, f_values, f_lambdified

# Plotting the function and Newton-Raphson iteration points
def plot_newton_raphson(func, x_values, f_values, solution):
    x_vals = np.linspace(min(x_values) - 1, max(x_values) + 1, 1000)
    f = sp.sympify(func, locals={'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan, 'ln': sp.ln})
    f_lambdified = sp.lambdify(sp.symbols('x'), f, 'numpy')
    y_vals = f_lambdified(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'f(x) = {func}')
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.scatter(x_values, f_values, color='red', zorder=5, label='Iteration Points')
    plt.scatter(solution, f_lambdified(solution), color='green', zorder=5, label=f'Solution x = {solution:.5f}')
    
    plt.title(f"Newton-Raphson Method for f(x) = {func}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # exp(x) + 2*exp(-x) + 2 * cos(x) - 6
    #x**2 - 4*x + 4 - ln(x)
    func = input("Enter the function (e.g., exp(x) - 2 - cos(exp(x) - 2)): OR (e.g., x**2 - 4*x + log(x)): ")
    interval_a = float(input("Enter the lower bound of the interval (a): "))
    interval_b = float(input("Enter the upper bound of the interval (b): "))
    
    try:
        solution, x_values, f_values, f_lambdified = newton_raphson_method(func, interval_a, interval_b)
        print(f"\nThe root is approximately at x = {solution:.5f}")
        plot_newton_raphson(func, x_values, f_values, solution)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
