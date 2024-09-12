import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Fixed-Point Iteration Method Implementation
def fixed_point_iteration(g_func, x0, tol=1e-2, max_iter=100):
    x = sp.symbols('x')
    local_dict = {'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan}
    
    # Convert the g(x) function to symbolic expression and lambdify it
    g = sp.sympify(g_func, locals=local_dict)
    g_lambdified = sp.lambdify(x, g, 'numpy')
    
    # Lists to track x-values during iterations
    x_values = [x0]
    
    print(f"\n{'Iteration':<10}{'x':<20}{'g(x)':<20}")
    print(f"{'-'*40}")
    
    for i in range(max_iter):
        x1 = g_lambdified(x0)
        x_values.append(x1)
        
        print(f"{i+1:<10}{x0:<20.10f}{x1:<20.10f}")
        
        # Check if the current guess is close enough to the solution
        if abs(x1 - x0) < tol:
            break
        
        x0 = x1
    
    return x0, x_values, g_lambdified

# Plotting the function and Fixed-Point Iteration points
def plot_fixed_point(g_func, x_values, solution):
    x_vals = np.linspace(min(x_values) - 1, max(x_values) + 1, 1000)
    g = sp.sympify(g_func, locals={'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tan': sp.tan})
    g_lambdified = sp.lambdify(sp.symbols('x'), g, 'numpy')
    y_vals = g_lambdified(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'g(x) = {g_func}')
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.scatter(x_values, [g_lambdified(x) for x in x_values], color='red', zorder=5, label='Iteration Points')
    plt.scatter(solution, g_lambdified(solution), color='green', zorder=5, label=f'Solution x = {solution:.5f}')
    
    plt.title(f"Fixed-Point Iteration for g(x) = {g_func}")
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    #f: x**4 - (3 * x**2) - 3
    #g: ((3 * x**2) + 3)**(1/4)

    #f: x**3 -x -1
    #g: (x + 1)**(1/3)

    # Prompt the user to input the function and the form g(x)
    print("Fixed-Point Iteration requires a function f(x) and a corresponding rearranged form g(x) such that x = g(x)")
    f_func = input("Enter the original function f(x) (e.g., exp(x) - x - 2): ")
    g_func = input("Enter the rearranged form g(x) (e.g., exp(x) - 2): ")
    
    # Ask for the interval and starting point
    interval_a = float(input("Enter the lower bound of the interval (a): "))
    interval_b = float(input("Enter the upper bound of the interval (b): "))
    x0 = float(input(f"Enter the initial guess (starting point within [{interval_a}, {interval_b}]): "))
    
    try:
        # Perform the fixed-point iteration
        solution, x_values, g_lambdified = fixed_point_iteration(g_func, x0)
        print(f"\nThe root is approximately at x = {solution:.5f}")
        
        # Plot the function and iterations
        plot_fixed_point(g_func, x_values, solution)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

