from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import io
from scipy.optimize import curve_fit

app = Flask(__name__)

##################################
# Hard-coded data & model setup
##################################
# Example data from your snippet
DEFAULT_GLUCOSE = np.array([1.2, 2.4, 3.6, 4.8])
MU_37 = np.array([0.0274, 0.0235, 0.0362, 0.041])  # hr^-1
MU_33 = np.array([0.011, 0.022, 0.032, 0.035])     # hr^-1

def log_func(x, a, b):
    """Logarithmic model: mu = a ln(x) + b."""
    return a * np.log(x) + b

# Fit the curves just once at startup, store in global
params_37, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_37)
params_33, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_33)

##################################
# Routes
##################################

@app.route('/')
def home():
    """
    Home page with:
    - Radio for temperature,
    - Slider for glucose,
    - Dropdown for days,
    - Text field for initial cell count,
    - The dynamic plot (via <img>).
    """
    return render_template('home.html')

@app.route('/plot')
def plot():
    """
    Generates the Matplotlib plot based on query parameters:
      ?temp=33 or ?temp=37
      &glucose=...
      &days=...
      &initial_count=...
    """
    # 1) Read query params (strings), convert to float
    temp_str = request.args.get('temp', '33')  # '33' or '37'
    try:
        glucose_val = float(request.args.get('glucose', '1.2'))
    except ValueError:
        glucose_val = 1.2

    try:
        days = float(request.args.get('days', '1'))
    except ValueError:
        days = 1.0

    try:
        initial_count = float(request.args.get('initial_count', '1000000'))
    except ValueError:
        initial_count = 1e6

    # 2) Hours from days
    hours = days * 24.0

    # 3) Pick parameters for the chosen temperature
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)  # hr^-1
        color = 'red'
        label = '37°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)  # hr^-1
        color = 'orange'
        label = '33°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_33)

    # 4) Compute final cell count after X days
    #    log(X) = mu*t + log(X0) => X(t) = X0 exp(mu t)
    final_cell_count = initial_count * np.exp(mu * hours)

    # 5) Estimate daily glucose needed (placeholder formula)
    daily_glucose_needed = final_cell_count * 1e-8

    # 6) Generate the figure
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)

    # For reference, plot the fitted curve across the range
    x_vals = np.linspace(1, 5, 100)
    if temp_str == '37':
        y_fit = log_func(x_vals, *params_37)
    else:
        y_fit = log_func(x_vals, *params_33)

    ax.plot(x_vals, y_fit, color=color, label=label)
    ax.scatter(DEFAULT_GLUCOSE, scatter_vals, color=color)

    # Mark chosen glucose
    ax.scatter([glucose_val], [mu], color='blue', zorder=5)
    ax.set_xlabel('Glucose (g/L)')
    ax.set_ylabel('Specific Growth Rate (hr^-1)')
    ax.set_title('Growth Rate vs Glucose')
    ax.legend()

    # Annotate chosen point
    ax.text(glucose_val + 0.1, mu, f"mu={mu:.4f}", color='blue')

    # Info text box
    info_text = (
        f"Days: {days:.1f}\n"
        f"Temp: {temp_str}°C\n"
        f"mu = {mu:.4f} hr^-1\n"
        f"Initial Count: {initial_count:.2e}\n"
        f"Final Count: {final_cell_count:.2e}\n"
        f"Daily Glucose Need: {daily_glucose_needed:.3f} g/day"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    # 7) Return figure as PNG
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

# Run the dev server
if __name__ == '__main__':
    app.run(debug=True)
