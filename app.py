from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import yaml
from scipy.optimize import curve_fit

app = Flask(__name__)

##################################
# Hard-coded data & initial fits
##################################
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

DEFAULT_GLUCOSE = np.array(config["default_glucose"])
MU_37 = np.array(config["mu_37"])
MU_33 = np.array(config["mu_33"])

def log_func(x, a, b):
    """Logarithmic growth model: mu = a ln(x) + b."""
    return a * np.log(x) + b

params_37, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_37)
params_33, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_33)

##################################
# Routes
##################################

@app.route('/', methods=['GET'])
def home():
    """
    The home page shows:
    - Temperature radio buttons,
    - Glucose slider,
    - Days dropdown,
    - Initial cell count,
    - Plot image,
    - File upload form (AJAX) for refined curve fitting.
    """
    return render_template('home.html')

@app.route('/plot')
def plot():
    """
    Generates a plot based on query params:
      ?temp=33 or ?temp=37
      &glucose=...
      &days=...
      &initial_count=...
    Returns a PNG.
    """
    temp_str = request.args.get('temp', '33')
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

    hours = days * 24.0

    # Evaluate mu
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)
        color = 'red'
        label = '37°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)
        color = 'orange'
        label = '33°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_33)

    final_cell_count = initial_count * np.exp(mu * hours)
    daily_glucose_needed = final_cell_count * 1e-8

    # Plot
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)

    x_vals = np.linspace(DEFAULT_GLUCOSE.min(), DEFAULT_GLUCOSE.max(), 100)
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

    # Info box
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

    # Return as PNG
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/upload', methods=['POST'])
def upload():
    """
    AJAX endpoint to read the file, ensuring it's valid.
    Return a success message WITHOUT redirecting.
    """
    excel_file = request.files.get('excel_file')
    if not excel_file:
        return "No file chosen. Please select a file.", 400

    # Just read it to confirm it's valid; not used further.
    try:
        df = pd.read_excel(excel_file)
        # If we wanted to do more, we would parse & re-fit here
    except Exception as e:
        return f"Error reading file: {e}", 400

    return "File submitted successfully!", 200

################################
# Run dev server
################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

