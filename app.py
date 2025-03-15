from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import yaml
from scipy.optimize import curve_fit
import random

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

@app.route('/', methods=['GET'])
def home():
    """
    Renders the home page (home.html).
    """
    return render_template('home.html')

@app.route('/plot')
def plot():
    """
    Returns a PNG image of the Growth Rate vs Glucose, 
    annotated with final cell density if user has requested multiple days.
    Query params:
      ?temp=33 or ?temp=37
      &glucose=...
      &days=...
      &initial_density=...
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
        initial_density = float(request.args.get('initial_density', '300000'))
    except ValueError:
        initial_density = 3e5

    try:
        initial_volume = float(request.args.get('volume', '2000'))
    except ValueError:
        initial_volume = 2000

    hours = days * 24.0

    # Calculate initial cell count
    initial_count = initial_density * initial_volume

    # Evaluate mu from the appropriate fit
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)
        color = 'red'
        label = '37°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)
        color = '#BBA14F'
        label = '33°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_33)

    # Compute final cell count with continuous model
    final_cell_count = initial_count * np.exp(mu * hours)
    final_cell_density = final_cell_count / initial_volume # Convert volume to liters

    # Fetch daily glucose needed values
    daily_data_response = daily_data()
    daily_data_json = daily_data_response.get_json()
    daily_glucose_needed_values = daily_data_json["daily_glucose_needed_values"]
    avg_daily_glucose_needed = np.mean(daily_glucose_needed_values)

    # PLOT: Growth Rate vs Glucose
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)

    x_vals = np.linspace(DEFAULT_GLUCOSE.min(), DEFAULT_GLUCOSE.max(), 100)
    if temp_str == '37':
        y_fit = log_func(x_vals, *params_37)
    else:
        y_fit = log_func(x_vals, *params_33)

    # Plot fitted curve
    ax.plot(x_vals, y_fit, color=color, label=label)
    # Plot scatter for known points
    ax.scatter(DEFAULT_GLUCOSE, scatter_vals, color=color)
    # Mark the chosen glucose
    ax.scatter([glucose_val], [mu], color='blue', zorder=5)

    ax.set_xlabel('Glucose (g/L)')
    ax.set_ylabel('Specific Growth Rate (hr^-1)')
    ax.set_title('Growth Rate vs Glucose')
    ax.legend()

    # Annotate chosen point
    ax.text(glucose_val + 0.1, mu, f"mu={mu:.4f}", color='blue')

    # Info text box
    info_text = (
        f"Days: {days}\n"
        f"Temp: {temp_str}°C\n"
        f"mu = {mu:.4f} hr^-1\n"
        f"Initial Density: {initial_density:.2e} cells/mL\n"
        f"Final Density: {final_cell_density:.2e} cells/mL\n"
        f"Avg Daily Glucose Need: {avg_daily_glucose_needed:.3f} g/day"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    plt.tight_layout()

    # Return as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/daily_data')
def daily_data():
    """
    Returns JSON with daily predictions for the selected parameters.
    We'll do a discrete daily approach:
      X_{i} = X_{i-1} * exp(mu * 24)  (if you want to tie to log_func)
    or
      X_{i} = X_{i-1} * (1 + daily_sgr)
    and also compute daily glucose required.

    Query params:
      ?temp=...
      &glucose=...
      &days=...
      &initial_density=...
      &volume=...
    """
    temp_str = request.args.get('temp', '33')
    try:
        glucose_val = float(request.args.get('glucose', '1.2'))
    except ValueError:
        glucose_val = 1.2

    try:
        days = int(request.args.get('days', '1'))
    except ValueError:
        days = 1

    try:
        initial_density = float(request.args.get('initial_density', '300000'))
    except ValueError:
        initial_density = 3e5

    try:
        initial_volume = float(request.args.get('volume', '2000'))
    except ValueError:
        initial_volume = 2000

    # Calculate initial cell count
    initial_count = initial_density * initial_volume  # Convert volume to liters

    # Evaluate mu from the appropriate fit
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)

    # Option 1: tie daily growth to the hourly mu from log_func
    #   => daily growth factor = exp(mu * 24)
    daily_growth_factor = np.exp(mu * 24)

    # We'll produce daily predictions in a table:
    # For day=1..days, X[day] = X[day-1] * daily_growth_factor
    # We'll also compute daily glucose usage as
    #   daily_glc = (X[day] - X[day-1]) * (1e-8)   (for instance)
    # Feel free to adjust the formula if you prefer.

    results = []
    daily_glucose_needed_values = []
    X = initial_count
    glucose_consumption_rate = 24 * 0.2 #pmol/cell/day
    MW_GLUCOSE = 180
    VOLUME = initial_volume / 1000 #L
    glucose_to_lactate_conversion_factor = 0.8
    remaining_glucose = glucose_val
    lactate_level = 0

    # Add day 0 entry
    results.append({
        "day": 0,
        "predicted_density": initial_density,
        "daily_glucose_concentration": remaining_glucose,
        "daily_glucose_needed": 0,
        "lactate_level": lactate_level
    })

    for d in range(1, days+1):
        old_X = X
        X = X * daily_growth_factor
        # daily usage is difference in cell number * consumption factor
        daily_glc_needed = (X - old_X) * glucose_consumption_rate * 1e-12 * MW_GLUCOSE #g
        daily_glucose_needed_values.append(daily_glc_needed)
        remaining_glucose -= daily_glc_needed / VOLUME
        # add lactate production
        lactate_level += daily_glc_needed / MW_GLUCOSE * glucose_to_lactate_conversion_factor * 2 * 1000 / VOLUME #mmol/L
        cell_density = X / (VOLUME*1000)  # Calculate cell density X/mL
        results.append({
            "day": d,
            "predicted_density": cell_density,
            "daily_glucose_concentration": remaining_glucose,
            "daily_glucose_needed": daily_glc_needed,
            "lactate_level": lactate_level
        })

    return jsonify({
        "results": results,
        "daily_glucose_needed_values": daily_glucose_needed_values
    })

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
    except Exception as e:
        return f"Error reading file: {e}", 400

    return "File submitted successfully!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
