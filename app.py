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
    annotated with final cell count if user has requested multiple days.
    Query params:
      ?temp=33 or ?temp=37
      &glucose=...
      &days=...
      &initial_count=...
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

    # Evaluate mu from the appropriate fit
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

    # Compute final cell count with continuous model
    final_cell_count = initial_count * np.exp(mu * hours)
    daily_glucose_needed = final_cell_count * 1e-8

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
        f"Initial Count: {initial_count:.2e}\n"
        f"Final Count: {final_cell_count:.2e}\n"
        f"Daily Glucose Need: {daily_glucose_needed:.3f} g/day"
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
      &initial_count=...
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
        initial_count = float(request.args.get('initial_count', '1000000'))
    except ValueError:
        initial_count = 1e6

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
    X = initial_count
    for d in range(1, days+1):
        old_X = X
        X = X * daily_growth_factor
        # daily usage is difference in cell number * consumption factor
        daily_glc_needed = (X - old_X) * 1e-8
        results.append({
            "day": d,
            "predicted_cells": X,
            "daily_glucose_needed": daily_glc_needed
        })

    return jsonify(results)

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
