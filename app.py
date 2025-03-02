from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
from scipy.optimize import curve_fit

app = Flask(__name__)

# Store fitted data (in a production app, you'd use session or a DB)
APP_DATA = {
    'params_33': None,
    'params_37': None,
    'glucose_range': None,
}

############################################
# Growth Model: y = a * ln(x) + b
############################################

def log_func(x, a, b):
    """Logarithmic model from your earlier code."""
    return a * np.log(x) + b

def fit_growth_rates_from_excel(df_cell_density, df_glucose):
    """
    Example function:
      - parse your Excel data,
      - compute or curve-fit parameters,
      - return them.
    We'll use your snippet's arrays for demonstration.
    """

    # Example data from your snippet:
    glucose = np.array([1.2, 2.4, 3.6, 4.8])
    mu_37   = np.array([0.0274, 0.0235, 0.0362, 0.041])  # hr^-1
    mu_33   = np.array([0.011, 0.022, 0.032, 0.035])     # hr^-1

    # Fit the log_func to each temperature's data
    params_37, _ = curve_fit(log_func, glucose, mu_37)
    params_33, _ = curve_fit(log_func, glucose, mu_33)

    return params_33, params_37, glucose

############################################
# Routes
############################################

@app.route('/', methods=['GET'])
def home():
    """Landing page with an upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle Excel upload:
      - read the sheets,
      - fit the model,
      - store parameters,
      - redirect to plot page.
    """
    excel_file = request.files.get('excel_file')
    if not excel_file:
        return "No file uploaded. Please go back and upload a file."

    # Read your sheets by name
    df_cell_density = pd.read_excel(excel_file, sheet_name='Cell Density')
    df_glucose      = pd.read_excel(excel_file, sheet_name='Glucose Concentration')

    # Fit parameters
    params_33, params_37, glucose_range = fit_growth_rates_from_excel(
        df_cell_density,
        df_glucose
    )

    # Store them
    APP_DATA['params_33'] = params_33
    APP_DATA['params_37'] = params_37
    APP_DATA['glucose_range'] = glucose_range

    return redirect(url_for('plot_page'))

@app.route('/plot_page')
def plot_page():
    """
    Page that shows:
    - temperature radio
    - glucose slider
    - days dropdown
    - the dynamic plot
    """
    return render_template('plot.html')

@app.route('/plot')
def plot():
    """
    Generates a PNG on-the-fly based on query parameters:
      ?temp=33 or ?temp=37
      &glucose= (slider value)
      &days= (dropdown selection)
    """
    # Retrieve query params
    temp_str     = request.args.get('temp', '33')
    glucose_str  = request.args.get('glucose', '1.2')
    days_str     = request.args.get('days', '1')

    # Convert to numeric
    try:
        glucose_val = float(glucose_str)
    except ValueError:
        glucose_val = 1.2

    try:
        days = float(days_str)
    except ValueError:
        days = 1.0

    # Hours from days
    hours = days * 24.0

    # Retrieve fitted parameters
    params_33 = APP_DATA['params_33']
    params_37 = APP_DATA['params_37']
    glucose   = APP_DATA['glucose_range']

    if any(x is None for x in [params_33, params_37, glucose]):
        return "No fitted data. Please upload an Excel file first."

    #######################################
    # 1) Determine the growth rate (mu)
    #######################################
    # We have a log-based fit: mu = a * ln(glucose_val) + b
    # Evaluate with the chosen temperature
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)  # hr^-1
        color = 'red'
        label= '37°C Fit'
        scatter_vals = log_func(glucose, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)  # hr^-1
        color = 'orange'
        label= '33°C Fit'
        scatter_vals = log_func(glucose, *params_33)

    #######################################
    # 2) Compute final cell count after X days
    #######################################
    # We'll define some initial cell count X0 (arbitrary).
    # For demonstration, let's say X0 = 1e6 cells/mL (just an example).
    X0 = 1e6
    # log(X) = mu*t + log(X0) => X(t) = X0 * exp(mu*t)
    # t is in hours, mu is hr^-1
    final_cell_count = X0 * np.exp(mu * hours)

    #######################################
    # 3) Estimate daily glucose needed
    #######################################
    # Real logic depends on biomass yields, stoichiometry, etc.
    # For demonstration, let's propose a simple placeholder formula:
    # daily_glucose_needed = final_cell_count * 1e-8 (units: grams?)
    # Adjust as you wish.
    daily_glucose_needed = final_cell_count * 1e-8

    #######################################
    # 4) Generate the figure
    #######################################
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)

    # For visual context: plot the fitted curve vs. glucose for a wide range:
    x_vals = np.linspace(glucose.min(), glucose.max(), 100)
    if temp_str == '37':
        y_fit = log_func(x_vals, *params_37)
    else:
        y_fit = log_func(x_vals, *params_33)
    ax.plot(x_vals, y_fit, color=color, label=label)

    # Show the original data points at those 4 glucose values
    ax.scatter(glucose, scatter_vals, color=color)

    # Mark the user's chosen glucose
    ax.scatter([glucose_val], [mu], color='blue', zorder=5)
    ax.set_xlabel('Glucose (g/L)')
    ax.set_ylabel('Specific Growth Rate (hr^-1)')
    ax.set_title('Growth Rate vs. Glucose')

    ax.legend()

    # Annotate the chosen glucose
    ax.text(glucose_val + 0.1, mu, f"mu={mu:.4f}", color='blue')

    # Show a text box about the final cell count after the chosen days
    info_text = (
        f"Days: {days:.0f}\n"
        f"mu = {mu:.4f} hr^-1\n"
        f"Final Count: {final_cell_count:.2e} cells\n"
        f"Daily Glucose Need: {daily_glucose_needed:.3f} g/day (demo)"
    )
    # Place it in the top-left corner
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    # Save figure to PNG
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

###########################
# Run the development app
###########################
if __name__ == '__main__':
    app.run(debug=True)
