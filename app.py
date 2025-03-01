from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
from scipy.optimize import curve_fit

app = Flask(__name__)

# Global or session-based data store; for demonstration only.
APP_DATA = {
    'params_33': None,
    'params_37': None,
    'glucose_range': None,
}

#######################
# 1. Data/model logic #
#######################

def log_func(x, a, b):
    """Example log-based model. Adjust as needed."""
    return a * np.log(x) + b

def fit_growth_rates_from_excel(df_cell_density, df_glucose):
    """
    Example function: parse your data from the dataframes, 
    compute or curve-fit the growth rate parameters, 
    and return them.
    """
    # For demonstration, just using the same data from your snippet:
    glucose = np.array([1.2, 2.4, 3.6, 4.8])
    mu_37   = np.array([0.0274, 0.0235, 0.0362, 0.041])
    mu_33   = np.array([0.011, 0.022, 0.032, 0.035])

    # Fit each temperature
    params_37, _ = curve_fit(log_func, glucose, mu_37)
    params_33, _ = curve_fit(log_func, glucose, mu_33)

    return params_33, params_37, glucose

#########################
# 2. Routes and Views   #
#########################

@app.route('/', methods=['GET'])
def home():
    """Simple landing page with upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle the file upload, parse the Excel, fit the model, then go to the interactive plot page."""
    excel_file = request.files.get('excel_file')
    if not excel_file:
        return "No file uploaded. Please go back and upload a file."

    # Read the required sheets by name (adjust names as needed)
    df_cell_density = pd.read_excel(excel_file, sheet_name='Cell Density')
    df_glucose      = pd.read_excel(excel_file, sheet_name='Glucose Concentration')

    # Fit the parameters from the data
    params_33, params_37, glucose_range = fit_growth_rates_from_excel(
        df_cell_density, df_glucose
    )

    # Store them for later use
    APP_DATA['params_33'] = params_33
    APP_DATA['params_37'] = params_37
    APP_DATA['glucose_range'] = glucose_range

    # Redirect to the plotting page
    return redirect(url_for('plot_page'))

@app.route('/plot_page')
def plot_page():
    """
    The page containing slider/radio buttons and an <img> 
    that dynamically queries the /plot PNG route.
    """
    return render_template('plot.html')

@app.route('/plot')
def plot():
    """
    Generates a PNG plot based on query parameters: 
        ?temp=33 or ?temp=37, and &glucose=...
    """
    # Retrieve query parameters
    temp_str = request.args.get('temp', '33')  # '33' or '37'
    try:
        glucose_val = float(request.args.get('glucose', '1.2'))
    except ValueError:
        glucose_val = 1.2

    # Retrieve stored fit parameters
    params_33 = APP_DATA['params_33']
    params_37 = APP_DATA['params_37']
    glucose   = APP_DATA['glucose_range']

    if any(x is None for x in [params_33, params_37, glucose]):
        return "No fitted data. Upload an Excel file first."

    # Prepare data for plotting
    x_vals = np.linspace(glucose.min(), glucose.max(), 100)

    if temp_str == '37':
        y_fit = log_func(x_vals, *params_37)
        y_data = log_func(glucose_val, *params_37)
        scatter_vals = log_func(glucose, *params_37)
        color = 'red'
        label= '37°C'
    else:
        y_fit = log_func(x_vals, *params_33)
        y_data = log_func(glucose_val, *params_33)
        scatter_vals = log_func(glucose, *params_33)
        color = 'orange'
        label= '33°C'

    # Plot
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)
    ax.plot(x_vals, y_fit, color=color, label=f'{label} Fit')
    ax.scatter(glucose, scatter_vals, color=color)
    ax.scatter([glucose_val], [y_data], color='blue', zorder=5)
    ax.set_xlabel('Initial Glucose (g/L)')
    ax.set_ylabel('Specific Growth Rate (h⁻¹)')
    ax.set_title('Specific Growth Rate vs Glucose')
    ax.legend()

    ax.text(glucose_val+0.1, y_data, f"{y_data:.4f}", color='blue')

    # Convert to PNG
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

####################
# 3. Run the App   #
####################
if __name__ == '__main__':
    app.run(debug=True)
