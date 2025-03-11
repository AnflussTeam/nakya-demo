# Use a slim Python image to keep the final size smaller
FROM python:3.12-slim


# Change working directory inside the container
WORKDIR /app

# Install system dependencies if needed (optional)
# RUN apt-get update && apt-get install -y <any system deps> && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first, so Docker can cache the pip install step
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port that Flask uses (default 5000)
EXPOSE 8000

# Define the command to run your Flask app
CMD ["python", "app.py"]
