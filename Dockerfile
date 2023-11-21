# Use a base image with Python installed
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from your project to the container
COPY . .

# Set the entrypoint command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
