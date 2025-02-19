# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire content of the repo (including the src folder) into the container
COPY . /app/

# Expose the port your application will run on (if needed)
EXPOSE 8080

# Set the command to run the application (adjust to your script)
CMD ["python", "/app/src/llm1.py"]
