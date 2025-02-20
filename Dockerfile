# Use an official Python runtime as a parent image
FROM python:3.11.9

# Copy the entire content of the repo (including the src folder) into the container
COPY . /app/

# Set the working directory in the container
WORKDIR /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirem

# Make sure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# Run your entrypoint script
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
#ENTRYPOINT ./entrypoint.sh
