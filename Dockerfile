# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask (you can include other dependencies here if needed)
RUN pip install -r requirements.txt

# Set the command to run the Flask app when the container starts
CMD ["python", "main.py"]
