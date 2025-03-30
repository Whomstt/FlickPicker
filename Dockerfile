# Use the official Python image as the base image
FROM python:3.12.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN apt-get update && apt-get install git -y && apt-get install curl -y
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /code/

# Expose the application on port 8000
EXPOSE 8000

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]