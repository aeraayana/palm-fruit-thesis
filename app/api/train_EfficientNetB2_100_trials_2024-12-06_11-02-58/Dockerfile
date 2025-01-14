# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    setuptools wheel \
    autogluon --extra-index-url https://download.pytorch.org/whl/cpu \
    'fastapi[standard]'

# Expose the application port
EXPOSE 8000

# Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
