# Step 1: Base Image - Start with a lightweight, official Python image.
FROM python:3.11-slim

# Step 2: Set Working Directory - All subsequent commands will run from here.
WORKDIR /app

# Step 3: Copy requirements and install dependencies
# We copy this first to leverage Docker's caching. This layer only rebuilds
# if requirements.txt changes, speeding up subsequent builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy Application Code
# Copy the rest of the application files into the /app directory in the image.
COPY . .

# Step 5: Expose Port
# Inform Docker that the container listens on port 5000.
# Render will use this to route traffic to your app.
EXPOSE 5000

# Step 6: Command to Run
# Run the app using the Gunicorn production server. It listens on all interfaces (0.0.0.0)
# on the port specified by the PORT environment variable, which Render will provide.
# We fall back to 5000 if PORT isn't set.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]