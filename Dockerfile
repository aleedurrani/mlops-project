FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY templates/ templates/
COPY models/ models/
COPY data/ data/

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]