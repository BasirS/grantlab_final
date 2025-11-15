# using python 3.11 slim image as base
FROM python:3.11-slim

# setting working directory
WORKDIR /app

# installing system dependencies required for python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# copying requirements file
COPY requirements.txt .

# installing python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copying application code
COPY src/ ./src/
COPY .env.example .env

# creating upload directory
RUN mkdir -p /app/data/uploads

# exposing port 8000 for fastapi application
EXPOSE 8000

# setting environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# running application with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
