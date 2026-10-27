FROM python:3.11-slim

# Java is required by PySpark
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides $PORT at runtime
CMD streamlit run app_pyspark.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
