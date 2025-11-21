FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh
# Backend http://127.0.0.1:5000
EXPOSE 5000
# Frontend http://0.0.0.0:8501
EXPOSE 8501 

CMD ["/app/start.sh"]