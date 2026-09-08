FROM python:3.13-slim (last push 2 days ago)
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80501

CMD ["streamlit", "run", "app.py", "--server.port=80501", "--server.address=0.0.0.0"]