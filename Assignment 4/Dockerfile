FROM python:3.12-bookworm

WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY app.py .
COPY score.py .
COPY best_model.pkl .

EXPOSE 5000

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "app.py"]

