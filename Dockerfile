From python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CM["python3","app.py"]