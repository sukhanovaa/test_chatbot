FROM python:3.9.17-alpine3.17

COPY src/* .
COPY weights/ weights/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 6077
RUN python3 serve.py