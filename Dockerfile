FROM python:3.9

COPY src/* .
COPY weights/ weights/
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 6077
RUN python3 serve.py