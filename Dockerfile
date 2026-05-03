FROM bitnami/spark:latest

RUN pip install mlflow delta-spark datasets pandas

WORKDIR /app