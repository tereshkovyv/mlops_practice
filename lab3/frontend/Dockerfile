FROM python:3.11-slim

WORKDIR /app

COPY frontend/static /app

EXPOSE 3000
CMD ["python", "-m", "http.server", "3000", "--bind", "0.0.0.0"]
