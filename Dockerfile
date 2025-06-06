FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir sentence-transformers fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "embedding_server:app", "--host", "0.0.0.0", "--port", "8000"]

