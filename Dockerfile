# Minimal CPU image. Mount a persistent volume at /data (SQLite + private PDFs). Build context = the repository root (backend/ and frontend/).
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend frontend
ENV QUELLUM_ENV=production QUELLUM_DATA_DIR=/data QUELLUM_FRONTEND_DIR=/app/frontend
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
