FROM python:3.12-slim

WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.10,<3.0' \
    && pip install --no-cache-dir -r backend/requirements.txt
COPY backend ./backend
COPY training ./training
COPY frontend/public/data ./frontend/public/data
COPY outputs ./outputs

ENV DPE_DB_PATH=/data/annotations.db
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
