# Single-process local deployment: FastAPI serves the built frontend.
# Build:  docker build -t heartseeker-med .
# Run:    docker run --rm -p 8765:8765 -v heartseeker-med-data:/data heartseeker-med

FROM node:20-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm ci && npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEARTSEEKER_MED_HOME=/data
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend ./backend
COPY --from=frontend /app/dist ./dist
RUN mkdir -p /data
EXPOSE 8765
# Mount /data to keep papers, SQLite and vectors on the host.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8765"]
