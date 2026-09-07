FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

#Cloud Run / Vertex inject $PORT; default to 8080 for local runs
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
