FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pdf_agent ./pdf_agent
COPY data ./data

ENTRYPOINT ["python", "-m", "pdf_agent.cli"]
