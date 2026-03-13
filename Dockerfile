FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY dax_bot/ dax_bot/
COPY telegram_cmd.py .

# Create data directory
RUN mkdir -p data

# Healthcheck: verify the bot process is running
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import os, signal; pid=int(open('/tmp/asrs.pid').read()); os.kill(pid, 0)" || exit 1

CMD ["python", "-m", "dax_bot.main"]
