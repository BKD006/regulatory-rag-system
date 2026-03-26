# -------------------------------
# Base image (Python 3.12)
# -------------------------------

FROM python:3.12-slim

# -------------------------------
# Environment settings
# -------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Logging mode for ECS
ENV LOG_MODE=prod
ENV LOG_LEVEL=INFO

# -------------------------------
# System dependencies
# Needed for numpy / transformers / psycopg / etc.
# -------------------------------

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ca-certificates \
    openssl \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# force IPv4 over IPv6
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

# -------------------------------
# Working directory
# -------------------------------

WORKDIR /app

# -------------------------------
# Copy requirements first (for caching)
# -------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# remove build tools after install
RUN apt-get purge -y gcc && apt-get autoremove -y

# -------------------------------
# Copy project files
# -------------------------------

COPY . .

# -------------------------------
# Expose port (ECS expects this)
# -------------------------------

EXPOSE 8000

# -------------------------------
# Start FastAPI
# -------------------------------

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]