FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL authors="ParrotXray"
LABEL description="NetGuardia for training models"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    tini \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --upgrade pip \
    && pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

RUN chmod +x main.py

CMD ["/usr/bin/tini", "--", "./main.py", "--help"]