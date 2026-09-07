FROM  pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git ffmpeg libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt


COPY . /app

ENTRYPOINT ["python", "main.py"]
