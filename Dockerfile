FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py ./app.py
COPY build ./static_frontend

# Create checkpoints folder (models uploaded separately via HF UI)
RUN mkdir -p checkpoints

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python", "app.py"]
