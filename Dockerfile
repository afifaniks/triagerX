FROM python:3.10

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Install PyTorch with CUDA support
#RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

# Run the main.py script when the container launches
CMD ["fastapi", "dev", "main.py"]
