FROM python:3.11-slim

# System deps for image decoding (Pillow/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Hugging Face Spaces run containers as uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch first — keeps the image small (the default CUDA wheels are
# ~2.5GB and would blow the Space build limits). requirements.txt then sees
# torch already satisfied and skips it.
RUN pip install --no-cache-dir --user \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV MODEL_PATH=checkpoints/combined_final.pt
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
