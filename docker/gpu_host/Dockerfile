FROM python:3.12-slim

WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm10.0 --no-cache-dir

RUN mkdir ~/.imagineit

COPY worker.json ~/.imagineit/worker.json

COPY . .

RUN pip install '.[gpu]' --no-cache-dir

ENTRYPOINT ["python"]

CMD ["gpu_worker.py", "--key-file", "${GPU_WORKER_KEY_FILE}"]
