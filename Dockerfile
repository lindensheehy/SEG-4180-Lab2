FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download a lightweight pre-trained segmentation model (optional, speeds up boot)
RUN python -c "import torchvision; torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)"

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]