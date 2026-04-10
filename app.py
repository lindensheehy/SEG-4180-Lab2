from flask import Flask, request, jsonify
from waitress import serve
import os
import torch
from torchvision import models, transforms
from PIL import Image
import io
from dotenv import load_dotenv

# 1. Load Secrets (Lab 2 Requirement)
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default-dev-key")

app = Flask(__name__)

# 2. Load Segmentation Model (Lab 2 Requirement)
print("Loading Custom Segmentation Model...")

# Initialize the architecture with 2 classes (matching your training script)
model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=2)

# Load your fine-tuned weights
model_path = os.path.join(os.path.dirname(__file__), 'house_model.pth')
if os.path.exists(model_path):
    # map_location='cpu' ensures it works even if Docker isn't using a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Custom weights loaded successfully.")
else:
    print("WARNING: house_model.pth not found. The model will output random noise.")

model.eval()

# Transformation for the incoming image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Lab 1 Fix: Health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# 4. Lab 1 Fix: Safe request handling & Model Inference
@app.route('/predict', methods=['POST'])
def predict():
    # Secrets Injection Check
    client_key = request.headers.get("X-API-Key")
    if client_key != API_SECRET_KEY:
        return jsonify({"error": "Unauthorized. Invalid or missing API Key."}), 401

    # Safe parsing: ensure a file is attached
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Process Image
        input_image = Image.open(file.stream).convert("RGB")
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        
        # Calculate percentage of pixels classified as class '1' (or whatever your house class is)
        house_pixels = (output_predictions == 1).sum().item()
        total_pixels = output_predictions.numel()

        return jsonify({
            "message": "Segmentation successful",
            "house_pixel_ratio": house_pixels / total_pixels
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    serve(app, host='0.0.0.0', port=port)