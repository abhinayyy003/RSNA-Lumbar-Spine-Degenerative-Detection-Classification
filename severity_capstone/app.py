import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pydicom
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_conditions, num_levels, num_severities):
        super(CustomEfficientNetV2, self).__init__()
        self.base_model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.base_model.classifier[-1].in_features
        
        # Remove the original classifier
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Single prediction head for severity across all levels
        self.level_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_severities)
            ) for _ in range(num_levels)
        ])

    def forward(self, x):
        # Get base features
        features = self.base_model(x)
        features = features.view(features.size(0), -1)
        
        # Get predictions for each level
        outputs = []
        for head in self.level_heads:
            level_output = head(features)
            outputs.append(level_output)
            
        return outputs

# Constants for model configuration
NUM_SEVERITIES = 3  # Normal/Mild, Moderate, Severe
LEVELS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
NUM_LEVELS = len(LEVELS)

# Severity classes
severity_classes = ['Normal/Mild', 'Moderate', 'Severe']

# Load the single model for severity prediction
model_path = 'efficient_netV2_ only_severity.pth'

# Initialize and load the single model
model = CustomEfficientNetV2(
    num_conditions=1,  # Using 1 as we're focusing on severity
    num_levels=NUM_LEVELS,
    num_severities=NUM_SEVERITIES
).to(device)

try:
    model_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weights)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

# Image preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Load and process DICOM function
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def process_dicom_for_display(dicom_array):
    img = Image.fromarray(dicom_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    series_type = request.form.get('series_type')  # May not be needed unless expanding

    if file.filename == '' or not series_type:
        return jsonify({'error': 'Missing file or series type'})

    try:
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process DICOM image
        dicom_image = load_dicom(filepath)
        image_b64 = process_dicom_for_display(dicom_image)
        
        # Prepare image for model
        image = transform(dicom_image).unsqueeze(0).to(device)

        results = []
        
        with torch.no_grad():
            level_outputs = model(image)
            
            # Process predictions for each level
            for level_idx, level_output in enumerate(level_outputs):
                level_output = level_output.view(-1, 1, NUM_SEVERITIES)  # Adjust based on your model's output shape
                
                # Get severity predictions for each level
                condition_output = level_output[0, 0]  # Assuming a single condition model for severity
                severity_idx = torch.argmax(condition_output).item()
                
                result = {
                    'level': LEVELS[level_idx],
                    'severity': severity_classes[severity_idx]
                }
                results.append(result)

        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results,
            'image_data': image_b64
        })

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
