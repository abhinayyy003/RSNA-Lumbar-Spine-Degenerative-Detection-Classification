import os
import shutil
from flask import Flask, request, render_template, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pydicom
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max-limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model definition
class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize models and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dict = {
    'Sagittal T1': CustomEfficientNetV2().to(device),
    'Axial T2': CustomEfficientNetV2().to(device),
    'Sagittal T2/STIR': CustomEfficientNetV2().to(device)
}

# Load model weights
for model_name in models_dict:
    models_dict[model_name].load_state_dict(torch.load('efficient_netV2_ only_severity.pth', map_location=device))
    models_dict[model_name].eval()

# Specific series ID to type mapping
SERIES_TYPE_MAPPING = {
    '2828203845': 'Sagittal T1',
    '3481971518': 'Axial T2',
    '3844393089': 'Sagittal T2/STIR'
    # '557434766': 'Sagittal T1',
    # '1359869694': 'Axial T2',
    # '296385739': 'Sagittal T2/STIR'
}

# Mapping of conditions for each series type
CONDITION_MAPPING = {
    'Sagittal T1': {
        'left': 'left_neural_foraminal_narrowing',
        'right': 'right_neural_foraminal_narrowing'
    },
    'Axial T2': {
        'left': 'left_subarticular_stenosis',
        'right': 'right_subarticular_stenosis'
    },
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

def load_dicom(path):
    """Load and preprocess DICOM image"""
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data
    except Exception as e:
        logger.error(f"Error loading DICOM file {path}: {str(e)}")
        raise

def get_series_type(dicom_path):
    """Determine series type from DICOM metadata"""
    try:
        ds = pydicom.dcmread(dicom_path)
        if hasattr(ds, 'SeriesDescription'):
            series_desc = ds.SeriesDescription.lower()
            if 'sagittal' in series_desc and 't1' in series_desc:
                return 'Sagittal T1'
            elif 'axial' in series_desc and 't2' in series_desc:
                return 'Axial T2'
            elif 'sagittal' in series_desc and ('t2' in series_desc or 'stir' in series_desc):
                return 'Sagittal T2/STIR'
        series_id = os.path.basename(os.path.dirname(dicom_path))
        return SERIES_TYPE_MAPPING.get(series_id)
    except Exception as e:
        logger.error(f"Error reading DICOM file {dicom_path}: {str(e)}")
        return None

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def classify_condition(row):
    """Classify condition based on the highest probability."""
    if row['normal_mild'] >= row['moderate'] and row['normal_mild'] >= row['severe']:
        return 'Normal/Mild'
    elif row['moderate'] >= row['normal_mild'] and row['moderate'] >= row['severe']:
        return 'Moderate'
    else:
        return 'Severe'

def process_series(study_id, series_id, series_path, series_type):
    """Process a single series of DICOM images"""
    results = []
    submission_data = []

    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    for dicom_file in dicom_files:
        try:
            dicom_path = os.path.join(series_path, dicom_file)
            image = load_dicom(dicom_path)
            image_tensor = transform(image).unsqueeze(0).to(device)

            model = models_dict[series_type]
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0)

            conditions = CONDITION_MAPPING[series_type]
            if isinstance(conditions, dict):
                condition_list = [conditions['left'], conditions['right']]
            else:
                condition_list = [conditions]

            for condition in condition_list:
                for level in LEVELS:
                    row_id = f"{study_id}_{condition}_{level}"
                    result = {
                        'study_id': study_id,
                        'series_id': series_id,
                        'condition': condition.replace('_', ' ').title(),
                        'level': level.upper().replace('_', '/'),
                        'normal_mild': float(probs[0]),
                        'moderate': float(probs[1]),
                        'severe': float(probs[2])
                    }
                    results.append(result)
                    submission_data.append({
                        'row_id': row_id,
                        'normal_mild': float(probs[0]),
                        'moderate': float(probs[1]),
                        'severe': float(probs[2])
                    })

        except Exception as e:
            logger.error(f"Error processing DICOM {dicom_file}: {str(e)}")
            continue
            
    return results, submission_data

def process_study_folder(study_path):
    """Process complete study folder with all series"""
    study_id = os.path.basename(study_path)
    all_results = []
    all_submission_data = []

    logger.info(f"Processing study {study_id}")

    for series_id in os.listdir(study_path):
        series_path = os.path.join(study_path, series_id)
        if not os.path.isdir(series_path):
            continue

        series_type = SERIES_TYPE_MAPPING.get(series_id)
        if not series_type:
            logger.warning(f"Unknown series ID: {series_id}")
            continue

        logger.info(f"Processing series {series_id} of type {series_type}")

        results, submission_data = process_series(study_id, series_id, series_path, series_type)
        all_results.extend(results)
        all_submission_data.extend(submission_data)

    if all_submission_data:
        submission_df = pd.DataFrame(all_submission_data)
        submission_df = submission_df.groupby('row_id').mean().reset_index()
        submission_df['classification'] = submission_df.apply(classify_condition, axis=1)
    else:
        submission_df = pd.DataFrame(columns=['row_id', 'normal_mild', 'moderate', 'severe'])

    return all_results, submission_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'study_folder' not in request.files:
        return jsonify({'error': 'No folder uploaded'})

    files = request.files.getlist('study_folder')
    if not files:
        return jsonify({'error': 'No files selected'})

    try:
        study_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_study')
        if os.path.exists(study_dir):
            shutil.rmtree(study_dir)
        os.makedirs(study_dir)

        for file in files:
            if file.filename:
                relative_path = os.path.join(*file.filename.split(os.sep)[1:])
                file_path = os.path.join(study_dir, relative_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)

        results, submission_df = process_study_folder(study_dir)

        if submission_df.empty:
            return jsonify({'error': 'No valid predictions could be generated'})

        submission_path = os.path.join(app.config['UPLOAD_FOLDER'], 'submission.csv')
        submission_df.to_csv(submission_path, index=False)

        shutil.rmtree(study_dir)

        results_with_classification = [
            {
                'study_id': res['study_id'],
                'series_id': res['series_id'],
                'condition': res['condition'],
                'level': res['level'],
                'normal_mild': f"{res['normal_mild'] * 100:.2f}%",
                'moderate': f"{res['moderate'] * 100:.2f}%",
                'severe': f"{res['severe'] * 100:.2f}%"
            }
            for res in results
        ]

        return jsonify({
            'success': True,
            'results': results_with_classification,
            'submission_file': 'submission.csv'
        })

    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/download_submission')
def download_submission():
    submission_path = os.path.join(app.config['UPLOAD_FOLDER'], 'submission.csv')
    return send_file(submission_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
