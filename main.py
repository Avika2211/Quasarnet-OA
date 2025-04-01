from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import astropy.io.fits as fits
import io
import numpy as np
import logging
import time
import os
from werkzeug.utils import secure_filename

# Configuration
class Config:
    MODEL_PATH = "spectral_classifier.pth"
    ALLOWED_EXTENSIONS = {'fits', 'fit'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    INPUT_SIZE = 1000
    CLASSES = ['Quasar', 'Star', 'Black Hole']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
class SpectralClassifier(nn.Module):
    def __init__(self, num_classes=3, input_size=1000):
        super().__init__()
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # Transformer
        self.embedding = nn.Linear(64, 128)
        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=3)
        
        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(128)

    def forward(self, x):
        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Transformer prep
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        # Transformer
        x = self.transformer(x)
        x = x.mean(dim=0)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Flask App
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def load_model():
    model = SpectralClassifier(num_classes=len(Config.CLASSES), input_size=Config.INPUT_SIZE)
    try:
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise

def process_spectrum(file_stream):
    try:
        with fits.open(io.BytesIO(file_stream.read())) as hdul:
            flux_fields = ['FLUX', 'flux', 'SPEC', 'spec']
            spectrum = None
            
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    for field in flux_fields:
                        if field in hdu.data.names:
                            spectrum = hdu.data[field]
                            break
                    if spectrum is not None:
                        break
            
            if spectrum is None:
                spectrum = hdul[0].data
            
            spectrum = np.nan_to_num(spectrum.astype(np.float32))
            min_val, max_val = np.min(spectrum), np.max(spectrum)
            
            if np.isclose(min_val, max_val):
                spectrum = np.zeros_like(spectrum)
            else:
                spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
            
            if len(spectrum) != Config.INPUT_SIZE:
                x_new = np.linspace(0, 1, Config.INPUT_SIZE)
                x_old = np.linspace(0, 1, len(spectrum))
                spectrum = np.interp(x_new, x_old, spectrum)
            
            return torch.tensor(spectrum, dtype=torch.float32)
    except Exception as e:
        logging.error(f"Spectrum processing error: {str(e)}")
        raise

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 400
    
    try:
        start_time = time.time()
        spectrum = process_spectrum(file.stream)
        spectrum = spectrum.unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = model(spectrum)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            response = {
                "prediction": Config.CLASSES[pred_idx.item()],
                "confidence": round(confidence.item(), 4),
                "probabilities": {
                    cls: round(prob.item(), 4) 
                    for cls, prob in zip(Config.CLASSES, probs[0])
                },
                "processing_time": round(time.time() - start_time, 4),
                "model": os.path.basename(Config.MODEL_PATH),
                "device": str(Config.DEVICE)
            }
            
            return jsonify(response)
    
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return jsonify({"error": "Classification failed", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(Config.DEVICE)
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        "input_size": Config.INPUT_SIZE,
        "classes": Config.CLASSES,
        "device": str(Config.DEVICE)
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    try:
        model = load_model()
        logging.info(f"Model loaded on {Config.DEVICE}")
        
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        raise
