
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .glaucoma {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    .normal {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        color: white;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 0.5s ease;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .sidebar-info {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CLAHE Transform (same as training)
# ============================================
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = self.clahe.apply(img_np)
        return Image.fromarray(img_np)

# ============================================
# Model Loading with Caching
# ============================================
@st.cache_resource
def load_model(model_path='models/best_model.pth'):
    """Load the trained DenseNet model with caching for performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model architecture
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # Ensure model is on correct device
        model.eval()
        return model, device, True
    else:
        return None, device, False

def get_transforms():
    """Get the same transforms used during training"""
    return transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============================================
# Prediction Function
# ============================================
def predict_image(model, image, device):
    """Make prediction on a single image"""
    transform = get_transforms()

    # Apply CLAHE and transform
    image_transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_transformed)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# ============================================
# Main Application
# ============================================
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>👁️ Glaucoma Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 18px;'>AI-Powered Retinal Image Analysis using DenseNet + CLAHE</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ System Info</h2>", unsafe_allow_html=True)

        # Device info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.markdown(f"<div class='sidebar-info'><b>Device:</b> {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}</div>", unsafe_allow_html=True)
        if torch.cuda.is_available():
            st.markdown(f"<div class='sidebar-info'><b>GPU:</b> {torch.cuda.get_device_name(0)}</div>", unsafe_allow_html=True)

        # Model status
        model, device, model_loaded = load_model()
        if model_loaded:
            st.success("✅ Model Loaded Successfully")
        else:
            st.error("❌ Model Not Found")
            st.info("Please run train.py first to generate the model.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # About section
        st.markdown("<h3>ℹ️ About</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='sidebar-info'>
        <b>Architecture:</b> DenseNet-121<br>
        <b>Preprocessing:</b> CLAHE<br>
        <b>Input Size:</b> 224×224<br>
        <b>Classes:</b> Normal, Glaucoma
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3>📊 Performance Metrics</h3>", unsafe_allow_html=True)
        if os.path.exists('results/metrics.json'):
            import json
            with open('results/metrics.json', 'r') as f:
                metrics = json.load(f)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='metric-card'><b>Accuracy</b><br>{metrics['accuracy']*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><b>Precision</b><br>{metrics['precision']*100:.1f}%</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><b>Recall</b><br>{metrics['recall']*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><b>F1-Score</b><br>{metrics['f1_score']*100:.1f}%</div>", unsafe_allow_html=True)
        else:
            st.info("Train the model to see metrics")

    # Main content
    if not model_loaded:
        st.warning("⚠️ Please train the model first by running `python train.py`")
        return

    # File uploader
    st.markdown("<h3 style='color: #2c3e50;'>📤 Upload Retinal Image</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a retinal image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a fundus/retinal image for glaucoma detection"
        )

    with col2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("<b>📋 Instructions:</b><br>1. Upload a clear retinal fundus image<br>2. The system will apply CLAHE enhancement<br>3. DenseNet model will classify the image<br>4. Results show Normal or Glaucoma with confidence", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')

        # Create columns for display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='text-align: center; color: #2c3e50;'>Original Image</h4>", unsafe_allow_html=True)
            st.image(image, use_column_width=True, caption=f"Size: {image.size}")

        # Apply CLAHE for display
        clahe_transform = CLAHETransform()
        clahe_image = clahe_transform(image)

        with col2:
            st.markdown("<h4 style='text-align: center; color: #2c3e50;'>CLAHE Enhanced</h4>", unsafe_allow_html=True)
            st.image(clahe_image, use_column_width=True, caption="Contrast Limited Adaptive Histogram Equalization")

        # Prediction button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 Analyze Image", use_container_width=True)

        if predict_button:
            with st.spinner('🧠 Analyzing image with DenseNet...'):
                # Make prediction
                prediction, confidence, probabilities = predict_image(model, image, device)

                # Results section
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #2c3e50;'>📊 Analysis Results</h2>", unsafe_allow_html=True)

                # Result box
                if prediction == 1:
                    result_class = "Glaucoma Detected"
                    result_style = "glaucoma"
                    emoji = "⚠️"
                    description = "The model has detected signs of glaucoma. Please consult an ophthalmologist for further evaluation."
                else:
                    result_class = "Normal"
                    result_style = "normal"
                    emoji = "✅"
                    description = "The model indicates a normal retinal image. Regular check-ups are still recommended."

                st.markdown(f"<div class='result-box {result_style}'>{emoji} {result_class}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 16px;'>{description}</p>", unsafe_allow_html=True)

                # Confidence visualization
                st.markdown("<h4 style='color: #2c3e50;'>Confidence Scores</h4>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    normal_conf = probabilities[0] * 100
                    st.markdown(f"<b>Normal:</b> {normal_conf:.2f}%", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {normal_conf}%; background: linear-gradient(90deg, #26de81, #20bf6b);'></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    glaucoma_conf = probabilities[1] * 100
                    st.markdown(f"<b>Glaucoma:</b> {glaucoma_conf:.2f}%", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {glaucoma_conf}%; background: linear-gradient(90deg, #ff6b6b, #ee5a24);'></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional info
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("<div class='info-card' style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown(f"<b>Prediction Confidence</b><br><span style='font-size: 24px; color: #667eea;'>{confidence*100:.2f}%</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<div class='info-card' style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown(f"<b>Analysis Time</b><br><span style='font-size: 24px; color: #667eea;'>{datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    st.markdown("<div class='info-card' style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown(f"<b>Model</b><br><span style='font-size: 24px; color: #667eea;'>DenseNet-121</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Disclaimer
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("⚠️ **Disclaimer:** This system is for research purposes only. Always consult a qualified ophthalmologist for medical diagnosis and treatment decisions.")

if __name__ == '__main__':
    main()