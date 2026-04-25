import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
import warnings
import logging

# ====== PERFORMANCE FIX ======
torch.set_num_threads(1)

warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('streamlit').setLevel(logging.ERROR)

st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.glaucoma { background: #ff6b6b; color: white; }
.normal { background: #26de81; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CLAHE Transform
# ============================================
class CLAHETransform:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)

# ============================================
# LOAD MODEL (GOOGLE DRIVE SUPPORT)
# ============================================
@st.cache_resource
def load_model():
    import gdown

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)

    model_path = "glaucoma.pth"

    # Download if not exists
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1sRI23GizKxjrgZuGtThDDg-CzxlmGD21"
        gdown.download(url, model_path, quiet=False)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 🔥 HANDLE BOTH CASES
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model, device, True

    except Exception as e:
        st.error(f"Loading error: {e}")   
        return None, device, False

# ============================================
# TRANSFORMS
# ============================================
def get_transforms():
    return transforms.Compose([
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ============================================
# PREDICTION
# ============================================
def predict_image(model, image, device):
    transform = get_transforms()
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item(), probs[0].cpu().numpy()

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("👁️ Glaucoma Detection System")
    st.write("AI-based retinal image classification")

    model, device, loaded = load_model()

    if not loaded:
        st.error("❌ Model not loaded. Check Google Drive file ID.")
        return

    uploaded = st.file_uploader("Upload retinal image", type=['jpg', 'png', 'jpeg'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")

        clahe = CLAHETransform()(image)
        with col2:
            st.image(clahe, caption="Enhanced")

        if st.button("Analyze"):
            pred, conf, probs = predict_image(model, image, device)

            st.subheader("Result")

            if pred == 1:
                st.markdown("<div class='result-box glaucoma'>⚠️ Glaucoma Detected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box normal'>✅ Normal</div>", unsafe_allow_html=True)

            st.write(f"Confidence: {conf*100:.2f}%")

            st.write("Normal:", f"{probs[0]*100:.2f}%")
            st.write("Glaucoma:", f"{probs[1]*100:.2f}%")

            st.info("⚠️ This is not a medical diagnosis. Consult a doctor.")

# ============================================
if __name__ == "__main__":
    main()
