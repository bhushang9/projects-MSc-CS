# app.py
"""
Skin Cancer Detection Streamlit App
- Multi-model selector: Custom CNN (skin_cancer_cnn.h5) or MobileNetV2 (skin_cancer_mobilenetv2.h5)
- Prediction with confidence %
- Optional Grad-CAM heatmap explainability
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# ---------- Config ----------
MODEL_FILES = {
    "MobileNetV2 (recommended)": "skin_cancer_mobilenetv2.h5",
    "Custom CNN": "skin_cancer_cnn.h5"
}
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Skin Cancer Detection", layout="wide")
st.title("🩺 Skin Cancer Detection")

st.markdown(
    "Upload a skin lesion image and the model will predict whether it is **Benign** or **Malignant**. "
    "Enable Grad-CAM to visualize which areas the model used for the prediction."
)

# ---------- Sidebar: model selection and info ----------
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Choose model", list(MODEL_FILES.keys()))
use_gradcam = st.sidebar.checkbox("Enable Grad-CAM (slower)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Model files present in the project directory will be loaded automatically.")
st.sidebar.markdown("MobileNetV2 is recommended for better accuracy and explainability.")

# ---------- Helper: load model with cache ----------
@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    if not os.path.exists(path):
        return None
    # allow custom objects / compile disabled to avoid metric warnings
    try:
        model = tf.keras.models.load_model(path)
    except Exception as e:
        # fallback: try load with compile=False
        model = tf.keras.models.load_model(path, compile=False)
    return model

MODEL_PATH = MODEL_FILES[model_choice]
model = load_keras_model(MODEL_PATH)

if model is None:
    st.sidebar.error(f"Model file not found: `{MODEL_PATH}`")
    st.warning("Place the model .h5 file in the app folder or choose a different model.")
    st.stop()

# small info box
st.sidebar.success(f"Loaded: `{os.path.basename(MODEL_PATH)}`")

# ---------- Utilities ----------
def preprocess_pil(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0), np.array(img)

def find_last_conv_layer_name(model):
    """Return the name of the last layer with 4D output (conv feature map)."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                return layer.name
        except Exception:
            # some layers may not expose .output shape
            continue
    return None

def compute_gradcam(model, image_batch, original_rgb, last_conv_layer_name=None):
    """Compute Grad-CAM overlay (RGB uint8) for given model & image batch."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)
        if last_conv_layer_name is None:
            raise ValueError("No 4D convolutional layer found in model (cannot compute Grad-CAM).")

    # Build grad model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        # target the positive class for binary sigmoid model (index 0)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Convert tensors to numpy arrays
    conv_outputs_np = conv_outputs[0].numpy()   # H x W x C
    grads_np = grads[0].numpy()                 # H x W x C

    # Global average pooling to get weights
    weights = np.mean(grads_np, axis=(0, 1))    # shape (C,)

    # Weighted combination of feature maps
    cam = np.zeros(conv_outputs_np.shape[0:2], dtype=np.float32)  # H x W
    for i, w in enumerate(weights):
        cam += w * conv_outputs_np[:, :, i]

    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam_max = cam.max() if cam.max() != 0 else 1e-8
    cam = cam / cam_max

    # Resize to original image size
    heatmap = cv2.resize(cam, (IMG_SIZE[1], IMG_SIZE[0]))  # width, height
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    # Original_rgb is RGB uint8; convert to BGR for overlay with OpenCV
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb

def predict_model(model, img_batch):
    """Return probability (float between 0 and 1)."""
    pred = model.predict(img_batch)
    # handle shapes: (1,) or (1,1)
    if isinstance(pred, np.ndarray):
        pred_val = float(pred.flatten()[0])
    else:
        pred_val = float(pred[0][0])
    return pred_val

# ---------- File uploader ----------
uploaded = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to get prediction and optional Grad-CAM visualization.")
    st.stop()

# ---------- Run prediction & optionally explain ----------
with st.spinner("Processing image..."):
    pil_img = Image.open(uploaded)
    img_batch, original_rgb = preprocess_pil(pil_img)

    prob = predict_model(model, img_batch)
    label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

    gradcam_img = None
    if use_gradcam:
        try:
            gradcam_img = compute_gradcam(model, img_batch, original_rgb)
        except Exception as e:
            gradcam_img = None
            st.warning(f"Grad-CAM failed: {e}")

# ---------- Layout: results ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("📌 Uploaded Image")
    st.image(original_rgb, use_column_width=True)
    st.markdown("**Prediction**")
    st.markdown(f"- **Label:** `{label}`")
    st.markdown(f"- **Confidence:** `{confidence:.2f}%`")

with right:
    st.subheader("🔥 Explanation")
    if use_gradcam and gradcam_img is not None:
        st.image(gradcam_img, use_column_width=True)
        st.caption("Grad-CAM heatmap (overlaid)")
    else:
        st.info("Grad-CAM not shown. Enable 'Enable Grad-CAM' in the sidebar to view explanation.")

# ---------- Footer / Model info ----------
st.markdown("---")
st.markdown("**Model Info**")
st.write(f"- Loaded model file: `{os.path.basename(MODEL_PATH)}`")
st.write(f"- Model type: `{model_choice}`")
st.write("- Note: MobileNetV2 is preferred for production; custom CNN kept for comparison.")
