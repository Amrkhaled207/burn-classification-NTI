import streamlit as st
import numpy as np
import cv2
import gdown
import h5py
import hashlib
from pathlib import Path
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="🔥 Burn Severity Classifier", layout="centered")

# ------------------------------------------------------------------------------
# Config via Secrets
#   - Put these in Streamlit Cloud → Settings → Secrets
#   - Example:
#       MODEL_DRIVE_ID = "1ZE-oRfPgsnXVcsXnv5IQ6Jmsb8MG-C9J"
#       # MODEL_SHA256 = "optional_sha256_here"
# ------------------------------------------------------------------------------
MODEL_PATH = Path("model.h5")
DRIVE_ID = st.secrets["MODEL_DRIVE_ID"]                   # REQUIRED
MODEL_SHA256 = st.secrets.get("MODEL_SHA256", None)       # OPTIONAL

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _is_valid_h5(path: Path) -> bool:
    """Return True if file is a readable HDF5 (.h5) model."""
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_from_drive_by_id(file_id: str, out_path: Path):
    """
    Use gdown with a Drive file ID only.
    Handles 'too large to scan' confirm token automatically.
    """
    # ensure clean target
    out_path.unlink(missing_ok=True)
    # gdown accepts id=... and handles cookies/confirm tokens
    gdown.download(id=file_id, output=str(out_path), quiet=False, use_cookies=True)

def _ensure_model():
    """
    Make sure a valid model exists locally:
    - reuse if valid (and checksum ok when provided)
    - otherwise download via gdown using the Drive file ID
    - verify it's a valid HDF5 and optional checksum
    """
    # Reuse valid file (and verify checksum if provided)
    if MODEL_PATH.exists() and _is_valid_h5(MODEL_PATH):
        if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
            MODEL_PATH.unlink(missing_ok=True)
        else:
            return

    with st.spinner("Downloading model from Google Drive…"):
        _download_from_drive_by_id(DRIVE_ID, MODEL_PATH)

    # Validate format
    if not _is_valid_h5(MODEL_PATH):
        size_mb = MODEL_PATH.stat().st_size / 1e6 if MODEL_PATH.exists() else 0.0
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid .h5 (size={size_mb:.2f} MB). "
            "Check MODEL_DRIVE_ID permissions (Anyone with the link) and that the file is a Keras .h5."
        )

    # Optional integrity check
    if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model checksum mismatch. Re-check MODEL_DRIVE_ID or SHA256.")

# ------------------------------------------------------------------------------
# Model + preprocessing
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_burn_model():
    _ensure_model()
    return load_model(str(MODEL_PATH))

@st.cache_data(show_spinner=False)
def preprocess_image(file_bytes, target_size=(128, 128)):
    image_data = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    batched = np.expand_dims(normalized, axis=0).astype("float32")
    return resized, batched

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
def main():
    st.title("🔥 Burn Severity Classifier")
    st.caption("Upload a skin burn image to classify it as First, Second, or Third degree.")

    uploaded = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("👈 Please upload a JPG or PNG image to continue.")
        return

    # Load model (downloads once, then cached)
    try:
        model = load_burn_model()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # Preprocess
    try:
        img_preview, input_tensor = preprocess_image(uploaded.read())
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        st.stop()

    # Show preview
    st.image(img_preview, caption="🖼️ Uploaded Image", use_column_width=True)
    st.markdown("---")
    st.subheader("📊 Prediction")

    # Predict
    with st.spinner("Analyzing burn severity..."):
        preds = model.predict(input_tensor)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

    labels = ["First Degree", "Second Degree", "Third Degree"]
    label = labels[idx]

    st.success(f"**{label}** detected with **{confidence*100:.2f}%** confidence.")
    st.markdown("### 🔬 Prediction Confidence")
    st.bar_chart({
        "Confidence": {
            "First Degree": float(preds[0]),
            "Second Degree": float(preds[1]),
            "Third Degree": float(preds[2]),
        }
    })

if __name__ == "__main__":
    main()
