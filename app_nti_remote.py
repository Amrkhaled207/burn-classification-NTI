import streamlit as st
import numpy as np
import cv2
import requests
import gdown
import h5py
from pathlib import Path
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="🔥 Burn Severity Classifier", layout="centered")

MODEL_PATH = Path("model.h5")
# Put this in Streamlit Cloud → Settings → Secrets
# MODEL_URL must be a *direct* or share link; gdown can handle both.
MODEL_URL = st.secrets["MODEL_URL"]               # e.g. "https://drive.google.com/uc?export=download&id=FILE_ID"
MODEL_SHA256 = st.secrets.get("MODEL_SHA256")     # optional integrity check

# --------------------------------------------------------------------------------------
# Download utilities (robust for Google Drive)
# --------------------------------------------------------------------------------------
def _is_valid_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_requests(url: str, out: Path) -> None:
    r = requests.get(url, stream=True, allow_redirects=True, timeout=60)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

def _download_with_gdown(url_or_id: str, out: Path) -> None:
    # Supports Google Drive share links, uc? links, or raw file IDs
    gdown.download(url_or_id, str(out), quiet=False, fuzzy=True)

def download_model():
    # If a good model already exists, reuse it (and optionally verify checksum)
    if MODEL_PATH.exists() and _is_valid_h5(MODEL_PATH):
        if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
            MODEL_PATH.unlink(missing_ok=True)   # bad checksum → re-download
        else:
            return

    # Clean any partial file
    if MODEL_PATH.exists():
        MODEL_PATH.unlink(missing_ok=True)

    # Try normal HTTP first (works for many hosts)
    try:
        with st.spinner("Downloading model (method 1)…"):
            _download_with_requests(MODEL_URL, MODEL_PATH)
    except Exception:
        # Fallback to gdown which handles Google Drive confirm tokens & big files
        if MODEL_PATH.exists():
            MODEL_PATH.unlink(missing_ok=True)
        with st.spinner("Downloading model (method 2: gdown)…"):
            _download_with_gdown(MODEL_URL, MODEL_PATH)

    # Validate the downloaded file
    if not _is_valid_h5(MODEL_PATH):
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file is not a valid .h5 model. "
            "Double‑check MODEL_URL (Drive often returns an HTML page if the link/permission is wrong)."
        )

    if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model checksum mismatch. Re-check MODEL_URL or SHA256.")

# --------------------------------------------------------------------------------------
# Model + preprocessing
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_burn_model():
    download_model()
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

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def main():
    st.title("🔥 Burn Severity Classifier")
    st.caption("Upload a skin burn image to classify it as First, Second, or Third degree.")

    uploaded = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("👈 Please upload a JPG or PNG image to continue.")
        return

    try:
        model = load_burn_model()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    try:
        img_preview, input_tensor = preprocess_image(uploaded.read())
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        st.stop()

    st.image(img_preview, caption="🖼️ Uploaded Image", use_column_width=True)
    st.markdown("---")
    st.subheader("📊 Prediction")

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
