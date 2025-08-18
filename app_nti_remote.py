import streamlit as st
import numpy as np
import cv2
import gdown
import h5py
import hashlib
from pathlib import Path
import keras
import tensorflow as tf

# --------------------------------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="🔥 Burn Severity Classifier", layout="centered")

# --------------------------------------------------------------------------------------
# Config via Secrets
#   - Streamlit Cloud → Settings → Secrets:
#       MODEL_DRIVE_ID = "1ZE-oRfPgsnXVcsXnv5IQ6Jmsb8MG-C9J"
#       # MODEL_SHA256 = "optional_sha256_here"
# --------------------------------------------------------------------------------------
MODEL_PATH = Path("model.h5")
DRIVE_ID = st.secrets["MODEL_DRIVE_ID"]
MODEL_SHA256 = st.secrets.get("MODEL_SHA256")

# --------------------------------------------------------------------------------------
# Helpers (download / validate)
# --------------------------------------------------------------------------------------
def _is_valid_h5(path: Path) -> bool:
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
    out_path.unlink(missing_ok=True)
    # gdown handles Drive confirm tokens / large files
    gdown.download(id=file_id, output=str(out_path), quiet=False, use_cookies=True)

def _ensure_model():
    # reuse valid file
    if MODEL_PATH.exists() and _is_valid_h5(MODEL_PATH):
        if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
            MODEL_PATH.unlink(missing_ok=True)
        else:
            return
    with st.spinner("Downloading model from Google Drive…"):
        _download_from_drive_by_id(DRIVE_ID, MODEL_PATH)
    if not _is_valid_h5(MODEL_PATH):
        size_mb = MODEL_PATH.stat().st_size / 1e6 if MODEL_PATH.exists() else 0.0
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid .h5 (size={size_mb:.2f} MB). "
            "Ensure sharing is 'Anyone with the link' and the file is a Keras .h5."
        )
    if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model checksum mismatch. Re-check MODEL_DRIVE_ID or SHA256.")

# --------------------------------------------------------------------------------------
# Compatibility shim for Resizing layer (filters unknown kwargs)
# --------------------------------------------------------------------------------------
class ResizingShim(keras.layers.Layer):
    """
    Wraps keras.layers.Resizing but tolerates extra kwargs that older/newer
    Keras variants may or may not recognize, such as:
      - pad_to_aspect_ratio
      - fill_mode
      - fill_value
    """
    def __init__(self, *args, **kwargs):
        # Keep only the kwargs actually supported by keras.layers.Resizing
        allowed_keys = {
            "height", "width", "interpolation",
            "crop_to_aspect_ratio", "name", "dtype", "data_format",
            # Keras 3 accepts 'dtype' as a policy; if not present it's fine
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed_keys}
        # height & width may be positional in configs; handle both
        if len(args) >= 2:
            height, width = args[0], args[1]
        else:
            height = filtered.pop("height", None)
            width = filtered.pop("width", None)
        if height is None or width is None:
            raise ValueError("ResizingShim requires 'height' and 'width'.")

        self._inner = keras.layers.Resizing(
            height=height,
            width=width,
            **filtered
        )
        super().__init__(name=filtered.get("name", "resizing_shim"), dtype=filtered.get("dtype", None))

    def call(self, inputs):
        return self._inner(inputs)

    def get_config(self):
        cfg = self._inner.get_config()
        # Keep the layer name stable
        cfg["name"] = self.name
        return cfg

# Some saved models may reference the fully qualified path; map common keys:
CUSTOM_OBJECTS = {
    "Resizing": ResizingShim,
    # Occasionally the serialized path could be module-qualified; add aliases:
    "keras.layers.Resizing": ResizingShim,
    "keras.src.layers.preprocessing.image_preprocessing.Resizing": ResizingShim,
}

# --------------------------------------------------------------------------------------
# Model + preprocessing
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_burn_model():
    _ensure_model()
    # Try plain load first; if it fails on Resizing kwargs, retry with shim
    try:
        return keras.models.load_model(str(MODEL_PATH))
    except Exception:
        # second attempt with custom_objects to swallow new kwargs
        return keras.models.load_model(str(MODEL_PATH), custom_objects=CUSTOM_OBJECTS)

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
