import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
HOME_IMAGE_PATH = os.path.join(BASE_DIR, "home_page.jpeg")
IMAGE_SIZE = (128, 128)

# ============================================================
# DISEASE CLASS NAMES
# IMPORTANT: Must match training class order
# ============================================================
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# ============================================================
# LOAD MODEL ONLY ONCE
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# GRAD-CAM EXPLAINABILITY (conv2d_2)
# ============================================================
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d_2"):
    """
    Computes Grad-CAM activations from the last convolution block (conv2d_2)
    and returns the heatmap array, full class predictions, and argmax index.
    """
    layers_before = []
    layers_after = []
    found = False
    for l in model.layers:
        if not found:
            layers_before.append(l)
            if l.name == last_conv_layer_name:
                found = True
        else:
            layers_after.append(l)

    with tf.GradientTape() as tape:
        x = img_array
        for l in layers_before:
            x = l(x)
        conv_output = x
        tape.watch(conv_output)
        for l in layers_after:
            x = l(x)
        predictions = x
        predicted_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, predicted_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    return heatmap_np, predictions.numpy(), predicted_index


def overlay_gradcam(pil_image, heatmap_np, alpha=0.45):
    """
    Superimposes a JET colormap representation of the Grad-CAM heatmap
    onto the original leaf image.
    """
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_np)[:, :, :3]
    raw_heatmap_img = Image.fromarray((colored_heatmap * 255).astype(np.uint8)).resize(
        pil_image.size, Image.Resampling.BILINEAR
    )

    original_np = np.array(pil_image.convert("RGB"))
    heatmap_resized = np.array(raw_heatmap_img)
    superimposed_np = ((1 - alpha) * original_np + alpha * heatmap_resized).clip(0, 255).astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_np)

    return superimposed_img, raw_heatmap_img


# ============================================================
# MULTI-VIEW PREPROCESSING (EXIF, TRANSPARENCY, TTA)
# ============================================================
def prepare_image_views(pil_image: Image.Image):
    """
    Cleans and standardizes the input image:
    1. Transposes smartphone EXIF orientation tags.
    2. Blends RGBA transparency smoothly over white instead of defaulting to black.
    3. Generates 3 complementary views for test-time augmentation (TTA):
       - Direct Bilinear resize
       - Center-square crop (preserves natural aspect ratio without stretching/squashing)
       - Inner border-trim crop (eliminates snipping tool/screenshot letterbox border artifacts)
    """
    img = ImageOps.exif_transpose(pil_image)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.convert("RGBA")
        bg = Image.new("RGB", alpha.size, (255, 255, 255))
        bg.paste(alpha, mask=alpha.split()[3])
        rgb_image = bg
    else:
        rgb_image = img.convert("RGB")

    w, h = rgb_image.size
    min_dim = min(w, h)
    cx = (w - min_dim) // 2
    cy = (h - min_dim) // 2

    # View 1: Direct Bilinear
    v1 = rgb_image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    # View 2: Center-square Bilinear
    v2 = rgb_image.crop((cx, cy, cx + min_dim, cy + min_dim)).resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    # View 3: Inner border-trim crop (trims screenshot/snipping tool solid borders)
    pad_w = int(w * 0.04)
    pad_h = int(h * 0.04)
    inner = rgb_image.crop((pad_w, pad_h, max(pad_w + 10, w - pad_w), max(pad_h + 10, h - pad_h)))
    min_inner = min(inner.size)
    cx_i = (inner.size[0] - min_inner) // 2
    cy_i = (inner.size[1] - min_inner) // 2
    v3 = inner.crop((cx_i, cy_i, cx_i + min_inner, cy_i + min_inner)).resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    return rgb_image, [v1, v2, v3]


# ============================================================
# REUSABLE PREDICTION & EXPLAINABILITY UI PIPELINE
# ============================================================
def render_prediction_pipeline(uploaded_img_file, key_prefix="home"):
    try:
        # Load image & model
        raw_image = Image.open(uploaded_img_file)
        model = load_model()

        if model is None:
            st.error(f"Model not found at: {MODEL_PATH}")
            return

        # Prepare standardized image views for robust multi-view TTA inference
        image, views = prepare_image_views(raw_image)
        views_array = np.stack([np.array(v, dtype=np.float32) for v in views], axis=0)

        # Uploaded image preview
        st.subheader("📷 Uploaded Leaf Image Preview")
        st.image(image, caption="Selected Plant Leaf", width=350)

        if st.button("Predict Disease & Generate Grad-CAM", type="primary", key=f"{key_prefix}_predict_btn"):

            with st.spinner("🔍 Analyzing leaf and generating Grad-CAM explainability map..."):
                # Multi-view TTA predictions (averaging views to eliminate border & crop biases)
                all_preds = model.predict(views_array, verbose=0)
                predictions = np.mean(all_preds, axis=0, keepdims=True)
                predicted_index = int(np.argmax(predictions[0]))

                # Grad-CAM on best matching view
                best_view_idx = int(np.argmax(all_preds[:, predicted_index]))
                primary_batch = np.expand_dims(views_array[best_view_idx], axis=0)
                heatmap, _, _ = generate_gradcam(
                    model=model,
                    img_array=primary_batch,
                    last_conv_layer_name="conv2d_2"
                )

            predicted_class = class_names[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100
            display_name = predicted_class.replace("___", " - ").replace("_", " ")

            # Three-column visual layout
            st.divider()
            st.subheader("🧠 Neural Attention & Grad-CAM Explainability")
            col_img, col_raw, col_cam = st.columns(3)

            with col_img:
                st.markdown("##### 1. Original Input Leaf")
                st.image(image, caption="Uploaded Leaf", use_container_width=True)

            with col_raw:
                st.markdown("##### 2. Raw Attention Map (JET)")
                superimposed_img, raw_heatmap_img = overlay_gradcam(
                    image,
                    heatmap,
                    alpha=0.45
                )
                st.image(
                    raw_heatmap_img,
                    caption="Normalized Grad-CAM Activation (conv2d_2)",
                    use_container_width=True
                )

            with col_cam:
                st.markdown("##### 3. Grad-CAM Overlay")
                st.image(
                    superimposed_img,
                    caption="Blended Overlay (Warm = Key Lesions)",
                    use_container_width=True
                )

            st.divider()

            # Confidence status
            if confidence < 40:
                confidence_status = "low"
            elif confidence < 70:
                confidence_status = "moderate"
            else:
                confidence_status = "high"

            # Low confidence
            if confidence_status == "low":
                st.subheader("⚠️ Inconclusive Analysis")
                st.warning(
                    f"**Low Neural Certainty ({confidence:.1f}%):** "
                    "The model cannot reliably classify this image. "
                    "Upload a clear, single leaf photographed against a simple background before acting on this result."
                )
                st.metric("Model Confidence", f"{confidence:.2f}%")
                st.progress(min(int(confidence), 100))
                st.info("🔄 **Recommended Action:** Upload a clearer image of one leaf with good lighting and a simple background.")
                st.info("🌱 **Treatment Recommendation:** Withheld because the prediction is inconclusive.")

            # Moderate confidence
            elif confidence_status == "moderate":
                st.subheader("🔍 Prediction Result")
                st.info(
                    f"**Moderate Prediction Confidence ({confidence:.1f}%):** "
                    "The model has some uncertainty about this image. For better results, upload a clear, single leaf."
                )
                st.write(f"**Possible Disease:** {display_name}")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.progress(min(int(confidence), 100))
                st.warning("⚠️ This prediction should be treated as a possible classification, not a confirmed diagnosis.")

            # High confidence
            else:
                st.subheader("🔍 Prediction Result")
                st.success(f"**Predicted Disease:** {display_name}")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.progress(min(int(confidence), 100))
                st.success(f"✅ **High Prediction Confidence ({confidence:.1f}%):** The model is sufficiently confident about this result.")

            # Top-3 predictions
            st.divider()
            st.subheader("📊 Top-3 Model Predictions")
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]

            for rank, idx in enumerate(top_3_indices, start=1):
                name = class_names[idx].replace("___", " - ").replace("_", " ")
                prob = float(predictions[0][idx]) * 100
                st.write(f"**{rank}. {name}** — {prob:.2f}%")
                st.progress(min(max(prob / 100.0, 0.0), 1.0))

            # Grad-CAM explanation expander
            with st.expander("🔬 Understand the Grad-CAM Heatmap"):
                if confidence_status == "low":
                    st.warning(
                        "Because the prediction confidence is low, this Grad-CAM visualization should only be considered "
                        "an explanation of the model's tentative prediction."
                    )

                st.markdown("""
                **How Grad-CAM Works:**
                - **Grad-CAM (Gradient-weighted Class Activation Mapping)** calculates the gradients of the predicted class score
                  with respect to feature maps from the final convolutional layer (`conv2d_2`).
                - **Red & Yellow Regions:** Highlight areas on the leaf that contributed most strongly to the model's prediction.
                - **Blue & Green Regions:** Indicate background or healthy tissue that had relatively less influence.
                """)

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.image(raw_heatmap_img, caption="Raw Activation Heatmap", width="stretch")
                with col_h2:
                    st.image(superimposed_img, caption="Superimposed Heatmap Overlay", width="stretch")

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)


# ============================================================
# SIDEBAR: MODEL DETAILS & NAVIGATION
# ============================================================
st.sidebar.markdown("## Model Details")
st.sidebar.markdown("""
**Model:** Custom CNN

**Model File:** trained_plant_disease_model.keras

**Classes:** 38 Plant Categories

**Input Resolution:** 128 × 128 × 3

**Architecture:** 3 convolution blocks + dense layers

**Explainability:** Grad-CAM (conv2d_2)
""")

st.sidebar.markdown("---")

st.sidebar.markdown("## Navigation")

app_mode = st.sidebar.radio(
    "Navigation",
    ["Home", "Disease Recognition"],
    label_visibility="collapsed"
)


# ============================================================
# HOME PAGE
# ============================================================
if app_mode == "Home":

    st.title("🌿 Plant Disease Recognition System")

    st.write(
        "Upload a plant leaf image to identify possible diseases "
        "using a trained Custom CNN with Grad-CAM explainability."
    )

    # --------------------------------------------------------
    # UPLOAD LEAF IMAGE DIRECTLY ON HOME PAGE
    # --------------------------------------------------------
    st.subheader("📤 Upload Leaf Image")
    home_uploaded_file = st.file_uploader(
        "Choose a plant leaf image to analyze right away",
        type=["jpg", "jpeg", "png"],
        key="home_leaf_uploader"
    )

    if home_uploaded_file is not None:
        # Render full preview, prediction, and Grad-CAM for the uploaded image
        render_prediction_pipeline(home_uploaded_file, key_prefix="home")

    else:
        # Default banner and overview when no image is uploaded yet
        if os.path.isfile(HOME_IMAGE_PATH):
            st.image(HOME_IMAGE_PATH, width="stretch")

        st.markdown("""
        ### Welcome!

        This system uses deep learning to classify plant leaf images across 38 crop disease categories.

        ### How It Works
        1. **Upload an image** using the uploader above or select **Disease Recognition** in the sidebar.
        2. Upload any JPG, JPEG, or PNG plant leaf image.
        3. Click **Predict Disease & Generate Grad-CAM**.
        4. View predicted class, confidence scores, and visual **Grad-CAM (conv2d_2)** attention heatmap.

        ### Features
        - **Custom CNN Architecture**: 3 convolution blocks with MaxPooling, GlobalAveragePooling, and Dense layers.
        - **Explainable AI (XAI)**: Grad-CAM using feature maps from `conv2d_2`.
        - **38 Configured Plant Disease Classes**: High-coverage crop disease classification.
        - **Top-3 Predictions**: Multi-class confidence distribution.

        > **Note:** Model predictions are AI-assisted estimates and should be verified with agricultural experts.
        """)


# ============================================================
# DISEASE RECOGNITION PAGE
# ============================================================
elif app_mode == "Disease Recognition":

    st.title("🔍 Disease Recognition & Explainability")

    st.write("Upload a clear image of a plant leaf to predict the disease and generate Grad-CAM explainability.")

    rec_uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="rec_leaf_uploader"
    )

    if rec_uploaded_file is not None:
        render_prediction_pipeline(rec_uploaded_file, key_prefix="rec")
    else:
        st.info("Please upload a plant leaf image to continue.")


# ============================================================
# CUSTOM CNN ARCHITECTURE SECTION
# ============================================================
st.divider()
st.subheader("🧠 Custom CNN Architecture Details")
st.markdown("The custom convolutional neural network classifies plant leaf images across 38 distinct categories.")

arch_col1, arch_col2 = st.columns([1, 1])

with arch_col1:
    st.code("""
Input Image: (128 × 128 × 3)
      │
      ▼
Rescaling Layer (1.0 / 255.0)
      │
      ▼
Block 1:
Conv2D(32, 3×3, ReLU)
+ MaxPool(2×2)
      │
      ▼
Block 2:
Conv2D(64, 3×3, ReLU)
+ MaxPool(2×2)
      │
      ▼
Block 3:
Conv2D(128, 3×3, ReLU) [conv2d_2 -> Grad-CAM]
+ MaxPool(2×2)
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
Dense Layer (128 Units, ReLU)
      │
      ▼
Dropout (0.2)
      │
      ▼
Output Dense (38 Classes, Softmax)
    """, language="text")

with arch_col2:
    st.markdown("""
    ### Architecture Highlights
    - **Input Dimensions:** `128 × 128 × 3` RGB images.
    - **Built-in Rescaling:** Automatically scales pixel values `[0, 255]` to `[0, 1]`.
    - **3 Feature Extraction Blocks:** 32, 64, and 128 filters to learn hierarchical patterns (edges, textures, lesions).
    - **Explainability Target:** Layer `conv2d_2` (the 3rd convolution block) preserves spatial resolution for Grad-CAM.
    - **Global Average Pooling:** Condenses feature maps into a 128-dimensional vector, avoiding parameter bloat.
    - **Dense & Dropout Head:** 128 units with dropout regularization to prevent overfitting.
    - **Output Layer:** Softmax distribution over all 38 configured plant categories.
    """)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🌿 Plant Disease Detection & Explainable AI | Powered by TensorFlow & Streamlit")
