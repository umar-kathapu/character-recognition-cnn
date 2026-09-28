import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Character Recognition using CNN",
    page_icon="🔤",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #AAAAAA;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .prediction-box {
        background-color: #1E1E1E;
        border: 1px solid #444444;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .prediction-title {
        font-size: 22px;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .prediction-letter {
        font-size: 85px;
        font-weight: bold;
        color: #FFFFFF;
        line-height: 1.1;
    }

    .confidence {
        font-size: 20px;
        color: #AAAAAA;
        margin-top: 5px;
    }

    .info-card {
        background-color: #181818;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        min-height: 90px;
    }

    .info-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 8px;
    }

    .info-value {
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "model/character_cnn.keras"

CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<div class='main-title'>🔤 Character Recognition using CNN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "Draw a handwritten English character and let the CNN recognize it"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# DRAWING SECTION
# ============================================================

st.subheader("✍️ Draw a Character")

st.write(
    "Draw **one uppercase English letter (A–Z)** inside the box."
)


canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="character_canvas"
)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Convert to NumPy
    img = np.array(image)

    # Detect white pixels
    threshold = 30

    coords = np.argwhere(img > threshold)

    # No drawing detected
    if coords.size == 0:
        return None

    # Find bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Character dimensions
    height, width = img.shape

    # Create square canvas
    size = max(height, width)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center character
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = img

    # Convert to PIL
    processed = Image.fromarray(square)

    # Resize character to 20x20
    processed = processed.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create final 28x28 image
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center character
    final_image.paste(
        processed,
        (4, 4)
    )

    # Match EMNIST orientation
    final_image = final_image.transpose(
        Image.Transpose.TRANSPOSE
    )

    # Normalize pixel values
    array = np.array(
        final_image
    ).astype("float32") / 255.0

    # Add channel dimension
    array = np.expand_dims(
        array,
        axis=-1
    )

    # Add batch dimension
    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Character",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if canvas_result.image_data is None:

        st.warning(
            "⚠️ Please draw a character first."
        )

    else:

        # Convert canvas to image
        canvas_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Preprocess
        processed_image = preprocess_image(
            canvas_image
        )

        if processed_image is None:

            st.warning(
                "⚠️ No character detected. Please draw again."
            )

        else:

            # CNN prediction
            predictions = model.predict(
                processed_image,
                verbose=0
            )[0]

            # Find top 3 predictions
            top_indices = np.argsort(
                predictions
            )[-3:][::-1]

            # Best prediction
            predicted_index = top_indices[0]

            predicted_character = CHARACTERS[
                predicted_index
            ]

            confidence = (
                predictions[predicted_index] * 100
            )


            # ==================================================
            # PREDICTION RESULT
            # ==================================================

            st.markdown(
                "<div class='prediction-box'>"
                "<div class='prediction-title'>"
                "Predicted Character"
                "</div>"
                f"<div class='prediction-letter'>"
                f"{predicted_character}"
                f"</div>"
                f"<div class='confidence'>"
                f"Confidence: {confidence:.2f}%"
                f"</div>"
                "</div>",
                unsafe_allow_html=True
            )


            # ==================================================
            # TOP 3 PREDICTIONS
            # ==================================================

            st.subheader("📊 Top 3 Predictions")

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                character = CHARACTERS[index]

                probability = (
                    predictions[index] * 100
                )

                st.write(
                    f"**{rank}. {character}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(predictions[index])
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🧠 Model Information")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Architecture</div>
            <div class="info-value">CNN</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Classes</div>
            <div class="info-value">26 Letters</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Input</div>
            <div class="info-value">28 × 28 Grayscale</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Test Accuracy</div>
            <div class="info-value">92.49%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CNN ARCHITECTURE
# ============================================================

st.divider()

st.subheader("🏗️ CNN Architecture")


st.code(
"""
Input Image
    ↓
28 × 28 × 1
    ↓
Conv2D (32 filters) + ReLU
    ↓
MaxPooling
    ↓
Conv2D (64 filters) + ReLU
    ↓
MaxPooling
    ↓
Conv2D (128 filters) + ReLU
    ↓
Flatten
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.3)
    ↓
Dense (26) + Softmax
    ↓
Predicted Character (A–Z)
"""
)


# ============================================================
# PROJECT DETAILS
# ============================================================

st.divider()

st.subheader("📌 Project Details")

details_col1, details_col2 = st.columns(2)


with details_col1:

    st.markdown(
        """
        **Dataset:** EMNIST Letters  
        
        **Number of Classes:** 26
        
        **Input Image:** 28 × 28 Grayscale
        
        **Model:** Convolutional Neural Network
        """
    )


with details_col2:

    st.markdown(
        """
        **Optimizer:** Adam  
        
        **Activation:** ReLU + Softmax
        
        **Regularization:** Dropout (0.3)
        
        **Test Accuracy:** 92.49%
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Character Recognition using CNN • "
    "EMNIST Letters Dataset • "
    "28×28 Grayscale Input • "
    "Test Accuracy: 92.49%"
)