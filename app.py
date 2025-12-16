import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import os
from scipy import stats
import cv2
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Fish Classification - Time Series",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================
# FISH CLASSIFIER CLASS
# ============================================


class FishClassifier:
    """Production-ready Fish Classification System"""

    def __init__(self):
        self.models = {}
        self.metadata = None
        self.is_loaded = False
        self.class_names = {
            1: "Spesies 1",
            2: "Spesies 2",
            3: "Spesies 3",
            4: "Spesies 4",
            5: "Spesies 5",
            6: "Spesies 6",
            7: "Spesies 7",
        }

    def load_models(self, model_dir="models"):
        """Load all saved models and metadata"""
        try:
            # Load metadata
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            # Load models
            with open(os.path.join(model_dir, "knn_euclidean.pkl"), "rb") as f:
                self.models["euclidean"] = pickle.load(f)

            with open(os.path.join(model_dir, "knn_dtw.pkl"), "rb") as f:
                self.models["dtw"] = pickle.load(f)

            with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as f:
                self.models["rf"] = pickle.load(f)

            self.is_loaded = True
            return True, "Models loaded successfully!"

        except FileNotFoundError as e:
            return False, f"Model files not found: {str(e)}"
        except Exception as e:
            return False, f"Error loading models: {str(e)}"

    def normalize(self, ts):
        """Z-score normalization"""
        mean = ts.mean()
        std = ts.std()
        if std > 0:
            return (ts - mean) / std
        return ts - mean

    def extract_features(self, ts):
        """Extract statistical features for Random Forest"""
        features = [
            np.mean(ts),
            np.std(ts),
            np.min(ts),
            np.max(ts),
            np.max(ts) - np.min(ts),
            np.median(ts),
            np.percentile(ts, 25),
            np.percentile(ts, 75),
            stats.skew(ts),
            stats.kurtosis(ts),
            np.sum(ts**2),
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, time_series, model="dtw", return_proba=False):
        """Make prediction on time series"""
        if not self.is_loaded:
            raise Exception("Models not loaded!")

        ts = np.array(time_series)

        # Validate input
        expected_length = self.metadata["ts_length"]
        if len(ts) != expected_length:
            raise ValueError(f"Expected length {expected_length}, got {len(ts)}")

        # Normalize
        ts_norm = self.normalize(ts)

        # Predict based on model type
        if model == "euclidean":
            pred = self.models["euclidean"].predict(ts_norm.reshape(1, -1))[0]
            if return_proba:
                proba = self.models["euclidean"].predict_proba(ts_norm.reshape(1, -1))[
                    0
                ]
                return int(pred), float(proba.max()), proba

        elif model == "dtw":
            ts_reshaped = ts_norm.reshape(1, -1, 1)
            pred = self.models["dtw"].predict(ts_reshaped)[0]
            if return_proba:
                return int(pred), None, None

        elif model == "rf":
            features = self.extract_features(ts_norm)
            pred = self.models["rf"].predict(features)[0]
            if return_proba:
                proba = self.models["rf"].predict_proba(features)[0]
                return int(pred), float(proba.max()), proba

        else:
            raise ValueError(f"Unknown model: {model}")

        return int(pred)

    def get_model_accuracy(self, model_name):
        """Get accuracy for specific model"""
        if not self.is_loaded:
            return 0.0

        model_key = (
            f"knn_{model_name}"
            if model_name in ["euclidean", "dtw"]
            else "random_forest"
        )
        return self.metadata["models"][model_key]["accuracy"]


# ============================================
# HELPER FUNCTIONS
# ============================================


def parse_ts_file(file_content):
    """Parse .ts file format"""
    lines = file_content.strip().split("\n")
    data = []
    labels = []
    data_started = False

    for line in lines:
        line = line.strip()

        if line.startswith("#") or line.startswith("@"):
            if line.startswith("@data"):
                data_started = True
            continue

        if data_started and line:
            parts = line.split(":")
            if len(parts) == 2:
                ts_values = [float(x) for x in parts[0].split(",")]
                label = int(parts[1])
                data.append(ts_values)
                labels.append(label)

    return np.array(data), np.array(labels)


def parse_csv_file(df):
    """Parse CSV file (assuming last column is label)"""
    if df.shape[1] > 1:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(int)
    else:
        X = df.values
        y = None
    return X, y


def plot_time_series(ts, title="Time Series", color="blue"):
    """Plot single time series"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, linewidth=2, color=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison(ts_raw, ts_norm):
    """Plot raw vs normalized time series"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(ts_raw, linewidth=2, color="blue")
    axes[0].set_title("Raw Time Series", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts_norm, linewidth=2, color="green")
    axes[1].set_title(
        "Normalized Time Series (Z-Score)", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Time Index")
    axes[1].set_ylabel("Normalized Value")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_probabilities(proba, class_labels):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_labels))
    bars = ax.bar(x, proba, color=plt.cm.viridis(proba / proba.max()), alpha=0.8)

    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax.set_title("Prediction Probabilities per Class", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in class_labels])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def extract_contour_from_image(image, target_points=463):
    """
    Extract contour from fish image and convert to time series
    with background removal

    Parameters:
    -----------
    image : numpy array
        RGB or grayscale image
    target_points : int
        Number of points to extract (default 463)

    Returns:
    --------
    contour_ts : numpy array
        Time series representation of contour (463 points)
    processed_image : numpy array
        Processed image showing detected contour
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply adaptive thresholding for better background removal
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError(
            "No contours found in image. Please use a clear fish image with good contrast."
        )

    # Get the largest contour (assumed to be the fish)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask for the fish only (remove background)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Apply mask to original image to isolate fish
    if len(image.shape) == 3:
        fish_only = cv2.bitwise_and(image, image, mask=mask)
    else:
        fish_only = cv2.bitwise_and(gray, gray, mask=mask)

    # Create visualization image with contour and bounding box
    contour_image = image.copy()
    if len(contour_image.shape) == 2:
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)

    # Draw contour
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    # Draw bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(contour_image, (cx, cy), 5, (255, 0, 0), -1)
    else:
        cx, cy = contour_points.mean(axis=0)
        cx, cy = int(cx), int(cy)

    # Extract contour points
    contour_points = largest_contour.squeeze()

    # Convert to polar coordinates from centroid
    if len(contour_points.shape) == 1:
        contour_points = contour_points.reshape(-1, 2)

    # Calculate distances from centroid (radial distances)
    distances = np.sqrt(
        (contour_points[:, 0] - cx) ** 2 + (contour_points[:, 1] - cy) ** 2
    )

    # Resample to exactly target_points using interpolation
    if len(distances) != target_points:
        # Linear interpolation
        x_old = np.linspace(0, 1, len(distances))
        x_new = np.linspace(0, 1, target_points)
        distances = np.interp(x_new, x_old, distances)

    return distances, contour_image


def plot_image_processing_steps(original, processed, contour_ts):
    """Plot image processing steps"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axes[0].imshow(original, cmap="gray" if len(original.shape) == 2 else None)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Processed with contour
    axes[1].imshow(processed)
    axes[1].set_title("Detected Contour", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Contour as time series
    axes[2].plot(contour_ts, linewidth=2, color="blue")
    axes[2].set_title(
        "Contour Time Series (463 points)", fontsize=12, fontweight="bold"
    )
    axes[2].set_xlabel("Time Index")
    axes[2].set_ylabel("Distance from Centroid")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
    plt.tight_layout()
    return fig


# ============================================
# STREAMLIT APP
# ============================================


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🐟 Fish Classification System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("### Klasifikasi Spesies Ikan dari Time Series Kontur")

    # Initialize session state
    if "classifier" not in st.session_state:
        st.session_state.classifier = FishClassifier()
        st.session_state.models_loaded = False

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/fish.png", width=80)
        st.title("⚙️ Settings")

        # Load Models Section
        st.markdown("### 📦 Load Models")
        model_dir = st.text_input(
            "Model Directory", value="models", help="Folder yang berisi model .pkl"
        )

        if st.button("🔄 Load Models", use_container_width=True):
            with st.spinner("Loading models..."):
                success, message = st.session_state.classifier.load_models(model_dir)
                if success:
                    st.session_state.models_loaded = True
                    st.success(message)
                else:
                    st.session_state.models_loaded = False
                    st.error(message)

        # Model Selection
        st.markdown("### 🤖 Model Selection")
        model_choice = st.selectbox(
            "Pilih Model",
            options=["dtw", "euclidean", "rf"],
            format_func=lambda x: {
                "dtw": "k-NN with DTW (Best)",
                "euclidean": "k-NN with Euclidean",
                "rf": "Random Forest",
            }[x],
            help="Pilih model untuk klasifikasi",
        )

        # Show model info if loaded
        if st.session_state.models_loaded:
            st.markdown("### 📊 Model Info")
            accuracy = st.session_state.classifier.get_model_accuracy(model_choice)
            st.metric("Model Accuracy", f"{accuracy:.2%}")

            metadata = st.session_state.classifier.metadata
            st.info(
                f"""
            **Dataset**: {metadata['dataset']}
            **Classes**: {metadata['n_classes']}
            **TS Length**: {metadata['ts_length']}
            **Created**: {metadata['timestamp']}
            """
            )

        # About
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
        Aplikasi ini mengklasifikasikan spesies ikan berdasarkan kontur time series.
        
        **Supported Formats:**
        - `.ts` (Time Series format)
        - `.csv` (Comma-separated values)
        - `.jpg/.png` (Fish images)
        
        **Upload Requirements:**
        - Time series dengan 463 timesteps
        - Format: nilai numerik dipisahkan koma
        - Images: Clear fish images with good contrast
        """
        )

    # Main content
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first from the sidebar!")
        st.info("👈 Click 'Load Models' button in the sidebar to start.")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📤 Single Prediction",
            "🖼️ Image Upload",
            "📊 Batch Prediction",
            "📈 Visualization",
        ]
    )

    # ============================================
    # TAB 1: SINGLE PREDICTION
    # ============================================
    with tab1:
        st.markdown(
            '<h2 class="sub-header">Single Sample Prediction</h2>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Time Series File (.ts or .csv)",
                type=["ts", "csv"],
                help="Upload file berisi time series dengan panjang 463",
            )

            # Manual input option
            st.markdown("**Atau input manual:**")
            manual_input = st.text_area(
                "Paste time series values (comma-separated)",
                height=100,
                placeholder="1.2, 1.5, -0.3, 2.1, ...",
                help="Masukkan 463 nilai dipisahkan koma",
            )

        with col2:
            st.markdown("### 🎯 Options")
            show_normalized = st.checkbox(
                "Show Normalized", value=True, key="show_norm_tab1"
            )
            show_probability = st.checkbox(
                "Show Probabilities", value=True, key="show_prob_tab1"
            )

        # Process prediction
        if st.button("🔮 Predict", use_container_width=True, type="primary"):
            try:
                # Get time series data
                if uploaded_file is not None:
                    file_extension = uploaded_file.name.split(".")[-1]

                    if file_extension == "ts":
                        content = uploaded_file.read().decode("utf-8")
                        X, y = parse_ts_file(content)
                        if len(X) > 0:
                            time_series = X[0]
                            true_label = y[0] if y is not None and len(y) > 0 else None
                        else:
                            st.error("No data found in file!")
                            return

                    elif file_extension == "csv":
                        df = pd.read_csv(uploaded_file, header=None)
                        X, y = parse_csv_file(df)
                        if len(X) > 0:
                            time_series = X[0]
                            true_label = y[0] if y is not None and len(y) > 0 else None
                        else:
                            st.error("No data found in file!")
                            return

                elif manual_input.strip():
                    values = [float(x.strip()) for x in manual_input.split(",")]
                    time_series = np.array(values)
                    true_label = None

                else:
                    st.warning("Please upload a file or provide manual input!")
                    return

                # Validate length
                if len(time_series) != 463:
                    st.error(
                        f"Expected 463 values, got {len(time_series)}. Please check your input."
                    )
                    return

                # Make prediction
                with st.spinner("Making prediction..."):
                    if show_probability and model_choice in ["euclidean", "rf"]:
                        prediction, confidence, proba = (
                            st.session_state.classifier.predict(
                                time_series, model=model_choice, return_proba=True
                            )
                        )
                    else:
                        prediction = st.session_state.classifier.predict(
                            time_series, model=model_choice
                        )
                        confidence = None
                        proba = None

                # Display results
                st.markdown("---")
                st.markdown(
                    '<h3 class="sub-header">🎯 Prediction Results</h3>',
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        '<div class="metric-container">', unsafe_allow_html=True
                    )
                    st.metric("Predicted Class", f"Class {prediction}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    if confidence is not None:
                        st.markdown(
                            '<div class="metric-container">', unsafe_allow_html=True
                        )
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    if true_label is not None:
                        st.markdown(
                            '<div class="metric-container">', unsafe_allow_html=True
                        )
                        is_correct = prediction == true_label
                        st.metric(
                            "Status",
                            "✅ Correct" if is_correct else "❌ Wrong",
                            delta=f"True: {true_label}",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                # Visualization
                st.markdown("### 📊 Time Series Visualization")

                if show_normalized:
                    # Show both raw and normalized
                    ts_norm = st.session_state.classifier.normalize(time_series)
                    fig = plot_comparison(time_series, ts_norm)
                    st.pyplot(fig)
                else:
                    # Show only raw
                    fig = plot_time_series(
                        time_series, "Time Series Data", color="blue"
                    )
                    st.pyplot(fig)

                # Probability plot
                if show_probability and proba is not None:
                    st.markdown("### 📈 Prediction Probabilities")
                    class_labels = list(range(1, 8))
                    fig = plot_prediction_probabilities(proba, class_labels)
                    st.pyplot(fig)

                # Statistics
                with st.expander("📋 Time Series Statistics"):
                    stats_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Mean",
                                "Std Dev",
                                "Min",
                                "Max",
                                "Range",
                                "Median",
                            ],
                            "Value": [
                                f"{time_series.mean():.4f}",
                                f"{time_series.std():.4f}",
                                f"{time_series.min():.4f}",
                                f"{time_series.max():.4f}",
                                f"{time_series.max() - time_series.min():.4f}",
                                f"{np.median(time_series):.4f}",
                            ],
                        }
                    )
                    st.table(stats_df)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)

    # ============================================
    # TAB 2: IMAGE UPLOAD
    # ============================================
    with tab2:
        st.markdown(
            '<h2 class="sub-header">🖼️ Fish Image Classification</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Upload gambar ikan untuk ekstraksi kontur otomatis dan prediksi spesies"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Image upload
            uploaded_image = st.file_uploader(
                "Upload Fish Image (.jpg, .png, .jpeg)",
                type=["jpg", "png", "jpeg", "bmp"],
                help="Upload gambar ikan dengan background kontras untuk hasil terbaik",
            )

            if uploaded_image is not None:
                # Display original image
                image = Image.open(uploaded_image)
                image_np = np.array(image)

                st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("### 🎯 Options")
            show_steps = st.checkbox(
                "Show Processing Steps", value=True, key="show_steps_tab2"
            )
            show_prob_img = st.checkbox(
                "Show Probabilities", value=True, key="show_prob_tab2"
            )

            st.markdown("### 🏷️ True Label (Optional)")
            true_label_input = st.selectbox(
                "Select true class if known:",
                ["None", "1", "2", "3", "4", "5", "6", "7"],
                key="true_label_tab2",
                help="Pilih class sebenarnya untuk membandingkan dengan prediksi",
            )
            true_label = None if true_label_input == "None" else int(true_label_input)

            st.markdown("### 📝 Tips")
            st.info(
                """
            **Untuk hasil terbaik:**
            - Gunakan gambar dengan background kontras
            - Ikan harus terlihat jelas
            - Hindari gambar terlalu blur
            - Format: JPG, PNG, JPEG
            """
            )

        # Process image
        if uploaded_image is not None and st.button(
            "🔮 Predict from Image", use_container_width=True, type="primary"
        ):
            try:
                with st.spinner("Processing image and extracting contour..."):
                    # Extract contour from image
                    contour_ts, processed_img = extract_contour_from_image(image_np)

                    st.success(
                        f"✅ Contour extracted successfully! ({len(contour_ts)} points)"
                    )

                # Show processing steps
                if show_steps:
                    st.markdown("### 🔍 Image Processing Steps")
                    fig = plot_image_processing_steps(
                        image_np, processed_img, contour_ts
                    )
                    st.pyplot(fig)

                # Make prediction
                with st.spinner("Making prediction..."):
                    if show_prob_img and model_choice in ["euclidean", "rf"]:
                        prediction, confidence, proba = (
                            st.session_state.classifier.predict(
                                contour_ts, model=model_choice, return_proba=True
                            )
                        )
                    else:
                        prediction = st.session_state.classifier.predict(
                            contour_ts, model=model_choice
                        )
                        confidence = None
                        proba = None

                # Display results
                st.markdown("---")
                st.markdown(
                    '<h3 class="sub-header">🎯 Prediction Results</h3>',
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        '<div class="metric-container">', unsafe_allow_html=True
                    )
                    st.metric("Predicted Class", f"Class {prediction}")
                    st.markdown(
                        f"**Species:** {st.session_state.classifier.class_names[prediction]}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    if confidence is not None:
                        st.markdown(
                            '<div class="metric-container">', unsafe_allow_html=True
                        )
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    if true_label is not None:
                        st.markdown(
                            '<div class="metric-container">', unsafe_allow_html=True
                        )
                        is_correct = prediction == true_label
                        st.metric(
                            "Status",
                            "✅ Correct" if is_correct else "❌ Wrong",
                            delta=f"True: Class {true_label}",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="metric-container">', unsafe_allow_html=True
                        )
                        st.metric("Contour Points", f"{len(contour_ts)}")
                        st.markdown("</div>", unsafe_allow_html=True)

                # Probability plot
                if show_prob_img and proba is not None:
                    st.markdown("### 📈 Prediction Probabilities")
                    class_labels = list(range(1, 8))
                    fig = plot_prediction_probabilities(proba, class_labels)
                    st.pyplot(fig)

                # Contour visualization
                st.markdown("### 📊 Extracted Contour Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Raw contour
                    fig = plot_time_series(
                        contour_ts, "Raw Contour Time Series", color="blue"
                    )
                    st.pyplot(fig)

                with col2:
                    # Normalized contour
                    ts_norm = st.session_state.classifier.normalize(contour_ts)
                    fig = plot_time_series(ts_norm, "Normalized Contour", color="green")
                    st.pyplot(fig)

                # Contour statistics
                with st.expander("📋 Contour Statistics"):
                    stats_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Mean Distance",
                                "Std Dev",
                                "Min Distance",
                                "Max Distance",
                                "Range",
                                "Median",
                            ],
                            "Value": [
                                f"{contour_ts.mean():.4f}",
                                f"{contour_ts.std():.4f}",
                                f"{contour_ts.min():.4f}",
                                f"{contour_ts.max():.4f}",
                                f"{contour_ts.max() - contour_ts.min():.4f}",
                                f"{np.median(contour_ts):.4f}",
                            ],
                        }
                    )
                    st.table(stats_df)

                # Download contour data
                csv = pd.DataFrame({"Contour_Distance": contour_ts}).to_csv(index=False)
                st.download_button(
                    label="📥 Download Contour Data (CSV)",
                    data=csv,
                    file_name=f"fish_contour_class_{prediction}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
                st.warning(
                    """
                **Troubleshooting:**
                - Pastikan gambar memiliki ikan yang jelas
                - Background harus kontras dengan ikan
                - Coba crop gambar untuk fokus pada ikan saja
                - Gunakan gambar dengan resolusi yang baik
                """
                )

    # ============================================
    # TAB 3: BATCH PREDICTION
    # ============================================
    with tab3:
        st.markdown(
            '<h2 class="sub-header">Batch Prediction</h2>', unsafe_allow_html=True
        )
        st.markdown("Upload file berisi multiple time series untuk batch prediction")

        batch_file = st.file_uploader(
            "Upload Batch File (.ts or .csv)",
            type=["ts", "csv"],
            key="batch_upload",
            help="File berisi multiple time series",
        )

        if batch_file is not None:
            try:
                file_extension = batch_file.name.split(".")[-1]

                # Parse file
                if file_extension == "ts":
                    content = batch_file.read().decode("utf-8")
                    X, y = parse_ts_file(content)
                elif file_extension == "csv":
                    df = pd.read_csv(batch_file, header=None)
                    X, y = parse_csv_file(df)

                st.success(f"✅ Loaded {len(X)} samples from file")

                # Show preview
                with st.expander("🔍 Data Preview"):
                    preview_df = pd.DataFrame(X[:5, :10])
                    preview_df.columns = [f"T{i}" for i in range(10)]
                    st.dataframe(preview_df)

                # Batch predict button
                if st.button(
                    "🚀 Run Batch Prediction", use_container_width=True, type="primary"
                ):
                    with st.spinner(f"Processing {len(X)} samples..."):
                        predictions = []

                        # Progress bar
                        progress_bar = st.progress(0)

                        for i, ts in enumerate(X):
                            try:
                                pred = st.session_state.classifier.predict(
                                    ts, model=model_choice
                                )
                                predictions.append(pred)
                            except:
                                predictions.append(-1)  # Error marker

                            progress_bar.progress((i + 1) / len(X))

                        progress_bar.empty()
                        predictions = np.array(predictions)

                    # Show results
                    st.markdown("### 📊 Batch Results")

                    results_df = pd.DataFrame(
                        {"Sample_ID": range(len(X)), "Predicted_Class": predictions}
                    )

                    if y is not None:
                        results_df["True_Class"] = y
                        results_df["Correct"] = (predictions == y).astype(int)
                        results_df["Status"] = results_df["Correct"].apply(
                            lambda x: "✅" if x else "❌"
                        )

                        accuracy = (predictions == y).mean()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(X))
                        with col2:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col3:
                            st.metric(
                                "Correct Predictions",
                                f"{(predictions == y).sum()}/{len(X)}",
                            )

                    # Display table
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )

                    # Confusion matrix if true labels available
                    if y is not None:
                        st.markdown("### 📈 Prediction Distribution")

                        col1, col2 = st.columns(2)

                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            pred_counts = (
                                pd.Series(predictions).value_counts().sort_index()
                            )
                            ax.bar(
                                pred_counts.index,
                                pred_counts.values,
                                color="steelblue",
                                alpha=0.7,
                            )
                            ax.set_xlabel(
                                "Predicted Class", fontsize=12, fontweight="bold"
                            )
                            ax.set_ylabel("Count", fontsize=12, fontweight="bold")
                            ax.set_title(
                                "Prediction Distribution",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax.grid(axis="y", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)

                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            true_counts = pd.Series(y).value_counts().sort_index()
                            ax.bar(
                                true_counts.index,
                                true_counts.values,
                                color="seagreen",
                                alpha=0.7,
                            )
                            ax.set_xlabel("True Class", fontsize=12, fontweight="bold")
                            ax.set_ylabel("Count", fontsize=12, fontweight="bold")
                            ax.set_title(
                                "True Class Distribution",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax.grid(axis="y", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")
                st.exception(e)

    # ============================================
    # TAB 4: VISUALIZATION
    # ============================================
    with tab4:
        st.markdown(
            '<h2 class="sub-header">Time Series Visualization</h2>',
            unsafe_allow_html=True,
        )

        viz_file = st.file_uploader(
            "Upload File for Visualization", type=["ts", "csv"], key="viz_upload"
        )

        if viz_file is not None:
            try:
                file_extension = viz_file.name.split(".")[-1]

                if file_extension == "ts":
                    content = viz_file.read().decode("utf-8")
                    X, y = parse_ts_file(content)
                elif file_extension == "csv":
                    df = pd.read_csv(viz_file, header=None)
                    X, y = parse_csv_file(df)

                st.success(f"✅ Loaded {len(X)} samples")

                # Sample selection
                sample_idx = st.slider("Select Sample Index", 0, len(X) - 1, 0)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Raw Time Series")
                    fig = plot_time_series(
                        X[sample_idx], f"Sample {sample_idx}", color="blue"
                    )
                    st.pyplot(fig)

                with col2:
                    st.markdown("### Normalized Time Series")
                    ts_norm = st.session_state.classifier.normalize(X[sample_idx])
                    fig = plot_time_series(
                        ts_norm, f"Sample {sample_idx} (Normalized)", color="green"
                    )
                    st.pyplot(fig)

                # Class comparison
                if y is not None:
                    st.markdown("### 📊 Class Comparison")

                    selected_class = st.selectbox("Select Class", sorted(np.unique(y)))
                    class_indices = np.where(y == selected_class)[0]

                    fig, ax = plt.subplots(figsize=(14, 6))

                    # Plot all samples from selected class
                    for idx in class_indices[:20]:  # Limit to 20 samples
                        ax.plot(X[idx], alpha=0.3, color="steelblue")

                    # Plot mean
                    mean_ts = X[class_indices].mean(axis=0)
                    ax.plot(mean_ts, linewidth=3, color="darkblue", label="Mean")

                    ax.set_title(
                        f"Class {selected_class} - Time Series Overlay",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax.set_xlabel("Time Index", fontsize=12)
                    ax.set_ylabel("Value", fontsize=12)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {str(e)}")


# Run app
if __name__ == "__main__":
    main()
