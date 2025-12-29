"""
Fish Classification - Streamlit App
FASE 3: Deployment Application

Upload gambar ikan → Prediksi spesies dengan visualisasi
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import json
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

# ========== CONFIG ==========
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")


# ========== LOAD MODEL ==========
@st.cache_resource
def load_model_and_metadata():
    """Load model, encoder, dan metadata"""
    try:
        model = joblib.load("models/knn_fish_classifier.pkl")
        encoder = joblib.load("models/label_encoder.pkl")

        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)

        return model, encoder, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# ========== IMAGE PROCESSING FUNCTIONS ==========
def preprocess_and_segment(image):
    """Preprocessing dan Binary Segmentation"""
    # Convert ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Otsu Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

    return binary_clean


def normalize_contour_starting_point(contour_points):
    """Normalize starting point ke leftmost (mulut ikan)"""
    leftmost_idx = np.argmin(contour_points[:, 0])
    contour_normalized = np.roll(contour_points, -leftmost_idx, axis=0)
    return contour_normalized, leftmost_idx


def extract_and_resample_contour(binary_mask, n_points, normalize_start=True):
    """Extract kontur dan resample ke N titik"""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("❌ Tidak ada kontur ditemukan di gambar")

    # Ambil kontur terbesar
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()

    if contour_points.ndim == 1:
        raise ValueError("❌ Kontur terlalu kecil")

    # IMPROVEMENT: Normalize starting point
    if normalize_start:
        contour_points, _ = normalize_contour_starting_point(contour_points)

    # Resample menggunakan interpolasi
    n_original = len(contour_points)
    t_original = np.linspace(0, 1, n_original)
    t_new = np.linspace(0, 1, n_points)

    fx = interpolate.interp1d(t_original, contour_points[:, 0], kind="linear")
    fy = interpolate.interp1d(t_original, contour_points[:, 1], kind="linear")

    x_resampled = fx(t_new)
    y_resampled = fy(t_new)

    contour_resampled = np.column_stack([x_resampled, y_resampled])

    return contour_resampled


def contour_to_timeseries(contour_points, method="centroid_distance"):
    """Konversi kontur ke time series signal"""
    if method == "centroid_distance":
        # Jarak Euclidean dari centroid
        centroid_x = np.mean(contour_points[:, 0])
        centroid_y = np.mean(contour_points[:, 1])

        distances = np.sqrt(
            (contour_points[:, 0] - centroid_x) ** 2
            + (contour_points[:, 1] - centroid_y) ** 2
        )
        time_series = distances

    elif method == "turn_angle":
        # Turn Angle
        angles = []
        n = len(contour_points)

        for i in range(n):
            p_prev = contour_points[i - 1]
            p_curr = contour_points[i]
            p_next = contour_points[(i + 1) % n]

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(angle)

        time_series = np.array(angles)

    # Normalisasi dan scaling
    scaler = MinMaxScaler()
    time_series_normalized = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    time_series_scaled = time_series_normalized * 3 - 1  # [0,1] → [-1, 2]

    return time_series_scaled


def image_to_features(image, n_features, method="centroid_distance"):
    """Pipeline: Image → Features untuk prediksi"""
    # Step 1: Preprocessing
    binary_mask = preprocess_and_segment(image)

    # Step 2: Extract & Resample Contour
    contour_resampled = extract_and_resample_contour(binary_mask, n_features)

    # Step 3: Convert to Time Series
    time_series = contour_to_timeseries(contour_resampled, method=method)

    return time_series, binary_mask, contour_resampled


# ========== STREAMLIT UI ==========
def main():
    # Header
    st.title("🐟 Fish Species Classifier")
    st.markdown(
        """
    Upload gambar ikan untuk prediksi spesies menggunakan **Contour-based Time Series** + **K-Nearest Neighbors**.
    
    **Workflow:**
    1. Preprocessing & Segmentation (Binary Thresholding)
    2. Contour Extraction & Resampling
    3. Time Series Feature Generation
    4. KNN Classification
    """
    )

    # Load model
    model, encoder, metadata = load_model_and_metadata()

    if model is None:
        st.error(
            "⚠️ Model tidak dapat dimuat. Pastikan file model ada di folder 'models/'"
        )
        return

    # Sidebar - Info
    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.write(f"**Classes:** {len(metadata['classes'])}")
        st.write(f"**Accuracy:** {metadata['accuracy']*100:.2f}%")
        st.write(f"**Feature Dim:** {metadata['n_features']}")

        st.markdown("---")
        st.header("⚙️ Settings")
        feature_method = st.selectbox(
            "Feature Method:",
            ["centroid_distance", "turn_angle"],
            help="Metode konversi kontur ke time series",
        )

        normalize_start = st.checkbox(
            "Normalize Starting Point",
            value=True,
            help="Pastikan contour dimulai dari titik paling kiri (mulut ikan)",
        )

        show_validation = st.checkbox(
            "Show Validation Details",
            value=True,
            help="Tampilkan validasi visual untuk deteksi prediksi yang salah",
        )

        st.markdown("---")
        st.caption("Classes:")
        for i, cls in enumerate(metadata["classes"]):
            st.caption(f"{i+1}. {cls}")

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload gambar ikan (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Convert RGBA to RGB jika perlu
        if image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Process
        with st.spinner("🔄 Processing..."):
            try:
                # Extract features
                features, binary_mask, contour = image_to_features(
                    image_array, metadata["n_features"], method=feature_method
                )

                # Predict
                features_reshaped = features.reshape(1, -1)
                prediction = model.predict(features_reshaped)
                predicted_class = encoder.inverse_transform(prediction)[0]

                # Get probabilities (KNN distance-based)
                distances, indices = model.kneighbors(features_reshaped)

                # Display Results
                st.success("✅ Processing completed!")

                # Prediction result
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("🎯 Predicted Species", predicted_class)

                with col2:
                    st.metric(
                        "📏 Distance to Nearest Neighbor", f"{distances[0][0]:.4f}"
                    )

                # Visualization
                st.markdown("---")
                st.subheader("📊 Processing Pipeline Visualization")

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # 1. Original Image
                axes[0, 0].imshow(image_array)
                axes[0, 0].set_title(
                    "1. Original Image", fontsize=12, fontweight="bold"
                )
                axes[0, 0].axis("off")

                # 2. Binary Segmentation
                axes[0, 1].imshow(binary_mask, cmap="gray")
                axes[0, 1].set_title(
                    "2. Binary Segmentation", fontsize=12, fontweight="bold"
                )
                axes[0, 1].axis("off")

                # 3. Contour
                axes[1, 0].plot(contour[:, 0], contour[:, 1], "b.-", markersize=1)
                axes[1, 0].set_title(
                    f'3. Contour ({metadata["n_features"]} points)',
                    fontsize=12,
                    fontweight="bold",
                )
                axes[1, 0].invert_yaxis()
                axes[1, 0].set_aspect("equal")
                axes[1, 0].grid(True, alpha=0.3)

                # 4. Time Series Signal
                axes[1, 1].plot(features, linewidth=1.5)
                axes[1, 1].set_title(
                    f"4. Time Series ({feature_method})", fontsize=12, fontweight="bold"
                )
                axes[1, 1].set_xlabel("Point Index")
                axes[1, 1].set_ylabel("Value")
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # VALIDATION SECTION
                if show_validation:
                    st.markdown("---")
                    st.subheader("🔍 Validation: Apakah Prediksi Valid?")

                    # Load training data for validation
                    # Note: Ini akan di-cache karena menggunakan joblib load
                    try:
                        # We need to load X_train for validation
                        # Ideally should be saved separately, but for now we'll skip detailed validation in Streamlit
                        st.info("💡 **Validation Tips:**")

                        col_v1, col_v2 = st.columns(2)

                        with col_v1:
                            st.metric("Nearest Distance", f"{distances[0][0]:.4f}")
                            if distances[0][0] < 5:
                                st.success("✅ Distance rendah - Good match!")
                            elif distances[0][0] < 15:
                                st.warning("⚠️ Distance moderate - Cukup yakin")
                            else:
                                st.error("❌ Distance tinggi - Mungkin 'Unknown'")

                        with col_v2:
                            # Calculate simple confidence based on distance
                            confidence = 1 / (1 + distances[0][0])
                            st.metric("Confidence Score", f"{confidence*100:.1f}%")
                            if confidence > 0.8:
                                st.success("✅ Sangat yakin")
                            elif confidence > 0.5:
                                st.warning("⚠️ Cukup yakin")
                            else:
                                st.error("❌ Tidak yakin - Check preprocessing!")

                        st.markdown(
                            """
                        **Cara Validasi Manual:**
                        1. ✅ **Visual Check**: Apakah time series (plot 4) berbentuk smooth/konsisten?
                        2. ✅ **Distance Check**: Distance < 10 biasanya valid
                        3. ✅ **Feature Range**: Harus mirip dengan training data (sekitar -1 sampai 2)
                        4. ⚠️ **Jika ragu**: Jalankan validation di notebook untuk analisis lengkap
                        """
                        )

                        # Warning for potential issues
                        if (
                            distances[0][0] > 20
                            or features.min() < -2
                            or features.max() > 3
                        ):
                            st.warning(
                                """
                            ⚠️ **PERINGATAN**: Terdeteksi kemungkinan mismatch preprocessing!
                            
                            Kemungkinan penyebab:
                            - Metode feature extraction berbeda dengan training
                            - Starting point tidak konsisten
                            - Normalisasi tidak tepat
                            
                            **Rekomendasi**: 
                            1. Coba toggle "Normalize Starting Point"
                            2. Coba metode feature extraction yang berbeda
                            3. Jalankan notebook untuk validasi detail
                            """
                            )
                    except Exception as e:
                        st.warning(f"Could not perform detailed validation: {e}")

                # Additional info
                with st.expander("🔍 Technical Details"):
                    st.write(f"**Image Shape:** {image_array.shape}")
                    st.write(f"**Feature Vector Shape:** {features.shape}")
                    st.write(
                        f"**Feature Range:** [{features.min():.3f}, {features.max():.3f}]"
                    )
                    st.write(f"**Method:** {feature_method}")
                    st.write(f"**K-Neighbors:** {model.n_neighbors}")
                    st.write(f"**Distance Metric:** {model.metric}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info(
                    "💡 Pastikan gambar memiliki objek ikan yang jelas dengan kontras yang baik terhadap background."
                )

    else:
        # Placeholder
        st.info("👆 Upload gambar ikan untuk memulai klasifikasi")

        # Example
        st.markdown("---")
        st.subheader("📝 Tips untuk Gambar yang Baik:")
        st.markdown(
            """
        - Ikan harus terlihat jelas dengan background yang kontras
        - Hindari background yang terlalu kompleks
        - Pencahayaan yang baik
        - Fokus pada 1 ikan saja (objek terbesar akan dideteksi)
        """
        )


if __name__ == "__main__":
    main()
