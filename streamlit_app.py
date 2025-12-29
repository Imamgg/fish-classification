"""
Fish Classification - Streamlit App with Explainability
FASE 3: Deployment Application + Explanation Features

Upload gambar ikan → Prediksi spesies dengan visualisasi + PENJELASAN
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
st.set_page_config(
    page_title="Fish Classifier with Explanation", page_icon="🐟", layout="wide"
)


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


@st.cache_data
def load_training_data():
    """Load training data untuk explainability"""
    try:
        # Parse ARFF file
        with open("data/Fish/Fish_TRAIN.arff", "r") as f:
            lines = f.readlines()

        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "@data":
                data_start_idx = i + 1
                break

        X_list = []
        y_list = []

        for line in lines[data_start_idx:]:
            line = line.strip()
            if line and not line.startswith("%"):
                parts = line.split(",")
                features = [float(x) for x in parts[:-1]]
                label = parts[-1].strip()
                X_list.append(features)
                y_list.append(label)

        return np.array(X_list), np.array(y_list)
    except Exception as e:
        st.warning(f"Could not load training data: {e}")
        return None, None


# ========== IMAGE PROCESSING FUNCTIONS ==========
def preprocess_and_segment(image):
    """Preprocessing dan Binary Segmentation"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

    return binary_clean


def extract_and_resample_contour(binary_mask, n_points):
    """Extract kontur dan resample ke N titik"""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("❌ Tidak ada kontur ditemukan di gambar")

    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()

    if contour_points.ndim == 1:
        raise ValueError("❌ Kontur terlalu kecil")

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
        centroid_x = np.mean(contour_points[:, 0])
        centroid_y = np.mean(contour_points[:, 1])

        distances = np.sqrt(
            (contour_points[:, 0] - centroid_x) ** 2
            + (contour_points[:, 1] - centroid_y) ** 2
        )
        time_series = distances

    elif method == "turn_angle":
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

    scaler = MinMaxScaler()
    time_series_normalized = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    time_series_scaled = time_series_normalized * 3 - 1

    return time_series_scaled


def image_to_features(image, n_features, method="centroid_distance"):
    """Pipeline: Image → Features untuk prediksi"""
    binary_mask = preprocess_and_segment(image)
    contour_resampled = extract_and_resample_contour(binary_mask, n_features)
    time_series = contour_to_timeseries(contour_resampled, method=method)

    return time_series, binary_mask, contour_resampled


# ========== STREAMLIT UI ==========
def main():
    # Header
    st.title("🐟 Fish Species Classifier")
    st.markdown(
        """
    Upload gambar ikan untuk prediksi spesies dengan **penjelasan lengkap** mengapa ikan masuk ke class tersebut!
    """
    )

    # Load model and training data
    model, encoder, metadata = load_model_and_metadata()
    X_train, y_train = load_training_data()

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
        st.write(
            f"**Training Samples:** {len(X_train) if X_train is not None else 'N/A'}"
        )

        st.markdown("---")
        st.header("⚙️ Settings")
        feature_method = st.selectbox(
            "Feature Method:",
            ["centroid_distance", "turn_angle"],
            help="Metode konversi kontur ke time series",
        )

        show_advanced = st.checkbox("Show Advanced Analysis", value=True)

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

                # Get neighbors for explainability
                k_neighbors = min(5, len(X_train)) if X_train is not None else 5
                distances, indices = model.kneighbors(
                    features_reshaped, n_neighbors=k_neighbors
                )

                # Display Results
                st.success("✅ Processing completed!")

                # Prediction result
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🎯 Predicted Species", predicted_class)

                with col2:
                    st.metric("📏 Distance to Nearest", f"{distances[0][0]:.4f}")

                with col3:
                    # Calculate confidence based on distance
                    max_dist = 50.0
                    confidence = max(0, 100 * (1 - distances[0][0] / max_dist))
                    st.metric("🎯 Confidence", f"{confidence:.1f}%")

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

                # ========== EXPLAINABILITY SECTION ==========
                st.markdown("---")
                st.subheader("🔍 Explanation: Kenapa Ikan Ini Masuk Class Tersebut?")

                # Explain KNN logic
                st.info(
                    f"""
                **Metode Klasifikasi: K-Nearest Neighbors (KNN)**
                
                Model mencari **{k_neighbors} contoh paling mirip** dari training data berdasarkan **jarak Euclidean** 
                dari time series features. Ikan Anda diklasifikasi sebagai **{predicted_class}** karena 
                paling mirip dengan contoh-contoh dari class tersebut di training set.
                """
                )

                if X_train is not None and y_train is not None:
                    # Get neighbor classes
                    neighbor_classes = y_train[indices[0]]
                    neighbor_distances = distances[0]

                    # Class distribution in neighborhood
                    st.markdown("### 📊 Distribusi Class di Neighborhood")
                    col_dist1, col_dist2 = st.columns(2)

                    with col_dist1:
                        # Count classes
                        unique_classes, counts = np.unique(
                            neighbor_classes, return_counts=True
                        )

                        # Create bar chart
                        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
                        colors = [
                            "green" if cls == predicted_class else "gray"
                            for cls in unique_classes
                        ]
                        ax_dist.bar(
                            unique_classes,
                            counts,
                            color=colors,
                            alpha=0.7,
                            edgecolor="black",
                        )
                        ax_dist.set_xlabel("Species", fontweight="bold")
                        ax_dist.set_ylabel(
                            "Count in Top-5 Neighbors", fontweight="bold"
                        )
                        ax_dist.set_title(
                            f"Class Distribution (Top-{k_neighbors} Neighbors)",
                            fontweight="bold",
                        )
                        ax_dist.grid(axis="y", alpha=0.3)
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig_dist)

                    with col_dist2:
                        st.markdown(f"**📋 {k_neighbors} Nearest Neighbors:**")
                        for i, (dist, cls) in enumerate(
                            zip(neighbor_distances, neighbor_classes)
                        ):
                            icon = "✅" if cls == predicted_class else "⚠️"
                            st.write(
                                f"{i+1}. {icon} Class: **{cls}** | Distance: {dist:.4f}"
                            )

                        # Voting explanation
                        st.markdown("---")
                        st.markdown("**🗳️ Voting Hasil:**")
                        for cls, count in zip(unique_classes, counts):
                            percentage = (count / k_neighbors) * 100
                            bar_length = int(percentage / 10)
                            bar = "█" * bar_length + "░" * (10 - bar_length)
                            st.write(
                                f"**{cls}**: {bar} {count}/{k_neighbors} ({percentage:.0f}%)"
                            )

                    # Visual comparison with nearest neighbor
                    st.markdown("---")
                    st.markdown(
                        "### 📈 Perbandingan Time Series dengan Nearest Neighbor"
                    )

                    fig_compare, axes_compare = plt.subplots(2, 2, figsize=(14, 8))

                    # Plot input time series
                    axes_compare[0, 0].plot(
                        features, linewidth=2, color="blue", label="Your Fish"
                    )
                    axes_compare[0, 0].set_title(
                        "Your Fish - Time Series", fontweight="bold"
                    )
                    axes_compare[0, 0].set_xlabel("Point Index")
                    axes_compare[0, 0].set_ylabel("Value")
                    axes_compare[0, 0].grid(True, alpha=0.3)
                    axes_compare[0, 0].legend()

                    # Plot nearest neighbor
                    nearest_idx = indices[0][0]
                    nearest_features = X_train[nearest_idx]
                    nearest_class = y_train[nearest_idx]

                    axes_compare[0, 1].plot(
                        nearest_features,
                        linewidth=2,
                        color="green",
                        label=f"Nearest: {nearest_class}",
                    )
                    axes_compare[0, 1].set_title(
                        f"Nearest Neighbor - {nearest_class} (Dist: {distances[0][0]:.4f})",
                        fontweight="bold",
                    )
                    axes_compare[0, 1].set_xlabel("Point Index")
                    axes_compare[0, 1].set_ylabel("Value")
                    axes_compare[0, 1].grid(True, alpha=0.3)
                    axes_compare[0, 1].legend()

                    # Overlay comparison
                    axes_compare[1, 0].plot(
                        features,
                        linewidth=2,
                        color="blue",
                        alpha=0.7,
                        label="Your Fish",
                    )
                    axes_compare[1, 0].plot(
                        nearest_features,
                        linewidth=2,
                        color="green",
                        alpha=0.7,
                        label=f"Nearest: {nearest_class}",
                    )
                    axes_compare[1, 0].set_title(
                        "Overlay Comparison", fontweight="bold"
                    )
                    axes_compare[1, 0].set_xlabel("Point Index")
                    axes_compare[1, 0].set_ylabel("Value")
                    axes_compare[1, 0].grid(True, alpha=0.3)
                    axes_compare[1, 0].legend()

                    # Difference plot
                    difference = features - nearest_features
                    axes_compare[1, 1].plot(difference, linewidth=2, color="red")
                    axes_compare[1, 1].axhline(
                        y=0, color="black", linestyle="--", linewidth=1
                    )
                    axes_compare[1, 1].set_title(
                        "Difference (Your Fish - Nearest Neighbor)", fontweight="bold"
                    )
                    axes_compare[1, 1].set_xlabel("Point Index")
                    axes_compare[1, 1].set_ylabel("Difference")
                    axes_compare[1, 1].grid(True, alpha=0.3)
                    axes_compare[1, 1].fill_between(
                        range(len(difference)), difference, alpha=0.3, color="red"
                    )

                    plt.tight_layout()
                    st.pyplot(fig_compare)

                    # Statistical comparison
                    st.markdown("### 📊 Statistical Similarity Analysis")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                    with col_stat1:
                        mae = np.mean(np.abs(difference))
                        st.metric("Mean Abs Difference", f"{mae:.4f}")

                    with col_stat2:
                        max_diff = np.max(np.abs(difference))
                        st.metric("Max Difference", f"{max_diff:.4f}")

                    with col_stat3:
                        correlation = np.corrcoef(features, nearest_features)[0, 1]
                        st.metric("Correlation", f"{correlation:.4f}")

                    with col_stat4:
                        rmse = np.sqrt(np.mean(difference**2))
                        st.metric("RMSE", f"{rmse:.4f}")

                    # Advanced analysis
                    if show_advanced:
                        with st.expander(
                            "🔬 Advanced: Region-wise Contribution Analysis",
                            expanded=False,
                        ):
                            st.markdown(
                                """
                            Analisis ini menunjukkan **bagian mana** dari time series yang paling berkontribusi 
                            terhadap similarity/distance dengan nearest neighbor.
                            """
                            )

                            # Divide into regions
                            n_regions = 5
                            region_size = len(features) // n_regions

                            region_contributions = []
                            region_labels_detailed = []

                            for i in range(n_regions):
                                start_idx = i * region_size
                                end_idx = (
                                    (i + 1) * region_size
                                    if i < n_regions - 1
                                    else len(features)
                                )

                                region_diff = (
                                    features[start_idx:end_idx]
                                    - nearest_features[start_idx:end_idx]
                                )
                                region_distance = np.sqrt(np.sum(region_diff**2))
                                region_contributions.append(region_distance)
                                region_labels_detailed.append(
                                    f"Region {i+1}\n({start_idx}-{end_idx})"
                                )

                            # Plot region contributions
                            fig_regions, ax_regions = plt.subplots(figsize=(10, 5))

                            colors_regions = plt.cm.RdYlGn_r(
                                np.array(region_contributions)
                                / max(region_contributions)
                            )
                            bars = ax_regions.bar(
                                range(n_regions),
                                region_contributions,
                                color=colors_regions,
                                edgecolor="black",
                                alpha=0.8,
                            )

                            ax_regions.set_xticks(range(n_regions))
                            ax_regions.set_xticklabels(
                                [f"Region {i+1}" for i in range(n_regions)]
                            )
                            ax_regions.set_xlabel(
                                "Time Series Region", fontweight="bold"
                            )
                            ax_regions.set_ylabel(
                                "Contribution to Distance", fontweight="bold"
                            )
                            ax_regions.set_title(
                                "Region-wise Distance Contribution", fontweight="bold"
                            )
                            ax_regions.grid(axis="y", alpha=0.3)

                            # Annotate values
                            for i, (bar, val) in enumerate(
                                zip(bars, region_contributions)
                            ):
                                height = bar.get_height()
                                ax_regions.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height,
                                    f"{val:.2f}",
                                    ha="center",
                                    va="bottom",
                                    fontweight="bold",
                                )

                            # Annotate max contribution
                            max_contrib_idx = np.argmax(region_contributions)
                            ax_regions.annotate(
                                "Highest\nDifference",
                                xy=(
                                    max_contrib_idx,
                                    region_contributions[max_contrib_idx],
                                ),
                                xytext=(
                                    max_contrib_idx,
                                    region_contributions[max_contrib_idx] * 1.15,
                                ),
                                ha="center",
                                fontweight="bold",
                                fontsize=10,
                                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                            )

                            plt.tight_layout()
                            st.pyplot(fig_regions)

                            st.markdown(
                                f"""
                            **🎯 Kesimpulan:**
                            - **Region {max_contrib_idx + 1}** memiliki perbedaan terbesar (distance: {region_contributions[max_contrib_idx]:.2f})
                            - Ini berarti bagian contour ini paling berbeda dengan nearest neighbor
                            - Jika prediksi salah, region ini mungkin perlu diperbaiki (preprocessing atau segmentasi)
                            """
                            )

                            # Show all neighbors comparison
                            st.markdown("---")
                            st.markdown(
                                f"**👥 Comparison dengan Semua {k_neighbors} Nearest Neighbors:**"
                            )

                            fig_all, axes_all = plt.subplots(
                                1, k_neighbors, figsize=(4 * k_neighbors, 4)
                            )
                            if k_neighbors == 1:
                                axes_all = [axes_all]

                            for idx, (neighbor_idx, dist, neighbor_class) in enumerate(
                                zip(indices[0], distances[0], neighbor_classes)
                            ):
                                ax = axes_all[idx]
                                neighbor_feat = X_train[neighbor_idx]

                                ax.plot(
                                    features, alpha=0.7, label="Your Fish", linewidth=2
                                )
                                ax.plot(
                                    neighbor_feat,
                                    alpha=0.7,
                                    label=f"{neighbor_class}",
                                    linewidth=2,
                                )
                                ax.set_title(
                                    f"Neighbor #{idx+1}\n{neighbor_class}\nDist: {dist:.3f}",
                                    fontweight="bold",
                                )
                                ax.legend(fontsize=8)
                                ax.grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig_all)

                else:
                    st.warning(
                        "⚠️ Training data tidak tersedia untuk explainability lengkap."
                    )

                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Image Shape:** {image_array.shape}")
                    st.write(f"**Feature Vector Shape:** {features.shape}")
                    st.write(
                        f"**Feature Range:** [{features.min():.3f}, {features.max():.3f}]"
                    )
                    st.write(f"**Method:** {feature_method}")
                    st.write(f"**K-Neighbors:** {model.n_neighbors}")
                    st.write(f"**Distance Metric:** {model.metric}")
                    if X_train is not None:
                        st.write(f"**Training Samples Used:** {len(X_train)}")

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
        col_tip1, col_tip2 = st.columns(2)

        with col_tip1:
            st.markdown(
                """
            **✅ GOOD:**
            - Ikan terlihat jelas
            - Background kontras
            - Pencahayaan merata
            - Fokus pada 1 ikan
            - Resolusi cukup tinggi
            """
            )

        with col_tip2:
            st.markdown(
                """
            **❌ BAD:**
            - Background kompleks
            - Multiple fish
            - Pencahayaan buruk
            - Gambar blur/rusak
            - Ikan terlalu kecil
            """
            )


if __name__ == "__main__":
    main()
