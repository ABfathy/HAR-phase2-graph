import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import load_model

st.set_page_config(
    page_title="Smart Human Activity Recognition",
    page_icon="🧙‍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {
    0: "🚶 WALKING",
    1: "🧗‍♂️WALKING UPSTAIRS",
    2: "📉 WALKING DOWNSTAIRS",
    3: "🪑 SITTING",
    4: "🧍 STANDING",
    5: "🛏️ LAYING",
}


@st.cache_resource
def load_scaler():
    return joblib.load("./models/scaler.pkl")


@st.cache_resource
def load_pca():
    return joblib.load("./models/pca.pkl")


@st.cache_resource
def load_random_forest():
    return joblib.load("./models/random_forest.pkl")


@st.cache_resource
def load_sgd():
    return joblib.load("./models/sgd.pkl")


@st.cache_resource
def load_lstm():
    return load_model("./models/lstm_model.keras")


@st.cache_resource
def load_lstm_scaler():
    return joblib.load("./models/scaler_lstm.pkl")


@st.cache_data
def load_uci_har_samples():
    """
    Load UCI HAR dataset samples for each activity.
    Returns dict with features (561) and raw signals (128, 9) for each activity.
    """
    DATASET_DIR = "UCI HAR Dataset"

    # Load 561 features (test set)
    X_features = np.loadtxt(f"{DATASET_DIR}/test/X_test.txt")
    y_labels = np.loadtxt(f"{DATASET_DIR}/test/y_test.txt", dtype=int) - 1  # 0-indexed

    # Load raw inertial signals (test set)
    signal_files = [
        "body_acc_x_test.txt",
        "body_acc_y_test.txt",
        "body_acc_z_test.txt",
        "body_gyro_x_test.txt",
        "body_gyro_y_test.txt",
        "body_gyro_z_test.txt",
        "total_acc_x_test.txt",
        "total_acc_y_test.txt",
        "total_acc_z_test.txt",
    ]

    signals = []
    for f in signal_files:
        data = np.loadtxt(f"{DATASET_DIR}/test/Inertial Signals/{f}")
        signals.append(data)

    X_raw = np.stack(signals, axis=2)

    activity_samples = {}
    for activity_id in range(6):
        mask = y_labels == activity_id
        activity_samples[activity_id] = {
            "features": X_features[mask],
            "raw_signals": X_raw[mask],
            "count": int(mask.sum()),
        }

    return activity_samples


def get_real_sample(activity_type, activity_samples, sample_idx=None):
    """
    Get a real sample from the UCI HAR dataset for a given activity.
    Returns both 561 features (for RF/SGD) and raw signals (128, 9) for LSTM.
    """
    samples = activity_samples[activity_type]

    if sample_idx is None:
        idx = np.random.randint(0, samples["count"])
    else:
        idx = sample_idx % samples["count"]

    features = samples["features"][idx]
    raw_signals = samples["raw_signals"][idx]

    return features, raw_signals, idx


def predict_classical(model, features, scaler, pca):
    """Predict using classical ML models (RF or SGD)."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_pca = pca.transform(features_scaled)

    prediction = model.predict(features_pca)[0]
    probabilities = model.predict_proba(features_pca)[0]

    # RF/SGD use 1-indexed labels (1-6), convert to 0-indexed (0-5)
    prediction = prediction - 1

    return prediction, probabilities


def predict_lstm(model, raw_signals, scaler):
    """Predict using LSTM model with raw signals."""
    n_timesteps, n_channels = raw_signals.shape

    # Scale
    raw_scaled = scaler.transform(raw_signals)
    raw_scaled = raw_scaled.reshape(1, n_timesteps, n_channels)

    # Predict
    probabilities = model.predict(raw_scaled, verbose=0)[0]
    prediction = np.argmax(probabilities)

    return prediction, probabilities


def display_model_result(model_name, prediction, probabilities, class_names):
    """Display a single model's prediction result with progress bar and expander."""
    activity_name = class_names[prediction]
    confidence = probabilities[
        prediction if prediction in probabilities else np.argmax(probabilities)
    ]

    # Handle different indexing for classical vs LSTM
    if prediction in class_names:
        confidence = probabilities[list(class_names.keys()).index(prediction)]
    else:
        confidence = np.max(probabilities)

    confidence_pct = confidence * 100

    st.markdown(f"### {model_name}")
    st.markdown(f"**Predicted:** {activity_name}")
    st.metric("Confidence", f"{confidence_pct:.1f}%")
    st.progress(confidence_pct / 100)

    with st.expander("View all class probabilities"):
        for i, (class_id, class_name) in enumerate(class_names.items()):
            prob = probabilities[i] if i < len(probabilities) else 0
            st.write(f"{class_name}: {prob * 100:.1f}%")
            st.progress(prob)


def display_model_result_simple(model_name, prediction, probabilities):
    activity_name = CLASS_NAMES[prediction]
    confidence = probabilities[prediction] * 100
    prob_list = probabilities

    st.markdown(f"### {model_name}")
    st.markdown(f"**Predicted:** {activity_name}")
    st.metric("Confidence", f"{confidence:.1f}%")
    st.progress(float(confidence / 100))

    with st.expander("View all class probabilities"):
        for i, prob in enumerate(prob_list):
            class_name = CLASS_NAMES[i]
            st.write(f"{class_name}: {prob * 100:.1f}%")
            st.progress(float(prob))


@st.cache_data
def compute_model_metrics():
    """
    Compute performance metrics for all models on the test set.
    Returns a dictionary with metrics for each model.
    """
    DATASET_DIR = "UCI HAR Dataset"
    
    # Load 561 features (test set) for classical models
    X_features = np.loadtxt(f"{DATASET_DIR}/test/X_test.txt")
    y_test = np.loadtxt(f"{DATASET_DIR}/test/y_test.txt", dtype=int)
    
    # Load raw inertial signals for LSTM
    signal_files = [
        "body_acc_x_test.txt", "body_acc_y_test.txt", "body_acc_z_test.txt",
        "body_gyro_x_test.txt", "body_gyro_y_test.txt", "body_gyro_z_test.txt",
        "total_acc_x_test.txt", "total_acc_y_test.txt", "total_acc_z_test.txt",
    ]
    signals = []
    for f in signal_files:
        data = np.loadtxt(f"{DATASET_DIR}/test/Inertial Signals/{f}")
        signals.append(data)
    X_raw = np.stack(signals, axis=2)  # (samples, 128, 9)
    
    # Load models and preprocessors
    scaler = joblib.load("./models/scaler.pkl")
    pca = joblib.load("./models/pca.pkl")
    rf_model = joblib.load("./models/random_forest.pkl")
    sgd_model = joblib.load("./models/sgd.pkl")
    lstm_model = load_model("./models/lstm_model.keras")
    lstm_scaler = joblib.load("./models/scaler_lstm.pkl")
    
    # Prepare data for classical models
    X_scaled = scaler.transform(X_features)
    X_pca = pca.transform(X_scaled)
    
    # Get classical model predictions (1-indexed labels)
    rf_preds = rf_model.predict(X_pca)
    sgd_preds = sgd_model.predict(X_pca)
    
    # Prepare data for LSTM
    X_raw_2d = X_raw.reshape(-1, X_raw.shape[2])
    X_raw_scaled = lstm_scaler.transform(X_raw_2d)
    X_raw_scaled = X_raw_scaled.reshape(X_raw.shape)
    
    # Get LSTM predictions (0-indexed labels, so we add 1)
    lstm_proba = lstm_model.predict(X_raw_scaled, verbose=0)
    lstm_preds = np.argmax(lstm_proba, axis=1) + 1  # Convert to 1-indexed
    
    # Compute metrics for each model
    metrics = {}
    
    for name, preds in [("Random Forest", rf_preds), ("SGD", sgd_preds), ("LSTM", lstm_preds)]:
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, preds) * 100,
            "Precision": precision_score(y_test, preds, average="weighted") * 100,
            "Recall": recall_score(y_test, preds, average="weighted") * 100,
            "F1 Score": f1_score(y_test, preds, average="weighted") * 100,
            "predictions": preds,
            "confusion_matrix": confusion_matrix(y_test, preds),
        }
    
    return metrics, y_test


st.sidebar.title("🧙‍ Smart Human Activity Recognition")
st.sidebar.markdown("---")
st.sidebar.markdown("### Available Models")
st.sidebar.markdown("""
#### Classical ML
- **Random Forest** 
- **SGD**
#### Deep Learning
- **LSTM**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 New: Model Comparison")
st.sidebar.markdown("""
Check out the **Model Comparison** tab to see:
- Performance metrics (Accuracy, Precision, Recall, F1)
- Interactive radar charts
- Confusion matrices
- Per-activity breakdown
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Input Formats")
st.sidebar.markdown("""
**Classical (RF/SGD):**  
CSV with 561 features per row

**LSTM:**  
CSV with 128 rows × 9 columns  
(raw inertial signals)
""")


st.title("🧙‍ Smart Human Activity Recognition")
st.markdown("Predict human activities using trained ML models")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Sample Data", "📁 Upload CSV", "📊 Model Comparison"])

with tab1:
    st.markdown("### Select an activity to test")
    st.caption("Real sensor data from UCI HAR test set will be used for predictions")

    # Load real samples
    activity_samples = load_uci_har_samples()

    # Activity selector
    selected_activity = st.selectbox(
        "Activity type",
        options=list(CLASS_NAMES.keys()),
        format_func=lambda x: CLASS_NAMES[x],
    )

    # Sample selector
    max_samples = activity_samples[selected_activity]["count"]
    sample_idx = st.slider(
        "Sample index",
        min_value=0,
        max_value=max_samples - 1,
        value=0,
        help=f"Select from {max_samples} available samples for this activity",
    )

    # Get real sample
    features, raw_signals, idx = get_real_sample(
        selected_activity, activity_samples, sample_idx
    )

    st.info(f"Using real sample #{idx} for: **{CLASS_NAMES[selected_activity]}**")

    st.markdown("---")
    st.markdown("### Prediction Results")

    try:
        scaler = load_scaler()
        pca = load_pca()
        rf_model = load_random_forest()
        sgd_model = load_sgd()
        lstm_model = load_lstm()
        lstm_scaler = load_lstm_scaler()

        # Get predictions from all models
        rf_pred, rf_proba = predict_classical(rf_model, features, scaler, pca)
        sgd_pred, sgd_proba = predict_classical(sgd_model, features, scaler, pca)
        lstm_pred, lstm_proba = predict_lstm(lstm_model, raw_signals, lstm_scaler)

        # Display side-by-side in 3 columns
        col1, col2, col3 = st.columns(3)

        with col1:
            display_model_result_simple("🌲 Random Forest", rf_pred, rf_proba)

        with col2:
            display_model_result_simple("📉 SGD", sgd_pred, sgd_proba)

        with col3:
            display_model_result_simple("🧠 LSTM", lstm_pred, lstm_proba)

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.warning("Make sure all model files exist in the `models/` directory")


# =============================================================================
# TAB 2: UPLOAD CSV
# =============================================================================
with tab2:
    st.markdown("### Upload a CSV file for prediction")
    st.markdown("""
    The app will **auto-detect** the format:
    - **128 rows × 9 columns** → LSTM model
    - **N rows × 561 columns** → Random Forest + SGD models
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file, header=None, sep=r"\s*,\s*|\s+", engine="python"
            )

            first_row = df.iloc[0]
            try:
                first_row.astype(float)
            except (ValueError, TypeError):
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r"\s*,\s*|\s+", engine="python")

            st.write("### Preview of Data")
            st.dataframe(df.head())
            st.caption(f"Shape: {df.shape}")

            st.markdown("---")

            if df.shape == (128, 9):
                st.success("Detected **LSTM format** (128 × 9 raw signals)")
                st.markdown("### Prediction Result")

                lstm_model = load_lstm()
                lstm_scaler = load_lstm_scaler()

                x_scaled = lstm_scaler.transform(df.values)
                x_scaled_3d = x_scaled.reshape(1, 128, 9)

                probabilities = lstm_model.predict(x_scaled_3d, verbose=0)[0]
                prediction = np.argmax(probabilities)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    display_model_result_simple("🧠 LSTM", prediction, probabilities)

            elif df.shape[1] == 561:
                st.success(
                    f"Detected **Classical ML format** (561 features, {df.shape[0]} sample(s))"
                )
                st.markdown("### Prediction Results")

                scaler = load_scaler()
                pca = load_pca()
                rf_model = load_random_forest()
                sgd_model = load_sgd()

                # Process each sample
                x_scaled = scaler.transform(df.values)
                x_pca = pca.transform(x_scaled)

                rf_preds = rf_model.predict(x_pca) - 1  # Convert to 0-indexed
                rf_probas = rf_model.predict_proba(x_pca)
                sgd_preds = sgd_model.predict(x_pca) - 1  # Convert to 0-indexed
                sgd_probas = sgd_model.predict_proba(x_pca)

                if df.shape[0] == 1:
                    col1, col2 = st.columns(2)

                    with col1:
                        display_model_result_simple(
                            "🌲 Random Forest",
                            rf_preds[0],
                            rf_probas[0],
                        )

                    with col2:
                        display_model_result_simple(
                            "📉 SGD", sgd_preds[0], sgd_probas[0]
                        )
                else:
                    results = pd.DataFrame(
                        {
                            "Sample": range(1, len(rf_preds) + 1),
                            "RF Prediction": [CLASS_NAMES[p] for p in rf_preds],
                            "RF Confidence": [
                                f"{np.max(p) * 100:.1f}%" for p in rf_probas
                            ],
                            "SGD Prediction": [CLASS_NAMES[p] for p in sgd_preds],
                            "SGD Confidence": [
                                f"{np.max(p) * 100:.1f}%" for p in sgd_probas
                            ],
                        }
                    )
                    st.dataframe(results, use_container_width=True)

            else:
                st.error(f"Unrecognized format: {df.shape}")
                st.markdown("""
                **Expected formats:**
                - LSTM: 128 rows × 9 columns
                - Classical ML: N rows × 561 columns
                """)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================
with tab3:
    st.markdown("### 📊 Model Performance Comparison")
    st.markdown("Compare the performance of all trained models on the UCI HAR test dataset")
    
    try:
        with st.spinner("Computing metrics on test dataset..."):
            metrics, y_test = compute_model_metrics()
        
        st.success(f"✅ Evaluated all models on **{len(y_test)}** test samples")
        
        # Prepare data for charts
        model_names = list(metrics.keys())
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        
        # Create comparison dataframe
        comparison_data = []
        for model in model_names:
            for metric in metric_names:
                comparison_data.append({
                    "Model": model,
                    "Metric": metric,
                    "Score (%)": metrics[model][metric]
                })
        comparison_df = pd.DataFrame(comparison_data)
        
        # =============================================================================
        # OVERALL METRICS BAR CHART
        # =============================================================================
        st.markdown("---")
        st.markdown("#### 📈 Overall Performance Metrics")
        
        fig_bar = px.bar(
            comparison_df,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            title="Model Performance Comparison",
            color_discrete_map={
                "Random Forest": "#2ecc71",
                "SGD": "#e74c3c",
                "LSTM": "#3498db"
            }
        )
        fig_bar.update_layout(
            yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # =============================================================================
        # METRICS TABLE
        # =============================================================================
        st.markdown("#### 📋 Detailed Metrics Table")
        
        metrics_table = pd.DataFrame({
            "Metric": metric_names,
            "🌲 Random Forest": [f"{metrics['Random Forest'][m]:.2f}%" for m in metric_names],
            "📉 SGD": [f"{metrics['SGD'][m]:.2f}%" for m in metric_names],
            "🧠 LSTM": [f"{metrics['LSTM'][m]:.2f}%" for m in metric_names],
        })
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)
        
        # =============================================================================
        # RADAR CHART
        # =============================================================================
        st.markdown("---")
        st.markdown("#### 🎯 Performance Radar Chart")
        
        fig_radar = go.Figure()
        
        colors = {"Random Forest": "#2ecc71", "SGD": "#e74c3c", "LSTM": "#3498db"}
        
        for model in model_names:
            values = [metrics[model][m] for m in metric_names]
            values.append(values[0])  # Close the radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names + [metric_names[0]],
                fill='toself',
                name=model,
                line_color=colors[model],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=500,
            title="Model Strengths Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # =============================================================================
        # CONFUSION MATRICES
        # =============================================================================
        st.markdown("---")
        st.markdown("#### 🔢 Confusion Matrices")
        
        activity_labels = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
        

        

        
        # =============================================================================
        # PER-CLASS ACCURACY
        # =============================================================================
        st.markdown("---")
        st.markdown("#### 📊 Per-Class Performance")
        
        # Compute per-class accuracy for each model
        per_class_data = []
        for model in model_names:
            cm = metrics[model]["confusion_matrix"]
            for i, activity in enumerate(activity_labels):
                if cm[i].sum() > 0:
                    class_acc = cm[i, i] / cm[i].sum() * 100
                else:
                    class_acc = 0
                per_class_data.append({
                    "Model": model,
                    "Activity": activity,
                    "Accuracy (%)": class_acc
                })
        
        per_class_df = pd.DataFrame(per_class_data)
        
        fig_per_class = px.bar(
            per_class_df,
            x="Activity",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            title="Per-Activity Classification Accuracy",
            color_discrete_map={
                "Random Forest": "#2ecc71",
                "SGD": "#e74c3c",
                "LSTM": "#3498db"
            }
        )
        fig_per_class.update_layout(
            yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
        st.plotly_chart(fig_per_class, use_container_width=True)
        
        # =============================================================================
        # MODEL SUMMARY
        # =============================================================================
        st.markdown("---")
        st.markdown("#### 🏆 Summary")
        
        # Find best model for each metric
        best_models = {}
        for metric in metric_names:
            best_model = max(model_names, key=lambda m: metrics[m][metric])
            best_models[metric] = (best_model, metrics[best_model][metric])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Model by Metric:**")
            for metric, (model, score) in best_models.items():
                emoji = "🌲" if model == "Random Forest" else ("📉" if model == "SGD" else "🧠")
                st.write(f"- **{metric}**: {emoji} {model} ({score:.2f}%)")
        
        with col2:
            st.markdown("**Model Characteristics:**")
            st.write("- 🌲 **Random Forest**: Ensemble of decision trees, good generalization")
            st.write("- 📉 **SGD**: Fast linear classifier with regularization")
            st.write("- 🧠 **LSTM**: Deep learning, captures temporal patterns in raw signals")
        
        # Overall winner
        avg_scores = {model: np.mean([metrics[model][m] for m in metric_names]) for model in model_names}
        overall_best = max(avg_scores, key=avg_scores.get)
        emoji = "🌲" if overall_best == "Random Forest" else ("📉" if overall_best == "SGD" else "🧠")
        
        st.success(f"🏆 **Overall Best Model**: {emoji} {overall_best} (Avg Score: {avg_scores[overall_best]:.2f}%)")
        
    except Exception as e:
        st.error(f"⚠️ Could not compute model metrics: {e}")
        st.warning("Make sure all models are trained and saved in the `models/` directory")
        st.info("Run `phase1_har.py` and `phase2_har.py` to train the models first.")

