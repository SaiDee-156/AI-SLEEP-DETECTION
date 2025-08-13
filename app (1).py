import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Page Configuration ------------
st.set_page_config("Sleep State Detection", layout="wide")
st.title(" Sleep State Detection App")

# ----------- Navigation Sidebar ------------
page = st.sidebar.radio("📍 Navigation", ["Overview", "EDA", "Predict"])

# ----------- Load Data & Model ------------
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

@st.cache_data
def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ----------- Histogram Plot Function ------------
def plot_histogram(df, column, color):
    fig, ax = plt.subplots()
    sns.histplot(df[column], bins=30, kde=True, color=color, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ----------- Overview Page ------------
if page == "Overview":
    st.header(" Project Overview")
    st.markdown("This app detects **sleep onset** and **wake-up states** using `anglez` and `enmo` values from a wearable sensor.")

    with st.expander(" Problem Statement"):
        st.markdown("""
        - Detect sleep and wake-up periods using wearable sensor data.  
        - Sleep is estimated from low-movement patterns.
        """)

    with st.expander(" Objective"):
        st.markdown("""
        - Classify sleep vs wake states  
        - Build an ML model that generalizes to real users  
        """)

    with st.expander(" Constraints"):
        st.markdown("""
        - Missing or noisy data  
        - Ensure low false alarms  
        - Simple, real-time capable models  
        """)

# ----------- EDA Page ------------
elif page == "EDA":
    st.header(" Exploratory Data Analysis")
    df = load_data("cleaned_sleep_data.csv")

    # ---- Multi-select Filter Sleep/Wake ----
    st.markdown("### 🔎 Filter by Sleep State")
    state_options = st.multiselect("Select sleep states to display", ["Sleep", "Wake-Up"], default=["Sleep", "Wake-Up"])

    if state_options:
        filter_map = {"Sleep": 1, "Wake-Up": 0}
        selected_values = [filter_map[opt] for opt in state_options]
        df = df[df["sleep"].isin(selected_values)]

    # ---- Histograms ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Anglez")
        plot_histogram(df, "anglez", "#74b9ff")
        st.markdown("- Distribution typical of rest posture")

    with col2:
        st.subheader(" ENMO")
        plot_histogram(df, "enmo", "#81ecec")
        st.markdown("- ENMO reflects movement intensity")

    # ---- Pairplot ----
    st.subheader(" Feature Relationships")
    with st.spinner("Creating pairplot..."):
        pairplot_fig = sns.pairplot(df, vars=['anglez', 'enmo'], hue='sleep', palette='coolwarm')
        st.pyplot(pairplot_fig.fig, use_container_width=True)
        plt.close()

    # ---- Boxplots ----
    st.subheader(" Boxplots")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df["anglez"], ax=axs[0], color='#74b9ff')
    axs[0].set_title("Boxplot: Anglez")
    sns.boxplot(y=df["enmo"], ax=axs[1], color='#81ecec')
    axs[1].set_title("Boxplot: ENMO")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ---- Correlation ----
    st.subheader(" Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[["anglez", "enmo"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ----------- Predict Page ------------
elif page == "Predict":
    st.header(" Sleep Prediction")
    model = load_model("new_sleep_model.pkl")

    # ---- Sleep/Wake Filter Dropdown
    st.markdown("###  Select Sample Type")
    state_choice = st.selectbox("Choose Sample Type", ["Custom Input", "Sleep Sample", "Wake-Up Sample"])

    # ---- Default Values Based on Choice
    if state_choice == "Sleep Sample":
        default_anglez = -45.0
        default_enmo = 0.01
    elif state_choice == "Wake-Up Sample":
        default_anglez = 20.0
        default_enmo = 0.2
    else:
        default_anglez = 0.0
        default_enmo = 0.0

    # ---- Input Sliders
    col1, col2 = st.columns(2)
    with col1:
        anglez = st.slider(" Anglez (-180° to 180°)", -180.0, 180.0, default_anglez)
    with col2:
        enmo = st.slider(" ENMO (0.0 to 1.0)", 0.0, 1.0, default_enmo)

    # ---- Prediction
    if st.button(" Predict Sleep State"):
        input_vector = np.array([[anglez, enmo]])
        prediction = model.predict(input_vector)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vector)[0]
            confidence = round(np.max(proba) * 100, 2)
            st.metric(" Model Confidence", f"{confidence}%")

        labels = {
            0: (" Wake-Up", "You're likely **awake** — motion and posture detected."),
            1: (" Sleep Onset", "Low motion detected — you may be **falling asleep**.")
        }

        label, message = labels.get(prediction, ("❓ Unknown", "⚠️ No clear state detected."))
        st.success(f"** Predicted State:** {label}")
        st.info(message)

        # Display image based on prediction
        if prediction == 1:
            st.image(
                "https://huggingface.co/spaces/Saidee156/AI_SLEEP_DETECTION/resolve/main/th%20(1).jpeg",
                use_container_width=True,
            )
            st.markdown("""
            ###  Personalized Sleep Tips
             **Tips to Fall Asleep Faster**  
            -  Avoid screens 30 mins before bed  
            -  Keep the room cool and dark  
            -  Try deep breathing or meditation  
            -  Stick to a regular sleep schedule  
            """)
        else:
            st.image(
                "https://huggingface.co/spaces/Saidee156/AI_SLEEP_DETECTION/resolve/main/cute-little-boy-wake-up-in-morning-stretching-hands-on-bed-in-bedroom-vector.jpg",
                use_container_width=True,
            )
            st.markdown("""
            ###  Tips to Wake Up Refreshed
            -  Get morning sunlight exposure  
            -  Move or stretch your body  
            -  Eat a light, energizing breakfast  
            -  Cold water splash or shower helps  
            """)










