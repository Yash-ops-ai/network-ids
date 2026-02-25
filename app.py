import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Network Anomaly Detection System")
st.write("Upload a CSV file containing network flows to detect attacks.")


@st.cache_resource
@st.cache_resource
def load_models():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    top_features = joblib.load('models/top_features.pkl')
    return model, scaler, le, top_features

model, scaler, le, top_features = load_models()


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


if uploaded_file is not None:
    st.write("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.write(f"Total rows: {len(df)}")
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.subheader("Predictions")
    st.write("Processing your file...")
    X_new = df.select_dtypes(include=[np.number])

    X_new = X_new[top_features]
    X_new = scaler.transform(X_new.values)
    y_pred = model.predict(X_new)
    predicted_labels = le.inverse_transform(y_pred)
    df['Prediction'] = predicted_labels
    st.dataframe(df[['Prediction']].join(df.drop('Prediction', axis=1)).head(20))
    st.subheader("Attack Distribution")
    attack_counts = df['Prediction'].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    attack_counts.plot(kind='bar', ax=ax[0], color='steelblue')
    ax[0].set_title('Attack Type Count')
    ax[0].set_xlabel('Attack Type')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].pie(attack_counts, labels=attack_counts.index, autopct='%1.1f%%')
    ax[1].set_title('Traffic Distribution')

    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Summary")
    total = len(df)
    benign = len(df[df['Prediction'] == 'BENIGN'])
    attacks = total - benign

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", total)
    col2.metric("Benign", benign)
    col3.metric("Attacks Detected", attacks)



