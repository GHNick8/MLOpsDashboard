import streamlit as st
import pandas as pd
from src.train import train_model
from src.predict import predict_ufo

# --- Load trained model ---
model, vectorizer = train_model("data/ufo.csv")

st.set_page_config(page_title="🛸 UFO Sightings Classifier", layout="centered")

st.title("🛸 UFO Sightings Classifier")
st.markdown("Classify UFO reports as **Explained** or **Unexplained**.")

# --- Single report classification ---
st.subheader("🔍 Classify a single report")
report_text = st.text_area("Enter a UFO report:", "")

if st.button("Classify"):
    if report_text.strip():
        prediction = predict_ufo(report_text, model, vectorizer)
        st.success(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter a UFO report.")

# --- Batch classification ---
st.subheader("📂 Batch classify from CSV")
uploaded_file = st.file_uploader("Upload a CSV with a 'comments' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "comments" in df.columns:
        df["prediction"] = df["comments"].apply(lambda x: predict_ufo(str(x), model, vectorizer))
        st.dataframe(df[["comments", "prediction"]])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv, "ufo_predictions.csv", "text/csv")
    else:
        st.error("Uploaded CSV must contain a 'comments' column.")
