import os
import json
import streamlit as st
import pandas as pd
import altair as alt
from src.train import train_model
from src.predict import predict_ufo

# --- Load trained model ---
model, vectorizer, LABEL_MAP = train_model("data/ufo.csv")

st.set_page_config(page_title="🛸 UFO Sightings Classifier", layout="centered")

st.title("🛸 UFO Sightings Classifier")
st.markdown("Classify UFO reports as **Explained** or **Unexplained** with calibrated confidence scores.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Classifier", "📂 Batch Prediction", "📊 Monitoring"])

# --- Single report classification ---
with tab1:
    st.subheader("🔍 Classify a single report")
    report_text = st.text_area("Enter a UFO report:", "")

    if st.button("Classify", key="single"):
        if not report_text.strip():
            st.warning("Please enter a UFO report.")
        else:
            # Sanity filter: does the text even mention UFO-like things?
            UFO_HINTS = {
                "light","lights","sky","flying","object","disc","ufo","saucer",
                "triangle","craft","orbs","hover","glowing","bright","night","aliens"
            }

            def looks_like_ufo(text: str) -> bool:
                t = text.lower()
                return any(w in t for w in UFO_HINTS)

            if not looks_like_ufo(report_text):
                st.info("Not UFO-like!")
            else:
                friendly_label, confidence, raw_class, proba = predict_ufo(report_text, model, vectorizer)
                confidence_pct = int(confidence * 100)
                THRESH = 75

                if confidence_pct < THRESH:
                    st.warning(f"❓ Uncertain / Not UFO-like!\nConfidence: {confidence_pct}%")
                else:
                    if raw_class == 1:
                        st.success(f"🛸 {friendly_label}\nConfidence: {confidence_pct}%")
                    else:
                        st.info(f"🟢 {friendly_label}\nConfidence: {confidence_pct}%")

                st.progress(confidence)
                                    
# --- Batch classification with visualization ---
with tab2:
    st.subheader("📂 Batch classify from CSV")
    uploaded_file = st.file_uploader("Upload a CSV with a 'comments' column", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "comments" in df.columns:
            preds = df["comments"].apply(lambda x: predict_ufo(str(x), model, vectorizer))
            df["friendly_label"] = preds.apply(lambda x: x[0])
            df["confidence"] = preds.apply(lambda x: int(x[1] * 100))

            st.dataframe(df[["comments", "friendly_label", "confidence"]])

            # Visualization: class distribution
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("friendly_label:N", title="Prediction"),
                    y=alt.Y("count():Q", title="Count"),
                    color="friendly_label:N"
                )
                .properties(title="Prediction Distribution")
            )
            st.altair_chart(chart, use_container_width=True)

            # Download predictions
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions", csv, "ufo_predictions.csv", "text/csv")
        else:
            st.error("Uploaded CSV must contain a 'comments' column.")

# --- Monitoring tab (drift summary) ---
with tab3:
    st.subheader("📊 Drift Monitoring Summary")

    drift_json_path = "reports/drift_report.json"
    if os.path.exists(drift_json_path) and os.path.getsize(drift_json_path) > 0:
        try:
            with open(drift_json_path, "r") as f:
                drift_data = json.load(f)

            drift_detected = drift_data["metrics"][0]["result"]["dataset_drift"]
            share_drifted = drift_data["metrics"][0]["result"]["share_of_drifted_columns"]

            st.metric("Drift Detected", str(drift_detected))
            st.metric("Share of Drifted Columns", f"{share_drifted:.2f}")

        except json.JSONDecodeError:
            st.error("⚠️ Drift report JSON is invalid. Try regenerating it with monitor.py.")
        except Exception:
            st.warning("⚠️ Could not parse drift report JSON structure.")

        # Add wrapper + embed report
        drift_html_path = "reports/drift_report.html"
        if os.path.exists(drift_html_path):
            with open(drift_html_path, "r", encoding="utf-8") as f:
                drift_html = f.read()

            st.markdown(
                """
                <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; background-color: #f9f9f9;">
                    <h4 style="color:#4CAF50;">Embedded Drift Report</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.components.v1.html(drift_html, height=800, scrolling=True)

            # Download button
            with open(drift_html_path, "rb") as f:
                st.download_button(
                    "⬇️ Download full drift report",
                    f,
                    file_name="drift_report.html",
                    mime="text/html"
                )

    else:
        st.info("No drift report available yet. Run `monitor.py` to generate one.")