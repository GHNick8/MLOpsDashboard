import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def download_ufo_data():
    try:
        import kagglehub
        # Download dataset
        path = kagglehub.dataset_download("NUFORC/ufo-sightings")
        print("Downloaded UFO dataset at:", path)

        # The dataset has multiple files; main one is 'scrubbed.csv'
        csv_path = os.path.join(path, "scrubbed.csv")

        # Copy to project data folder
        os.makedirs("data", exist_ok=True)
        target_path = "data/ufo.csv"
        if not os.path.exists(target_path):
            pd.read_csv(csv_path).to_csv(target_path, index=False)
            print(f"Saved dataset to {target_path}")
        return target_path

    except Exception as e:
        print("KaggleHub download failed:", e)
        print("Make sure you ran `pip install kagglehub` and have a Kaggle API key set up.")
        return None

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def load_and_preprocess(path: str = "data/ufo.csv"):
    if not os.path.exists(path):
        print("Dataset not found, downloading...")
        path = download_ufo_data()
        if path is None:
            raise RuntimeError("Dataset unavailable.")

    df = pd.read_csv(path, low_memory=False)

    # For now, create a placeholder 'label' column
    # Later you should add proper labels (0=explainable, 1=unexplained)
    if "label" not in df.columns:
        df["label"] = [0 if "plane" in str(c).lower() or "star" in str(c).lower() else 1 for c in df["comments"]]

    df["comments"] = df["comments"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["comments"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
