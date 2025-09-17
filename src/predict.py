from src.preprocess import clean_text

def predict_ufo(report: str, model, vectorizer):
    report = clean_text(report)
    vec = vectorizer.transform([report])
    pred = model.predict(vec)[0]
    return "Unexplained UFO" if pred == 1 else "Explainable (likely)"
