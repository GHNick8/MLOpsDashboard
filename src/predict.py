from src.preprocess import clean_text
from src.train import LABEL_MAP


def predict_ufo(report, model, vectorizer):
    """
    Predict UFO classification with confidence scores.

    Args:
        report (str): UFO sighting text
        model: trained ML model
        vectorizer: fitted vectorizer

    Returns:
        tuple: (friendly_label, confidence, raw_class, proba)
            - friendly_label (str): human-readable label with emoji
            - confidence (float): max confidence score (0-1)
            - raw_class (int): numeric class (0 or 1)
            - proba (list): class probabilities in same order as model.classes_
    """
    # Preprocess
    cleaned = clean_text(report)
    X = vectorizer.transform([cleaned])

    # Predictions
    proba = model.predict_proba(X)[0]
    raw_class = int(model.classes_[proba.argmax()])
    confidence = float(max(proba))

    # Map to human-friendly label
    friendly_label = LABEL_MAP.get(raw_class, str(raw_class))

    return friendly_label, confidence, raw_class, proba
