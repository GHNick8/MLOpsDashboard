import sys
from src.train import train_model
from src.predict import predict_ufo

if __name__ == "__main__":
    model, vectorizer = train_model("data/ufo.csv")

    if len(sys.argv) > 1:
        report = " ".join(sys.argv[1:])
        prediction = predict_ufo(report, model, vectorizer)
        print(f"Report: {report}")
        print(f"Prediction: {prediction}")
    else:
        print("No report provided. Example usage:")
        print("python main.py 'I saw bright lights moving rapidly across the sky'")
