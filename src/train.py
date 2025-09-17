import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import load_and_preprocess

def train_model(data_path="data/ufo.csv"):
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess(data_path)

    with mlflow.start_run():
        # Train
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")

        # MLflow logging (modern syntax)
        mlflow.log_metric("accuracy", acc)

        # Example input for model signature
        input_example = vectorizer.transform(["I saw strange lights in the sky"])
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",  
            input_example=input_example,
            registered_model_name="ufo_sightings_classifier"
        )

    return model, vectorizer

if __name__ == "__main__":
    train_model()
