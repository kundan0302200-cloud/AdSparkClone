import os
import sys
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score

# Set base directory and ensure it's in Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def load_preprocessed_data():
    train_path = os.path.join(BASE_DIR, "data", "preprocessed_train.csv")
    test_path = os.path.join(BASE_DIR, "data", "preprocessed_test.csv")
    
    # Auto-generate preprocessed data if not yet created
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from src.data.preprocess import preprocess_data
        print("Preprocessed files not found. Running data preprocessing pipeline...")
        preprocess_data()

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def split_features_target(train_data, test_data):
    target_col = "click" if "click" in train_data.columns else ("PlacementStatus" if "PlacementStatus" in train_data.columns else train_data.columns[-1])
    x_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    x_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    return x_train, x_test, y_train, y_test


def create_model():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    return model


def train_model(model, x_train, y_train):
    if y_train.dtype == float:
        y_train = y_train.astype(int)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    if y_test.dtype == float:
        y_test = y_test.astype(int)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    if y_prob is not None and len(set(y_test)) > 1:
        loss = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        print(f"Log Loss: {loss:.4f}")
        print(f"ROC-AUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def save_model(model):
    # Save in both results/models/ and models/ for compatibility
    models_dir = os.path.join(BASE_DIR, "results", "models")
    root_models_dir = os.path.join(BASE_DIR, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(root_models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    root_model_path = os.path.join(root_models_dir, "logistic_regression_model.pkl")

    joblib.dump(model, model_path)
    joblib.dump(model, root_model_path)

    print("\nModel saved successfully at:")
    print(model_path)
    print(root_model_path)


if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()

    print("Training Data Shape:")
    print(train_data.shape)

    print("\nTesting Data Shape:")
    print(test_data.shape)

    # Split features and target
    x_train, x_test, y_train, y_test = split_features_target(
        train_data,
        test_data
    )

    # Create model
    model = create_model()

    # Train model
    model = train_model(model, x_train, y_train)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Save model
    save_model(model)
