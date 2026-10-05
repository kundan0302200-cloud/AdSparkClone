import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score

# Set base directory and ensure it's in Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def load_preprocessed_data():
    """Load preprocessed training and testing dataset, running preprocessing if files are missing."""
    train_path = os.path.join(BASE_DIR, "data", "preprocessed_train.csv")
    test_path = os.path.join(BASE_DIR, "data", "preprocessed_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from src.data.preprocess import preprocess_data
        print("Preprocessed files not found. Running data preprocessing pipeline...")
        preprocess_data()

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def split_features_target(train_data, test_data):
    """Separate target variable from feature set."""
    target_col = "click" if "click" in train_data.columns else ("PlacementStatus" if "PlacementStatus" in train_data.columns else train_data.columns[-1])
    x_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    x_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    return x_train, x_test, y_train, y_test


def create_model(n_estimators=50, learning_rate=1.0, random_state=42):
    """Initialize AdaBoost Classifier model."""
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    return model


def train_model(model, x_train, y_train):
    """Train AdaBoost Classifier model."""
    if y_train.dtype == float:
        y_train = y_train.astype(int)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate AdaBoost Classifier model with classification metrics and feature importances."""
    if y_test.dtype == float:
        y_test = y_test.astype(int)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    print("\n--- AdaBoost Evaluation Metrics ---")
    print(f"Accuracy     : {acc:.4f}")

    if y_prob is not None and len(set(y_test)) > 1:
        loss = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        print(f"Log Loss     : {loss:.4f}")
        print(f"ROC-AUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if hasattr(model, "feature_importances_") and hasattr(x_test, "columns"):
        importances = pd.Series(model.feature_importances_, index=x_test.columns).sort_values(ascending=False)
        print("\nTop 10 Important Features:")
        print(importances.head(10).to_string())


def save_model(model, filename="ada_boost_model.pkl"):
    """Save trained model to results/models/ and models/ directories."""
    models_dir = os.path.join(BASE_DIR, "results", "models")
    root_models_dir = os.path.join(BASE_DIR, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(root_models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, filename)
    root_model_path = os.path.join(root_models_dir, filename)

    joblib.dump(model, model_path)
    joblib.dump(model, root_model_path)

    print("\nModel saved successfully at:")
    print(model_path)
    print(root_model_path)


if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()

    print("Training Data Shape:", train_data.shape)
    print("Testing Data Shape :", test_data.shape)

    x_train, x_test, y_train, y_test = split_features_target(train_data, test_data)

    model = create_model()
    model = train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model)
