import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss, roc_auc_score

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


def create_model(fit_intercept=True):
    """Initialize Linear Regression model."""
    model = LinearRegression(fit_intercept=fit_intercept)
    return model


def train_model(model, x_train, y_train):
    """Train Linear Regression model."""
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate Linear Regression model with continuous regression & probability classification metrics."""
    y_pred_continuous = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred_continuous)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_continuous)
    r2 = r2_score(y_test, y_pred_continuous)

    print("\n--- Linear Regression Evaluation Metrics ---")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score)    : {r2:.4f}")

    # Binary Classification / Probability Evaluation (if target is binary)
    unique_vals = set(y_test)
    if len(unique_vals) <= 2:
        y_prob = np.clip(y_pred_continuous, 1e-15, 1 - 1e-15)
        y_pred_binary = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred_binary)
        print(f"\nClassification Equivalent Accuracy (Threshold 0.5): {acc:.4f}")
        
        if len(unique_vals) > 1:
            loss = log_loss(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            print(f"Log Loss (Clipped Probabilities)                   : {loss:.4f}")
            print(f"ROC-AUC Score                                      : {auc:.4f}")


def save_model(model, filename="linear_regression_model.pkl"):
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
