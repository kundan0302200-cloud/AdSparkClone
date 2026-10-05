import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from pandas import isnull
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

try:
    from src.data import load_data
except ImportError:
    import load_data


def split_data(df, target_column="click", drop_columns=None):
    df_copy = df.copy()
    
    # Auto-adjust target_column if specified target is not in df
    if target_column not in df_copy.columns:
        if "click" in df_copy.columns:
            target_column = "click"
        elif "PlacementStatus" in df_copy.columns:
            target_column = "PlacementStatus"
        else:
            target_column = df_copy.columns[-1]

    if drop_columns:
        cols_to_drop = [col for col in drop_columns if col in df_copy.columns]
        df_copy = df_copy.drop(columns=cols_to_drop)

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    stratify = y if (y.dtype == 'object' or y.nunique() < 20) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def handle_missing_values(X_train, X_test, numerical_features):
    X_train = X_train.copy()
    X_test = X_test.copy()

    valid_numerical = [col for col in numerical_features if col in X_train.columns]
    if not valid_numerical:
        return X_train, X_test, None

    imputer = SimpleImputer(strategy="median")

    # Fit only on training data
    X_train[valid_numerical] = imputer.fit_transform(
        X_train[valid_numerical]
    )

    # Transform test data using the same imputer
    X_test[valid_numerical] = imputer.transform(
        X_test[valid_numerical]
    )
    return X_train, X_test, imputer


def identify_features(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return numerical_features, categorical_features


def standardize_data(X_train, X_test, numerical_features):
    X_train = X_train.copy()
    X_test = X_test.copy()

    valid_numerical = [col for col in numerical_features if col in X_train.columns]
    if not valid_numerical:
        return X_train, X_test, None

    scaler = StandardScaler()

    # Fit only on training data
    X_train[valid_numerical] = scaler.fit_transform(
        X_train[valid_numerical]
    )

    # Use the same scaler for test data
    X_test[valid_numerical] = scaler.transform(
        X_test[valid_numerical]
    )
    return X_train, X_test, scaler


def one_hot_encode_data(X_train, X_test, categorical_features):
    X_train = X_train.copy()
    X_test = X_test.copy()

    valid_categorical = [col for col in categorical_features if col in X_train.columns]
    if not valid_categorical:
        return X_train, X_test, None

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[valid_categorical]
    )

    # Use the same encoder for test data
    test_encoded = encoder.transform(
        X_test[valid_categorical]
    )

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(
        valid_categorical
    )

    # Create DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # Remove original categorical columns
    X_train = X_train.drop(columns=valid_categorical)
    X_test = X_test.drop(columns=valid_categorical)

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


def ordinal_encode_data(X_train, X_test, ordinal_features):
    X_train = X_train.copy()
    X_test = X_test.copy()

    valid_ordinal = [col for col in ordinal_features if col in X_train.columns]
    if not valid_ordinal:
        return X_train, X_test, None

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[valid_ordinal]
    )

    # Transform test data
    test_encoded = encoder.transform(
        X_test[valid_ordinal]
    )

    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=valid_ordinal,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=valid_ordinal,
        index=X_test.index
    )

    # Remove original ordinal columns
    X_train = X_train.drop(columns=valid_ordinal)
    X_test = X_test.drop(columns=valid_ordinal)

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


def split_X_data(df, drop_columns=None):  # for unsupervised learning
    if drop_columns is None:
        drop_columns = []

    existing_to_drop = [col for col in drop_columns if col in df.columns]
    X = df.drop(columns=existing_to_drop)

    return X


def preprocess_data(df=None, target_column="click", drop_columns=None):
    if df is None:
        df = load_data.load_data()

    if drop_columns is None:
        drop_columns = ["id"]

    # Target column validation
    if target_column not in df.columns:
        if "click" in df.columns:
            target_column = "click"
        elif "PlacementStatus" in df.columns:
            target_column = "PlacementStatus"
        else:
            target_column = df.columns[-1]

    X_train, X_test, y_train, y_test = split_data(df, target_column=target_column, drop_columns=drop_columns)

    numerical_features, categorical_features = identify_features(X_train)

    # Partition categorical features based on cardinality or specific preset
    one_hot_features = [col for col in categorical_features if X_train[col].nunique() <= 50]
    ordinal_features = [col for col in categorical_features if col not in one_hot_features]

    # Clean missing values
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)

    # Standardize numerical features
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)

    # Encode categorical features
    if one_hot_features:
        X_train, X_test, ohe = one_hot_encode_data(X_train, X_test, one_hot_features)

    if ordinal_features:
        X_train, X_test, ord_enc = ordinal_encode_data(X_train, X_test, ordinal_features)

    # Prepare outputs with target column attached
    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df[target_column] = y_train.values
    test_df[target_column] = y_test.values

    # Determine data directory path
    base_dir = load_data.get_base_dir()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save preprocessed datasets
    train_df.to_csv(os.path.join(data_dir, "preprocessed_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "preprocessed_test.csv"), index=False)

    return train_df, test_df


if __name__ == "__main__":
    # Load dataset
    df = load_data.load_data()
    print("Original Dataset Shape:")
    print(df.shape)

    # Identify target column present in dataset
    target_col = "click" if "click" in df.columns else ("PlacementStatus" if "PlacementStatus" in df.columns else df.columns[-1])

    drop_cols = ["id"] if "id" in df.columns else []

    X_train, X_test, y_train, y_test = split_data(df, target_column=target_col, drop_columns=drop_cols)

    print("\nTraining Shape:")
    print(X_train.shape)
    print("\nTesting Shape:")
    print(X_test.shape)

    numerical_features, categorical_features = identify_features(X_train)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # Categorical split: check if custom placement features or avazu features present
    if "Gender" in categorical_features:
        one_hot_features = [
            "Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"
        ]
        ordinal_features = [
            "CollegeTier", "CGPA_Tier"
        ]
    else:
        one_hot_features = [c for c in categorical_features if X_train[c].nunique() <= 50]
        ordinal_features = [c for c in categorical_features if c not in one_hot_features]

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    if numerical_features:
        print("\nNumerical missing values count:")
        print(X_train[numerical_features].isnull().sum())
    print("Missing Value Handling Completed.")

    X_train, X_test, scaler = standardize_data(
        X_train,
        X_test,
        numerical_features
    )
    print("\nStandardization Completed.")

    if one_hot_features:
        X_train, X_test, encoder = one_hot_encode_data(
            X_train,
            X_test,
            one_hot_features
        )
        print("\nOne-Hot Encoding Completed.")

    if ordinal_features:
        X_train, X_test, ordinal_encoder = ordinal_encode_data(
            X_train,
            X_test,
            ordinal_features
        )
        print("\nOrdinal Encoding Completed.")

    X_train[target_col] = y_train.values
    X_test[target_col] = y_test.values

    # Create data folder path
    base_dir = load_data.get_base_dir()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save preprocessed data
    X_train.to_csv(os.path.join(data_dir, "preprocessed_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "preprocessed_test.csv"), index=False)

    print("\nTraining Data Head:")
    print(X_train.head())

    print("\nTesting Data Head:")
    print(X_test.head())

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)
