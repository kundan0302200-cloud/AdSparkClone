import os
import pandas as pd


def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(nrows=None):
    base_dir = get_base_dir()
    train_path = os.path.join(base_dir, "data", "train.csv")
    sample_sub_path = os.path.join(base_dir, "data", "sampleSubmission.csv")

    if os.path.exists(train_path):
        df = pd.read_csv(train_path, nrows=nrows)
    elif os.path.exists(sample_sub_path):
        df = pd.read_csv(sample_sub_path, nrows=nrows)
    else:
        raise FileNotFoundError(
            f"Dataset file not found. Expected {sample_sub_path} or {train_path}"
        )
    return df


def get_summary(df):
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "click",
    }
    if "click" in df.columns:
        summary["ctr"] = round(df["click"].mean() * 100, 2)
    return summary


if __name__ == "__main__":
    df = load_data()
    print(get_summary(df))
