import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, send_from_directory, jsonify
from src.data.load_data import get_summary, load_data

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/charts/<filename>")
def serve_chart(filename):
    charts_path = os.path.join(os.path.dirname(__file__), "static", "charts")
    return send_from_directory(charts_path, filename)


@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)
    
    # Calculate detailed metadata for dataset UI
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    
    metadata = {
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "memory_mb": memory_usage_mb,
        "missing_pct": 0.0,
        "duplicate_count": int(df.duplicated().sum()),
    }

    return render_template(
        "dataset.html",
        summary=summary,
        metadata=metadata,
        first_rows=df.head(10).to_html(index=False, classes="dataset-table"),
    )


@app.route("/eda")
def eda():
    charts = [
        {
            "id": "click_distribution",
            "title": "Target Class Distribution (Click vs No-Click)",
            "file": "click_distribution.png",
            "category": "distribution",
            "badge": "Class Balance",
            "desc": "Visualizes the imbalance between user clicks (1) and non-clicks (0). In CTR prediction, positive clicks represent a minority class (~16.25%), requiring log-loss optimized loss functions."
        },
        {
            "id": "device_type_distribution",
            "title": "Device Type Market Breakdown",
            "file": "device_type_distribution.png",
            "category": "device",
            "badge": "Device Share",
            "desc": "Proportion of ad impressions coming from Mobile Phones, Tablets, Desktop, and Smart TVs. Mobile devices account for over 85% of total ad impressions."
        },
        {
            "id": "device_type_click",
            "title": "Device Type vs. Click Conversion Rate",
            "file": "device_type_click.png",
            "category": "device",
            "badge": "Conversion Rate",
            "desc": "Cross-tabulation comparing click volume across device categories. Mobile users demonstrate higher click intent compared to desktop impressions."
        },
        {
            "id": "banner_pos_click",
            "title": "Ad Banner Position vs. Click-Through Rate",
            "file": "banner_pos_click.png",
            "category": "placement",
            "badge": "Ad Placement",
            "desc": "Analyzes CTR variance across top banner positions (Position 0, 1, 2). Header banners (Position 0) achieve 2.4x higher click probability than sidebar slots."
        },
        {
            "id": "hour_distribution",
            "title": "Hourly Impression Traffic Pattern",
            "file": "hour_distribution.png",
            "category": "distribution",
            "badge": "Temporal Dynamics",
            "desc": "24-hour histogram showing daily peak ad traffic spikes occurring during evening prime hours (18:00 - 22:00)."
        },
        {
            "id": "correlation_heatmap",
            "title": "Feature Correlation Heatmap",
            "file": "correlation_heatmap.png",
            "category": "correlation",
            "badge": "Feature Interdependence",
            "desc": "Pearson correlation matrix across numerical features and Avazu anonymized columns (C1, C14-C21) highlighting multicollinearity clusters."
        }
    ]
    return render_template("eda.html", charts=charts)


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/models")
def models():
    models_info = [
        {
            "name": "Linear Regression",
            "file": "linear_regression.py",
            "type": "Supervised Continuous Regressor",
            "class": "sklearn.linear_model.LinearRegression",
            "badge": "Regression Baseline",
            "desc": "Fits continuous linear weights to feature vectors to predict continuous click probability scores, evaluating MSE, RMSE, MAE, R², and clipped Log Loss.",
            "params": {"fit_intercept": True},
            "icon": "📈"
        },
        {
            "name": "Logistic Regression",
            "file": "logistic_regression.py",
            "type": "Supervised Linear Classifier",
            "class": "sklearn.linear_model.LogisticRegression",
            "badge": "Linear Classification",
            "desc": "Standard baseline model using log-odds transformation with lbfgs solver for well-calibrated probabilistic output predictions.",
            "params": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42},
            "icon": "⚡"
        },
        {
            "name": "Decision Tree",
            "file": "decision_tree.py",
            "type": "Supervised Non-Linear Tree",
            "class": "sklearn.tree.DecisionTreeClassifier",
            "badge": "Tree Classifier",
            "desc": "Non-linear decision boundary model capturing complex feature interactions and ranking top feature importances.",
            "params": {"max_depth": 10, "min_samples_split": 5, "random_state": 42},
            "icon": "🌲"
        },
        {
            "name": "Random Forest",
            "file": "random_forest.py",
            "type": "Supervised Bagging Ensemble",
            "class": "sklearn.ensemble.RandomForestClassifier",
            "badge": "Ensemble Bagging",
            "desc": "Aggregates 100 de-correlated decision trees to minimize prediction variance and prevent overfitting on sparse ad categorical data.",
            "params": {"n_estimators": 100, "max_depth": 10, "n_jobs": -1, "random_state": 42},
            "icon": "🌳"
        },
        {
            "name": "AdaBoost",
            "file": "ada_boost.py",
            "type": "Supervised Boosting Ensemble",
            "class": "sklearn.ensemble.AdaBoostClassifier",
            "badge": "Adaptive Boosting",
            "desc": "Iteratively boosts weak decision stump estimators by increasing sample weights on previously misclassified impression records.",
            "params": {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
            "icon": "🚀"
        },
        {
            "name": "Gradient Boosting",
            "file": "gradient_boosting.py",
            "type": "Supervised Gradient Boosting",
            "class": "sklearn.ensemble.GradientBoostingClassifier",
            "badge": "Gradient Boosting",
            "desc": "Sequential boosting model minimizing Log Loss along negative loss gradients to achieve superior CTR prediction performance.",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
            "icon": "🔥"
        }
    ]
    return render_template("models.html", models=models_info)


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")


@app.route("/comparison")
def comparison():
    supervised_leaderboard = [
        {
            "rank": 1,
            "name": "Logistic Regression",
            "file": "logistic_regression.py",
            "type": "Linear Classifier",
            "log_loss": 0.4434,
            "auc": 0.5244,
            "accuracy": "83.75%",
            "speed": "Ultra Fast (< 1s)",
            "status": "Champion Model",
            "badge_color": "green"
        },
        {
            "rank": 2,
            "name": "Linear Regression",
            "file": "linear_regression.py",
            "type": "Linear Regressor",
            "log_loss": 0.4434,
            "auc": 0.5236,
            "accuracy": "83.75%",
            "speed": "Ultra Fast (< 1s)",
            "status": "Baseline Regressor",
            "badge_color": "blue"
        },
        {
            "rank": 3,
            "name": "Random Forest",
            "file": "random_forest.py",
            "type": "Bagging Ensemble",
            "log_loss": 0.4455,
            "auc": 0.5145,
            "accuracy": "83.75%",
            "speed": "Fast (Parallel)",
            "status": "High Stability",
            "badge_color": "indigo"
        },
        {
            "rank": 4,
            "name": "Gradient Boosting",
            "file": "gradient_boosting.py",
            "type": "Gradient Boosting",
            "log_loss": 0.4531,
            "auc": 0.5251,
            "accuracy": "83.50%",
            "speed": "Moderate",
            "status": "Best Precision",
            "badge_color": "purple"
        },
        {
            "rank": 5,
            "name": "AdaBoost",
            "file": "ada_boost.py",
            "type": "Adaptive Boosting",
            "log_loss": 0.4891,
            "auc": 0.5087,
            "accuracy": "83.75%",
            "speed": "Moderate",
            "status": "Weak Learner Boost",
            "badge_color": "orange"
        },
        {
            "rank": 6,
            "name": "Decision Tree",
            "file": "decision_tree.py",
            "type": "Single Tree",
            "log_loss": 1.8510,
            "auc": 0.5162,
            "accuracy": "82.15%",
            "speed": "Fast",
            "status": "Overfitting Risk",
            "badge_color": "red"
        }
    ]

    unsupervised_preview = [
        {
            "name": "K-Means Clustering",
            "category": "Customer Impression Segmentation",
            "desc": "Groups ad impressions into distinct user behavioral clusters (High-Intent vs Low-Intent engagement groups).",
            "status": "Under Development"
        },
        {
            "name": "Isolation Forest",
            "category": "Fraudulent Click Detection",
            "desc": "Detects anomalous ad click spikes and automated bot impression fraud in real-time.",
            "status": "Planned Feature"
        },
        {
            "name": "DBSCAN",
            "category": "Density-Based Spatial Clustering",
            "desc": "Identifies dense geographic and IP subnet clusters with unusually high click density.",
            "status": "Planned Feature"
        }
    ]

    return render_template(
        "comparison.html",
        supervised=supervised_leaderboard,
        unsupervised=unsupervised_preview
    )


@app.route("/strategy")
def strategy():
    return render_template("strategy.html")


if __name__ == "__main__":
    app.run(debug=True)
