# ==============================
# Mushroom Classification App
# Built with Streamlit
# ==============================


# Streamlit for web app development
import streamlit as st

# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# model explainability
import shap

# PCA for dimensionality reduction
from sklearn.decomposition import PCA


# Load the dataset
@st.cache_data
def load_data():
    import os

    data_path = os.path.join(os.path.dirname(__file__), "mushrooms.csv")
    data = pd.read_csv(data_path)
    label = LabelEncoder()

    # encode categorical values into numerical values
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


# Exploratory data analysis (EDA)
def eda(df):
    """
    display basic information, class distribution and coorelation heatmap.
    """
    st.subheader("Basic Data Information")
    st.write(f"Data Shape: {df.shape}")  # display shape of the data
    st.write(df.describe())  # summary stats
    st.write(df.head())  # display first 5 rows

    # Class distribution plot
    st.subheader("Class Distribution")
    fig = px.histogram(df, x="class", title="Edible vs Poisonous", color="class")
    st.plotly_chart(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# Split data function
@st.cache_data
def split_data(df):
    """
    splits the data into training and testing sets
    """
    X = df.drop(columns=["class"])  # features
    y = df["class"]  # target
    return train_test_split(
        X, y, test_size=0.3, random_state=0
    )  # 70% train and 30% test


# Model training with the hyperparameter tuning
def tune_model(model, X_train, y_train, param_grid):
    """
    Uses GridSearchCV to find the best hyperparameters for the selected model
    """
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)  # train with different hyperparameters
    return grid_search.best_estimator_, grid_search.best_params_


# Plot model metrics
def plot_metrics(metrics_list, model, X_test, y_test, class_names):
    """
    plots different evaluation metrics, like Confusing Metrix, ROC Curve and Precision-Recall Curve
    """
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        display = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=class_names
        )
        st.pyplot(display.figure_)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        display = RocCurveDisplay.from_estimator(model, X_test, y_test)
        st.pyplot(display.figure_)

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        st.pyplot(display.figure_)


# PCA visualization
def pca_visualization(X, y):
    """
    applies PCA to reduce data dimensions and visualizes it in 2D
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)  # reduce features to 2D
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["class"] = y.reset_index(drop=True)
    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="class",
        title="PCA - Dimensionality Reduction",
    )
    st.plotly_chart(fig)


# Main unction for STreamlit application
def main():
    st.title("Mushroom Classification App 🍄")
    st.sidebar.title("Mushroom Classification App 🍄")
    df = load_data()

    # sidebar option to show raw data
    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    # feature scaling option
    scale_data = st.sidebar.checkbox("Apply Feature Scaling", False)
    if scale_data:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    else:
        df_scaled = df

    # EDA section
    if st.sidebar.checkbox("Show Data Exploration", False):
        eda(df)

    # split data
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    class_names = ["Edible", "Poisonous"]

    # choose classifier
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        (
            "Support Vector Machine (SVM)",
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "AdaBoost",
            "Gradient Boosting",
        ),
    )

    # define models for their hyperparameters
    param_grid = {}
    model = None
    if classifier == "Support Vector Machine (SVM)":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

    elif classifier == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider("Max Iterations", 100, 500)
        model = LogisticRegression(C=C, max_iter=max_iter)
        param_grid = {"C": [0.1, 1, 10], "max_iter": [100, 200, 300]}

    elif classifier == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 100, 500, step=10)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Bootstrap", ("True", "False"))
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=(bootstrap == "True"),
        )
        param_grid = {"n_estimators": [100, 200, 500], "max_depth": [10, 15, 20]}

    elif classifier == "XGBoost":
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, step=10)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, step=0.01)
        max_depth = st.sidebar.slider("Max Depth", 3, 20, step=1)
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            use_label_encoder=False,
        )
        param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.3]}

    elif classifier == "AdaBoost":
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, step=10)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, step=0.01)
        model = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate
        )
        param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}

    elif classifier == "Gradient Boosting":
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, step=10)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, step=0.01)
        max_depth = st.sidebar.slider("Max Depth", 3, 20, step=1)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 5, 7],
        }

    # Grid search for best parameters
    if st.sidebar.checkbox("Optimize Hyperparameters with GridSearchCV", False):
        model, best_params = tune_model(model, X_train, y_train, param_grid)
        st.write(f"Best Parameters: {best_params}")

    # Cross-validation
    cross_val = st.sidebar.checkbox("Perform Cross-Validation", False)
    if cross_val:
        skf = StratifiedKFold(n_splits=5)
        st.subheader("Cross-Validation Results")
        for train_idx, test_idx in skf.split(X_train, y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
            model.fit(X_train_cv, y_train_cv)
            score = model.score(X_test_cv, y_test_cv)
            st.write(f"Cross-validation score: **{score:.2f}**")

    # metrics and predictions
    st.sidebar.subheader("Choose Metrics to Plot")
    metrics = st.sidebar.multiselect(
        "Metrics", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
    )

    if st.sidebar.button("Classify", key="classify"):
        st.subheader(f"{classifier} Results")
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write(f"Accuracy: **{accuracy:.2f}**")
        st.write(f"Precision: **{precision_score(y_test, y_pred):.2f}**")
        st.write(f"Recall: **{recall_score(y_test, y_pred):.2f}**")
        st.write(f"F1-Score: **{f1_score(y_test, y_pred):.2f}**")
        st.write(
            f"AUC-ROC: **{roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.2f}**"
        )
        st.write(
            f"PR-AUC: **{average_precision_score(y_test, model.predict_proba(X_test)[:,1]):.2f}**"
        )

        if metrics:
            plot_metrics(metrics, model, X_test, y_test, class_names)

        # SHAP for model explainability
        if classifier in ["Random Forest", "XGBoost", "Logistic Regression"]:
            explainer = (
                shap.TreeExplainer(model)
                if classifier in ["Random Forest", "XGBoost"]
                else shap.KernelExplainer(model.predict, X_train)
            )
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)

    # PCA Visualization
    if st.sidebar.checkbox("PCA Visualization", False):
        pca_visualization(X_train, y_train)


if __name__ == "__main__":
    main()
