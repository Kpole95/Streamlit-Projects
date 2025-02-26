import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import io
import optuna
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("App initialized")

# UI setup
st.set_page_config(page_title="🍄 Mushroom Classifier", page_icon="🍄", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")
st.markdown("""
    <div style='text-align:center; padding:10px; background:#ecf0f1; border-radius:8px;'>
        <h3 style='color:#2c3e50;'>Classify Mushrooms</h3>
        <p style='color:#7f8c8d;'>Analyze, model, and predict edibility.</p>
    </div>
""", unsafe_allow_html=True)
st.write("Hey there! This app digs into the wild world of mushrooms—figuring out if they’re edible or poisonous based on features like cap shape, odor, and more. We’ll explore the data with cool charts, tweak some models, and let you predict edibility yourself. I’ll explain each step so it’s crystal clear—let’s jump in!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    file_path = "mushrooms.csv"
    data_path = os.path.join(os.path.dirname(__file__), file_path)
    try:
        df = pd.read_csv(data_path)
        st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        logging.error(f"Load failed: {str(e)}")
    st.subheader("📖 Guide")
    st.write("1. Load the data\n2. Explore the tabs\n3. Train the model\n4. Predict and save")

# Data loading and preprocessing
st.write("### Getting the Mushroom Data Ready")
st.markdown("""
    **What’s going on here?**  
    We’re grabbing mushroom data from a CSV file—like opening a field guide. Then we prep it for analysis.

    **How do we pull it off?**  
    - We use `pandas` to load the CSV into the app.
    - Since mushroom features are text (like 'bell' for cap shape), we turn them into numbers with `LabelEncoder`.
    - If anything goes wrong, we’ll flag it with an error message.

    **Why bother?**  
    - Raw data’s messy—text instead of numbers, potential missing bits. Cleaning it up lets us explore and model it.
    - Plus, fast loading keeps things smooth!
""")

@st.cache_data
def load_data(file_path):
    try:
        data_path = os.path.join(os.path.dirname(__file__), file_path)
        df = pd.read_csv(data_path)
        logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        st.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        logging.error(f"Load failed: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None
    
    st.write("#### Turning Mushroom Traits into Numbers")
    st.markdown("""
        **What’s this about?**  
        Mushroom data has words—like 'fishy' for odor or 'brown' for color. We need numbers for models.

        **How do we do it?**  
        - We use `LabelEncoder` to swap each unique word for a number (e.g., 'fishy' = 1, 'sweet' = 2).
        - This happens for every column since they’re all categorical.

        **Why’s it worth it?**  
        - Models only crunch numbers, not text. This lets us analyze and predict with traits like cap shape or spore color.
    """)
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    return df

data = load_data(file_path)
if data is None:
    st.stop()

# Feature engineering and split
st.write("### Splitting the Mushroom Data")
st.markdown("""
    **What’s this step?**  
    We’re dividing the data into training and testing sets to build and check our model.

    **How do we make it happen?**  
    - We split 80% for training (teaching the model) and 20% for testing (seeing how it does).
    - The target is 'class' (0 = edible, 1 = poisonous), and features are everything else.

    **Why’s it cool?**  
    - Training on most data gives the model a solid foundation, while testing on the rest shows how it handles new mushrooms.
""")
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
st.write("### Teaching the Mushroom Model")
st.markdown("""
    **What’s this all about?**  
    We’re training a model to guess if a mushroom is edible or poisonous based on its features.

    **How do we set it up?**  
    - Pick a model from a dropdown—like Random Forest or XGBoost.
    - A caching trick keeps it fast for repeated runs.

    **Why’s it neat?**  
    - Different models catch different patterns—Random Forest loves complex data, SVM might nail tricky splits. Pick what works best!
""")
st.write("Global Model Settings (affects all tabs):")
model_choice = st.selectbox("Model", ["Random Forest", "SVM", "XGBoost"], index=0, key="train_model",
                            help="Choose your classifier! Random Forest is great for lots of features; XGBoost boosts accuracy.")

@st.cache_resource
def train_model(choice, X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42)
    }
    model = models[choice]
    model.fit(X_train, y_train)
    return model

model = train_model(model_choice, X_train, y_train)
y_pred = model.predict(X_test)

# Tabs
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Raw Data and EDA", "📊 Distribution", "🌐 Corr", "⏰ Feature Importance", "Optuna Tuning", "Model"]
)

with tab0:
    st.subheader("Raw Data")
    st.markdown("""
        **What’s this tab?**  
        Your first look at the mushroom data—like flipping through a nature book.

        **How’s it laid out?**  
        - First 5 rows give a quick peek.
        - Basic stats show ranges and counts.
        - Bar charts reveal feature spreads.

        **Why check it out?**  
        - See what we’re working with—how many edible vs. poisonous, common traits. It’s like scouting the forest!
    """)
    display_data = data.head()
    st.dataframe(display_data)
    st.markdown("**Data Summary:**")
    st.write(data.describe())
    st.markdown("**Feature Distributions (Bar Charts):**")
    for col in data.columns:
        fig = px.histogram(data, x=col, title=f"{col} Distribution", nbins=len(data[col].unique()))
        st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.subheader("Class Distribution")
    st.markdown("""
        **What’s up here?**  
        We’re checking how many mushrooms are edible vs. poisonous.

        **How do we show it?**  
        - A bar chart plots the counts of each class.

        **Why’s it handy?**  
        - If it’s lopsided (lots of edible, few poisonous), the model might need tweaking to handle it.
    """)
    fig = px.histogram(data, x="class", title="Edible vs. Poisonous", labels={"class": "Class (0=Edible, 1=Poisonous)"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Feature Correlations")
    st.markdown("""
        **What’s this about?**  
        We’re seeing how mushroom traits relate to each other and edibility.

        **How’s it done?**  
        - A heatmap shows correlations from -1 (opposite) to 1 (same).
        - Top 5 links to 'class' are listed.

        **Why’s it useful?**  
        - If odor strongly ties to edibility, the model can lean on it. Spotting overlaps helps simplify things!
    """)
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    st.pyplot(fig)
    st.write("Top 5 Correlations with Class:")
    st.write(corr["class"].sort_values(ascending=False)[1:6].to_frame().style.format("{:.2f}"))

with tab3:
    st.subheader("Feature Importance")
    st.markdown("""
        **What’s this telling us?**  
        Which traits—like odor or gill size—matter most for predicting edibility?

        **How do we see it?**  
        - For Random Forest or XGBoost, we get scores for each feature and plot them in a bar chart.

        **Why’s it a big deal?**  
        - Knowing the top players (e.g., spore print color) helps us understand what drives the model’s guesses!
    """)
    if model_choice in ["Random Forest", "XGBoost"]:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model.")

with tab4:
    st.subheader("Optuna Hyperparameter Tuning")
    st.markdown("""
        **What’s this fancy stuff?**  
        We’re tweaking the model—like tuning a mushroom recipe—for top performance.

        **How do we tweak it?**  
        - Optuna tries 50 combos of settings (e.g., tree depth for Random Forest) to minimize error.
        - It spits out the best setup.

        **Why mess with it?**  
        - Default settings are okay, but tuning can boost accuracy—especially for tricky mushrooms!
    """)
    model_choice_optuna = st.selectbox("Model", ["Random Forest", "XGBoost"], index=0, key="optuna_model",
                                       help="Pick one to tune—Random Forest and XGBoost have lots of dials to twist!")
    
    def objective(trial):
        model_name = model_choice_optuna
        if model_name == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "XGBoost":
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        else:
            return float('inf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return 1 - accuracy_score(y_test, y_pred)

    if st.button("Start Tuning"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        st.write("Best Parameters:", study.best_params)
        st.write("Best Accuracy:", 1 - study.best_value)
        st.write("Heads up: This tunes a fresh model—apply these settings up top to use them everywhere!")

with tab5:
    st.subheader("Modeling")
    st.markdown("""
        **What’s the plan here?**  
        We’re seeing how well the model identifies edible vs. poisonous mushrooms and letting you test it.

        **How do we roll?**  
        - **Metrics**: Accuracy (how often it’s right) and a detailed report (precision, recall).
        - **Prediction**: Pick traits and guess edibility.
        - **Save/Export**: Save the model or download predictions.

        **Why’s this fun?**  
        - Metrics show if it’s trustworthy. Predicting feels like playing mushroom detective!
    """)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"])
    st.metric("Accuracy", f"{accuracy:.2f}", help="How often the model guesses right.")
    st.text("Classification Report:")
    st.text(report)

    st.write("#### Prediction Form")
    st.markdown("Try it out! Pick some mushroom traits and see if it’s safe to eat.")
    with st.form("predict"):
        col1, col2 = st.columns(2)
        with col1:
            cap_shape = st.selectbox("Cap Shape", data['cap-shape'].unique(), help="Shape of the cap—like 'bell' or 'flat'.")
            odor = st.selectbox("Odor", data['odor'].unique(), help="Smell—like 'almond' or 'foul'.")
        with col2:
            gill_color = st.selectbox("Gill Color", data['gill-color'].unique(), help="Color under the cap.")
            spore_print = st.selectbox("Spore Print Color", data['spore-print-color'].unique(), help="Color of spore dust.")
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        input_data = pd.DataFrame({
            "cap-shape": [cap_shape], "odor": [odor], "gill-color": [gill_color],
            "spore-print-color": [spore_print]
        }, columns=X.columns).fillna(0)
        pred = model.predict(input_data)[0]
        result = "Edible" if pred == 0 else "Poisonous"
        st.success(f"Predicted Edibility: **{result}**")

    st.write("#### Save Model")
    st.markdown("Save your model for later—like preserving a mushroom recipe!")
    if st.button("Save Model"):
        joblib.dump(model, "mushroom_model.pkl")
        st.success("Model saved")

    st.write("#### Export Predictions")
    st.markdown("Download what we guessed vs. reality—perfect for checking our work.")
    csv = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(index=False)
    st.download_button("Download Predictions", csv, "mushroom_predictions.csv", "text/csv")