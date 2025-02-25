import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import io
import optuna
from sklearn.model_selection import KFold, cross_val_score
import os

# Models (only import what's used)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Logging setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("App initialized")

# UI setup
st.set_page_config(page_title="🍄 Mushroom Classifier", page_icon="🍄", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")
st.markdown("""
    <div style='text-align:center; padding:10px; background:#ecf0f1; border-radius:8px;'>
        <h3 style='color:#2c3e50;'>Classify Mushrooms</h3>
        <p style='color:#7f8c8d;'>Analize, model, and predict edibility.</p>
    </div>
""", unsafe_allow_html=True)
st.write("""
    Hey there! This app’s all about figuring out if mushrooms are safe to eat or not. We got data on stuff like cap shape and smell, and we’ll mess around with charts, tweak some models, and let ya predict yerself. I’ll walk ya thru it so it makes sence—lets dive in!
""")

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

# Data loading and preprocessing
st.write("### Getting the mushroom data ready")
st.markdown("""
    **Whats this about?**  
    Were grabbin the mushroom data from a file—like openin a book bout em. Then we fix it up fer analysis.

    **How we doin it?**  
    - Using pandas to load that csv into the app.
    - Mushrooms got words like ‘fishy’ fer smell—we turn em into numbers with labelencoder.
    - If somethin messes up, we’l tell ya with a error.

    **Why bother tho?**  
    - Raw datas a mess—words not numbers, maybe missin stuff. Cleanin it up let’s us play with it and model it
    - Plus nobody wants to wait fer a slow load!
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
    
    # Encode categorical features
    st.write("#### Turnin Mushroom Traits Into Numbers")
    st.markdown("""
        **Whats this bout?**  
        Mushroom stuffs words—like ‘fishy’ smell or ‘brown’ color. We gotta make em numbers fer models.

        **How we doin it?**  
        - Use labelencoder to swap words fer numbers—like ‘fishy’ gets 1, ‘sweet’ gets 2.
        - Doin this fer every column since its all words.

        **Why’s it worth it?**  
        - Models don’t get words—they need numbers. This way we can check how cap shape or smell helps predict edibility
    """)
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    
    return df

data = load_data(file_path)
if data is None:
    st.stop()

# Feature engineering and split
st.write("### Splittin the mushroom data")
st.markdown("""
    **Whats this step?**  
    Were choppin the data into trainin and testin bits to bild and check our model.

    **How we makin it happen?**  
    - 80% fer trainin—teachin the model—and 20% fer testin—seein how it does.
    - Targets ‘class’ (0 fer edible, 1 fer poison), rest is features.

    **Why’s it cool tho?**  
    - Trainin on most data gives the model a good base, testin on the rest shows how it handels new shrooms
""")
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
st.write("### Teachin the mushroom model")
st.markdown("""
    **Whats this all bout?**  
    Were trainin a model to guess if a shroom’s edible or poison from its bits.

    **How we settin it up?**  
    - Pick a model from a dropdown—like random forest or xgboost
    - Cache trick keeps it quick fer reruns.

    **Why’s it neat?**  
    - Differnt models catch differnt things—random forest’s good with messy data, svm might split tricky ones. Pick what ya like!
""")
st.write("Global Model Settings (affects all tabs):")
model_choice = st.selectbox("Model", ["Random Forest", "SVM", "XGBoost"], index=0, key="train_model",
                            help="Chose yer classifier! Random forest’s great fer lotsa features, xgboost ups the accuracy.")

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

# Tab 0: Raw Data Display
with tab0:
    st.subheader("Raw Data")
    st.markdown("""
        **Whats this tab?**  
        Yer first peek at the shroom data—like flippin thru a nature book.

        **How’s it laid out?**  
        - First 5 rows fer a quick look
        - Basic stats fer ranges n counts
        - Bar charts show how features spread

        **Why check it?**  
        - See what were dealin with—how many edibles vs poison, common traits. Like scoutin the woods!
    """)
    display_data = data.head()
    st.dataframe(display_data)
    st.markdown("**Data Summry:**")
    st.write(data.describe())
    st.markdown("**Feature Distributions (bar charts):**")
    for col in data.columns:
        fig = px.histogram(data, x=col, title=f"{col} Distribution", nbins=len(data[col].unique()))
        st.plotly_chart(fig, use_container_width=True)

# Tab 1: Distribution
with tab1:
    st.subheader("Class Distribution")
    st.markdown("""
        **Whats up here?**  
        Were checkin how many shrooms are edible vs poison.

        **How we showin it?**  
        - Bar chart counts each class

        **Why’s it handy?**  
        - If its lopsided—tons edible, few poison—the model might need a tweak to catch em
    """)
    fig = px.histogram(data, x="class", title="Edible vs. Poisonous", labels={"class": "Class (0=Edible, 1=Poisonous)"})
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Correlations
with tab2:
    st.subheader("Feature Corellations")
    st.markdown("""
        **Whats this bout?**  
        Were seein how shroom traits tie to eachother and edibility

        **How’s it done?**  
        - Heatmap shows corellations from -1 (opposite) to 1 (same)
        - Top 5 links to ‘class’ listed

        **Why’s it usefull?**  
        - If odor’s big with edibility, model can use it. Spottin overlaps keeps it simple!
    """)
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    st.pyplot(fig)
    st.write("Top 5 Corelaltions with Class:")
    st.write(corr["class"].sort_values(ascending=False)[1:6].to_frame().style.format("{:.2f}"))

# Tab 3: Feature Importance
with tab3:
    st.subheader("Feature Imporance")
    st.markdown("""
        **Whats this tellin us?**  
        Which traits—like smell or gill size—matter most fer guessin edibility?

        **How we seein it?**  
        - Fer random forest or xgboost, we get scores fer each trait n plot em in a bar chart

        **Why’s it a big deal?**  
        - Knowin top dogs (like spore color) shows what drives the model’s guesses!
    """)
    if model_choice in ["Random Forest", "XGBoost"]:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Imporance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature imporance not availble fer this model.")

# Tab 4: Optuna Hyperparameter Tuning
with tab4:
    st.subheader("Optuna Hyperparamter Tuning")
    st.markdown("""
        **Whats this fancy stuff?**  
        Were tweakin the model—like tunin a shroom recipe—fer best results

        **How we tweakn it?**  
        - Optuna tries 50 combos of settins (like tree depth fer random forest) to cut error
        - Spits out the best setup

        **Why mess with it?**  
        - Default’s ok, but tunin can make it way sharper—espeshally fer tricky shrooms!
    """)
    model_choice_optuna = st.selectbox("Model", ["Random Forest", "XGBoost"], index=0, key="optuna_model",
                                       help="Pick one to tune—random forest n xgboost got lotsa dials to twist!")
    
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
        return 1 - accuracy_score(y_test, y_pred)  # Minimize error (1 - accuracy)

    if st.button("Start Tunin"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        st.write("Best Paramters:", study.best_params)
        st.write("Best Accurasy:", 1 - study.best_value)
        st.write("Heads up: this tunes a fresh model—put these settins up top to use em everywhere!")

# Tab 5: Modeling
with tab5:
    st.subheader("Modelin")
    st.markdown("""
        **Whats the plan here?**  
        Were seein how good the model spots edible vs poison shrooms n lettin ya test it

        **How we rollin?**  
        - **Metrics**: Accurasy (how often its right) n a detaild report (precison, recall)
        - **Predictin**: Pick traits n guess edibility
        - **Save/Export**: Keep the model or grab a csv of guesses

        **Why’s this fun?**  
        - Metrics tell if its trustable. Predictin’s like playin shroom detective!
    """)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"])
    st.metric("Accurasy", f"{accuracy:.2f}", help="How often the model guesses right.")
    st.text("Classifcation Report:")
    st.text(report)

    st.write("#### Predictin Form")
    st.markdown("""
        Try it out! Pick some shroom traits n see if its safe to eat
    """)
    with st.form("predict"):
        col1, col2 = st.columns(2)
        with col1:
            cap_shape = st.selectbox("Cap Shape", data['cap-shape'].unique(), help="Shape of cap—like ‘bell’ or ‘flat’.")
            odor = st.selectbox("Odor", data['odor'].unique(), help="Smell—like ‘almond’ or ‘foul’.")
        with col2:
            gill_color = st.selectbox("Gill Color", data['gill-color'].unique(), help="Color under cap.")
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
    st.markdown("Save yer model fer later—like preservin a shroom recipe!")
    if st.button("Save Model"):
        joblib.dump(model, "mushroom_model.pkl")
        st.success("Model saved")

    st.write("#### Export Predictins")
    st.markdown("Download what we guessed vs reality—great fer checkin our work")
    csv = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(index=False)
    st.download_button("Download Predictins", csv, "mushroom_predictions.csv", "text/csv")

# Sidebar docs
with st.sidebar:
    st.subheader("📖 Guide")
    st.write("1. Load the data\n2. Check the tabs\n3. Train the model\n4. Predict n save")