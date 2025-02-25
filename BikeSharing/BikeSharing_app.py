import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import optuna
from sklearn.model_selection import KFold, cross_val_score
import os

# Models (only import whats used)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("App initialized")

# UI setup
st.set_page_config(page_title="🚲 Seoul Bike Predictor", page_icon="🚲", layout="wide")
st.title("🚲 Seoul Bike Demand Predictor")
st.markdown("""
    <div style='text-align:center; padding:10px; background:#ecf0f1; border-radius:8px;'>
        <h3 style='color:#2c3e50;'>Predict Bike Rentals</h3>
        <p style='color:#7f8c8d;'>Analyze, model, and export results.</p>
    </div>
""", unsafe_allow_html=True)
st.write("""
    Hey there! This app’s about figuring out how many bikes get rented in Seoul. We got data on rentals, weather, and more, and we’ll dig in with cool charts, play with some models, and let ya predict rentals yerself. I’ll walk ya through it so it makes sense—let’s jump in!
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    file_path = "SeoulBikeData.csv"
    data_path = os.path.join(os.path.dirname(__file__), file_path)

    try:
        df = pd.read_csv(data_path, encoding="latin1")
        st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        logging.error(f"Load failed: {str(e)}")
    

# Data loading and preprocessing
st.write("### Gettin the Data Ready")
st.markdown("""
    **What’s goin on here?**  
    First we grab the bike rental data from a file—like openin a spreadsheet. Then we tidy it up to roll.

    **How we pullin it off?**  
    - Use pandas to load the csv into the app.
    - If somethin goes wrong—like the file’s missing—we’ll toss ya an error message.
    - We save it in memory with this @st.cache_data trick so it’s fast next time.

    **Why bother?**  
    - Raw data’s usually messy—wrong formats, missing bits. Cleaning it up makes it ready fer all the fun stuff later.
    - Plus nobody likes waitin around fer a slow load!
""")

@st.cache_data
def load_data(file_path):
    try:
        data_path = os.path.join(os.path.dirname(__file__), file_path)
        df = pd.read_csv(data_path, encoding="latin1")
        logging.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        st.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        logging.error(f"Load failed: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None
    
    # Check if ‘Date’ column exists
    if 'Date' not in df.columns:
        st.error("The csv’s missing the ‘Date’ column, which we need fer this app.")
        logging.error("Missing 'Date' column in csv file")
        return None

    # Date engineering
    st.write("#### Messing with Dates")
    st.markdown("""
        **What’s this about?**  
        The ‘Date’ column’s got stuff like ‘01/12/2017’, and we’re gonna break it into bits—like day, month, year—to make it useful.

        **How we doin it?**  
        - Turn them date strings into somethin the computer gets with pd.to_datetime.
        - Pull out day (like 1-31), month (1-12), year, and even weekday (0 fer Monday, up to 6 fer Sunday).
        - Add extras: ‘Is_Weekend’ (1 if Saturday or Sunday, 0 if not) and ‘Is_Holiday_Season’ (1 fer busy months like July or December).

        **Why’s it worth it?**  
        - Rentals change with time—weekends vs weekdays or summer vs winter. Splittin the date helps spot them trends.
        - Models can’t chew plain dates, so we give em numbers instead.
    """)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Weekday"] = df["Date"].dt.weekday
    df["Is_Weekend"] = df["Weekday"].isin([5, 6]).astype(int)
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Is_Holiday_Season"] = df["Month"].isin([7, 8, 12]).astype(int)
    
    # Categorical encoding
    st.write("#### Turnin Words Into Numbers")
    st.markdown("""
        **What’s the deal?**  
        Some columns—like ‘Holiday’ or ‘Seasons’—are words, not numbers. We gotta switch em fer the model to handle.

        **How we managin it?**  
        - Use pd.get_dummies to make new columns fer each option—like ‘Seasons_Spring’ gets 1 if it’s spring, 0 if not.
        - Drop one category (like ‘Seasons_Winter’) to keep it simple and avoid overlap.

        **Why do it?**  
        - Models don’t get text—they need numbers. This way they see if spring days mean more rentals.
        - It’s like givin the model a cheat sheet fer categories!
    """)
    return pd.get_dummies(df, columns=["Holiday", "Functioning Day", "Seasons"], drop_first=True)

data = load_data(file_path)
if data is None:
    st.stop()

# Feature engineering and split
st.write("### Cookin Up New Features and Splittin Data")
st.markdown("""
    **What’s this step?**  
    We’re makin some new columns from what we got and then splittin the data into two chunks—one fer trainin, one fer testin.

    **How we makin it happen?**  
    - **New Features**:  
      - `Temp_Humidity`: Multiply temp and humidity—maybe hot muggy days affect rentals different.
      - `Hour_Temp`: Mix hour and temp—could be rentals peak at certain warm hours.
      - `Log_Rented`: Take log of rental counts to smooth out big spikes.
    - **Split**:  
      - Chop data into 80% trainin and 20% testin with train_test_split.
      - Standardize everythin with StandardScaler so numbers play nice.

    **Why’s it cool?**  
    - New features catch patterns—like if rentals soar when it’s warm and humid together.
    - Log trick tones down crazy high rental days so model don’t freak out.
    - Splittin lets us train on most data and test how good we are on the rest—like a practice run before the real game.
""")
data["Temp_Humidity"] = data["Temperature(°C)"] * data["Humidity(%)"]
data["Hour_Temp"] = data["Hour"] * data["Temperature(°C)"]
data["Log_Rented"] = np.log1p(data["Rented Bike Count"])
X = data.drop(["Rented Bike Count", "Log_Rented", "Date"], axis=1)
y = data["Rented Bike Count"]
y_log = data["Log_Rented"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
_, _, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# Model training
st.write("### Teachin the Model")
st.markdown("""
    **What’s this about?**  
    Here’s where we teach a model to guess how many bikes get rented based on our data.

    **How we settings it up?**  
    - Ya pick a model from a dropdown—like Random Forest or Linear Regression.
    - There’s a checkbox to use log version of rentals if ya want.
    - Got a little function that grabs yer choice, trains it on data, and keeps it quick with a caching trick.

    **Why’s it neat?**  
    - Different models good at different things—Random Forest catches tricky patterns, Linear Regression keeps it simple.
    - Puttin this up top means all tabs use same model, and log option helps if rentals got wild swings.
""")
st.write("Global Model Settings (affects all tabs):")
model_choice = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", 
                                      "Gradient Boosting", "XGBoost", "SVR"], index=4, key="train_model",
                                      help="Pick yer fighter! Random Forest and XGBoost great fer messy data; Linear Regression’s straightforward.")
use_log = st.checkbox("Use Log Target", 
                     help="Tick this if ya wanna predict smoothed-out rentals—helps when there’s crazy high days.")

@st.cache_resource
def train_model(choice, X_train, y_train, log):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "SVR": SVR()
    }
    model = models[choice]
    model.fit(X_train, y_train_log if log else y_train)
    return model

model = train_model(model_choice, X_train, y_train, use_log)
y_pred = model.predict(X_test)
if use_log:
    y_pred = np.expm1(y_pred)

# Tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Raw Data and EDA", "📊 Distribution", "🔍 Outliers", "🌐 Corr", 
     "⏰ Trends", "Optuna Tuning", "Feature Importance", "Cross-Validation", "Model"]
)

# Tab 0: Raw Data Display
with tab0:
    st.subheader("Raw Data")
    st.markdown("""
        **What’s this tab?**  
        This is yer first peek at the data—like flippin open a book to see what’s inside.

        **How’s it laid out?**  
        - Show first 5 rows fer a quick look.
        - Toss in basic stats—like averages and ranges—fer the numbers.
        - Then we got histograms to see how everythin shakes out.

        **Why check it out?**  
        - Like meetin the data—ya see what yer dealin with, like how many rentals or what temps look like.
        - Histograms give ya a heads-up on patterns, like if rentals mostly low with a few big days.
    """)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')

    display_data = data.head()
    st.dataframe(display_data)
    st.markdown("**Data Summary:**")
    st.write(data.describe())
    st.markdown("**Feature Distributions (Histograms):**")
    st.write("""
        These histograms like a bar graph showin how often each number pops up—like how many days had 100 rentals or 20°C temps. Tall bars mean common stuff; long tails mean some wild ones.
    """)
    for col in data.select_dtypes(include=np.number).columns:
        fig = px.histogram(data, x=col, nbins=50, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Tab 1: Distribution
with tab1:
    st.subheader("Bike Rental Distribution")
    st.markdown("""
        **What’s up here?**  
        We’re zoomin in on how many bikes get rented—like, most days quiet or busy?

        **How we showin it?**  
        - Histogram with a fancy violin shape on top plots rental counts.
        - Little dropdown with extra stats if yer curious.

        **Why’s it handy?**  
        - Seein if rentals mostly low or got big spikes helps tweak the model—like maybe need that log trick to calm em down.
    """)
    fig = px.histogram(data, x="Rented Bike Count", nbins=50, marginal="violin", 
                      title="Rental Distribution", color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Stats"):
        stats_df = data["Rented Bike Count"].describe().to_frame().T
        st.write(stats_df.style.format("{:.2f}"))

# Tab 2: Outliers
with tab2:
    st.subheader("Outlier Detection")
    st.markdown("""
        **What’s an outlier?**  
        Think them crazy days—way more rentals than usual. We’re huntin em down here.

        **How we spottin it?**  
        - Figure how far each rental count is from average, in a stats way called z-scores.
        - Ya slide a bar to say how far’s “too far”—like 3 jumps from normal.
        - Boxplot shows regular range and dots fer weird ones.

        **Why care?**  
        - Them oddballs can throw off the model—like predictin too many rentals all the time. Knowin they’re there helps us decide what to do.
    """)
    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, 
                         help="Slide this to pick how wild a day’s gotta be to call it outlier—bigger number, less get flagged.")
    z_scores = np.abs((data["Rented Bike Count"] - data["Rented Bike Count"].mean()) / 
                     data["Rented Bike Count"].std())
    outliers = data[z_scores > threshold]
    st.write(f"Outliers (z > {threshold}): {len(outliers)}")
    
    fig, ax = plt.subplots()
    sns.boxplot(x=data["Rented Bike Count"], color='#ff7f0e', ax=ax)
    st.pyplot(fig)

# Tab 3: Correlations
with tab3:
    st.subheader("Feature Correlations")
    st.markdown("""
        **What’s this about?**  
        We’re checkin how much stuff like temp or hour ties into rentals—like, does warm weather mean more bikes?

        **How’s it done?**  
        - Crunch numbers to see how everythin lines up, from -1 (opposite) to 1 (perfect match).
        - Colorful heatmap shows it all, and we list top 5 that matter most to rentals.

        **Why’s it useful?**  
        - If temp’s a big deal, model should lean on it. If two things overlap a lot, maybe don’t need both—like pickin best players fer the team.
    """)
    corr = data.select_dtypes('number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    st.write("Top 5 Correlations with Rentals:")
    st.write(corr["Rented Bike Count"].sort_values(ascending=False)[1:6].to_frame().style.format("{:.2f}"))

# Tab 4: Hourly Trends
with tab4:
    st.subheader("Hourly Trends")
    st.markdown("""
        **What’s the scoop?**  
        This is where we see how rentals change thru the day—like mornin rush vs late night.

        **How we showin it?**  
        - Line graph tracks rentals by hour.
        - Ya can pick seasons to zoom in—like just Spring or Winter.

        **Why’s it cool?**  
        - Spottin when folks grab bikes (say 8 AM or 5 PM) helps model guess better, especially if seasons shake things up.
    """)
    season_cols = [col for col in data.columns if col.startswith("Seasons_")]
    season_names = [col.replace("Seasons_", "") for col in season_cols]
    season_filter = st.multiselect("Select Seasons", season_names, default=season_names, 
                                  help="Choose seasons to see how they change hourly vibe—like Spring might be busier!")
    
    filtered = data[data[[f"Seasons_{s}" for s in season_filter]].eq(1).any(axis=1)] if season_filter else data
    fig = px.line(filtered, x="Hour", y="Rented Bike Count", title="Rentals by Hour")
    st.plotly_chart(fig, use_container_width=True)

# Tab 5 : Optuna Hyperparameter Tuning
with tab5:
    st.subheader("Optuna Hyperparameter Tuning")
    st.markdown("""
        **What’s this fancy stuff?**  
        We’re finetunin the model—like tweakin a bike to ride smoother—fer best settings.

        **How we tweakin it?**  
        - Use Optuna to try 50 different combos—like how many trees or how deep they go.
        - Checks how close predictions are with rmse (lower’s better).
        - Shows ya the winnin combo.

        **Why mess with it?**  
        - Out-of-the-box settings might be okay, but tunin can make predictions sharper—especially if rentals tricky.
    """)
    model_choice_optuna = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", 
                                        "Gradient Boosting", "XGBoost", "SVR"], index=4, key="optuna_model",
                                        help="Pick one to tune—only Random Forest and XGBoost work here, they got knobs to twist!")
    
    def objective(trial):
        model_name = model_choice_optuna
        if model_name == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_name == "XGBoost":
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        else:
            return float('inf')  # Skip unsupported models
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    if st.button("Start Tuning"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        st.write("Best Parameters:", study.best_params)
        st.write("Best RMSE:", np.sqrt(study.best_value))
        st.write("Hey, just so ya know—this tweaks a fresh model. To use these settings everywhere, pick same model up top!")

# Tab 6: Feature Importance
with tab6:
    st.subheader("Feature Importance")
    st.markdown("""
        **What’s this tellin us?**  
        Fer Random Forest or XGBoost, we’re figuring out which things—like hour or temp—matter most fer guessin rentals.

        **How we seein it?**  
        - Model spits scores fer each feature—higher means it’s a bigger deal.
        - Throw em into sideways bar chart so ya see top dogs.

        **Why’s it a big deal?**  
        - If temp’s king, we know what drives rentals—like knowin which teammate’s carryin the team. Super helpful fer gettin the model!
    """)
    if model_choice in ["Random Forest", "XGBoost"]:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                         orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available fer this model.")

# Tab 7: Cross Validation Scores
with tab7:
    st.subheader("Cross Validation Scores")
    st.markdown("""
        **What’s this check?**  
        We’re testin the model’s skills by splittin data five ways—trainin on four parts, testin on one, and doin that five times.

        **How we runnin it?**  
        - Use trick called KFold to divvy it up.
        - Each split gets a rmse score—how off our guesses are—then average em.

        **Why’s it worth it?**  
        - One test might be a fluke—maybe got lucky or unlucky with split. This gives solid idea how model holds up overall.
    """)
    def cross_validate_model(model, X, y):
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        return rmse_scores

    if st.checkbox("Show Cross-Validation Scores"):
        scores = cross_validate_model(model, X_scaled, y)
        st.write("Cross-Validation RMSE Scores:", scores)
        st.write("Mean RMSE:", scores.mean())

# Tab 8: Modeling
with tab8:
    st.subheader("Modeling")
    st.markdown("""
        **What’s the plan here?**  
        This is where we see how good model is, make some guesses, and save everythin fer later.

        **How we rollin?**  
        - **Metrics**: Check MAE (average mistake), RMSE (bigger mistakes hurt more), and R² (how well it fits, 0 to 1).
        - **Prediction**: Plug in numbers—like hour or temp—and get a rental guess.
        - **Save/Export**: Keep model or grab csv of predictions.

        **Why’s this the fun part?**  
        - Metrics tell if model’s any good on data it ain’t seen.
        - Predictin’s like takin it for a spin yerself.
        - Savin means ya can come back anytime.
    """)
    mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}", help="Average slip-up—how far off we are on usual guess.")
    col2.metric("RMSE", f"{rmse:.2f}", help="Root mean squared error—big misses sting more here.")
    col3.metric("R²", f"{r2:.2f}", help="Fit score—1’s perfect, 0’s no good. How much rental vibe we nailed.")
    
    st.write("#### Prediction Form")
    st.markdown("""
        Play around with this! Toss in numbers—like what time or how hot—and see how many bikes model thinks’ll roll out.
    """)
    with st.form("predict"):
        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Hour", 0, 23, 12, help="What time of day? Maybe rush hour’s busier.")
            temp = st.slider("Temperature (°C)", -20.0, 40.0, 20.0, help="How warm’s it? Folks love bikin when it’s nice.")
            humidity = st.slider("Humidity (%)", 0, 100, 50, help="Sweaty weather might slow stuff down.")
        with col2:
            wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, help="Windy days could keep bikes parked.")
            season = st.selectbox("Season", season_names, help="Spring might mean more riders than Winter.")
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        input_data = pd.DataFrame({
            "Hour": [hour], "Temperature(°C)": [temp], "Humidity(%)": [humidity],
            "Wind speed (m/s)": [wind], "Temp_Humidity": [temp * humidity], "Hour_Temp": [hour * temp],
            **{f"Seasons_{s}": [1 if s == season else 0] for s in season_names}
        }, columns=X.columns).fillna(0)
        pred = np.expm1(model.predict(scaler.transform(input_data))[0]) if use_log else model.predict(scaler.transform(input_data))[0]
        st.success(f"Predicted Rentals: **{int(pred)}**")

    st.write("#### Save Model")
    st.markdown("Hit this to stash yer model—like puttin it on a shelf fer next time.")
    if st.button("Save Model"):
        joblib.dump(model, "bike_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model saved")

    st.write("#### Export Predictions")
    st.markdown("Grab a file with what we guessed vs what really happened—great fer checkin our work.")
    csv = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(index=False)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# Sidebar docs
with st.sidebar:
    st.subheader("📖 Guide")
    st.write("1. Set dataset path\n2. Check out tabs\n3. Play with model\n4. Save or export yer work")