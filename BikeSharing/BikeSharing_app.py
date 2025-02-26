import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import optuna
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
st.write("Hey there! This app’s all about figuring out how many bikes get rented in Seoul. We’ve got data on rentals, weather, and more, and we’ll dig into it with cool charts, tweak some fancy models, and even let you predict rentals yourself. I’ll walk you through each part so it all makes sense—let’s dive in!")

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
    st.subheader("📖 Guide")
    st.write("1. Set dataset path\n2. Check out the tabs\n3. Play with the model\n4. Save or export your work")

# Data loading and preprocessing
st.write("### Getting the Data Ready")
st.markdown("""
    **What’s going on here?**  
    First, we grab the bike rental data from a file—like opening a spreadsheet. Then we tidy it up so it’s ready to roll.

    **How do we pull it off?**  
    - We use a tool called `pandas` to load the CSV file into the app.
    - If something goes wrong (like the file’s missing), we’ll let you know with an error message.
    - We also save it in memory with a caching trick so it loads fast next time.

    **Why bother?**  
    - Raw data’s usually a mess—wrong formats, missing bits. Cleaning it up makes it usable for all the fun stuff we’ll do later.
    - Plus, nobody likes waiting around for a slow load!
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
    
    if 'Date' not in df.columns:
        st.error("The CSV file is missing the 'Date' column, which is required for this app.")
        logging.error("Missing 'Date' column in CSV file")
        return None

    st.write("#### Messing with Dates")
    st.markdown("""
        **What’s this about?**  
        The 'Date' column’s got stuff like '01/12/2017', and we’re gonna break it into pieces—like day, month, year—to make it more useful.

        **How do we do it?**  
        - We turn those date strings into something the computer gets with `pd.to_datetime`.
        - Then we pull out the day (like 1-31), month (1-12), year, and even the weekday (0 for Monday, up to 6 for Sunday).
        - We add a couple extras: 'Is_Weekend' (1 if it’s Saturday or Sunday, 0 if not) and 'Is_Holiday_Season' (1 for busy months like July or December).

        **Why’s it worth it?**  
        - Bike rentals change with time—think weekends versus weekdays or summer versus winter. Splitting the date helps us spot those trends.
        - Models can’t chew on plain dates, so we give them numbers instead.
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
    
    st.write("#### Turning Words into Numbers")
    st.markdown("""
        **What’s the deal?**  
        Some columns—like 'Holiday' or 'Seasons'—are words, not numbers. We’ve gotta switch them to something the model can handle.

        **How do we manage that?**  
        - We use `pd.get_dummies` to make new columns for each option—like 'Seasons_Spring' gets a 1 if it’s spring, 0 if not.
        - We drop one category (like 'Seasons_Winter') to keep things simple and avoid overlap.

        **Why do it?**  
        - Models don’t get text—they need numbers. This way, they can see if, say, spring days mean more rentals.
        - It’s like giving the model a cheat sheet for categories!
    """)
    return pd.get_dummies(df, columns=["Holiday", "Functioning Day", "Seasons"], drop_first=True)

data = load_data(file_path)
if data is None:
    st.stop()

# Feature engineering and split
st.write("### Cooking Up New Features and Splitting the Data")
st.markdown("""
    **What’s this step?**  
    We’re making some new columns from what we’ve got and then splitting the data into two chunks—one for training, one for testing.

    **How do we make it happen?**  
    - **New Features**:  
      - `Temp_Humidity`: Multiply temperature and humidity—maybe hot, muggy days affect rentals differently.
      - `Hour_Temp`: Mix hour and temperature—could be rentals peak at certain warm hours.
      - `Log_Rented`: Take the log of rental counts to smooth out big spikes.
    - **Split**:  
      - Chop the data into 80% training and 20% testing with `train_test_split`.
      - Standardize everything with `StandardScaler` so all numbers play nice together.

    **Why’s it cool?**  
    - New features can catch patterns—like if rentals soar when it’s warm and humid together.
    - The log trick tones down crazy high rental days so the model doesn’t freak out.
    - Splitting lets us train on most of the data and test how good we are on the rest—like a practice run before the real game.
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
st.write("### Teaching the Model")
st.markdown("""
    **What’s this all about?**  
    Here’s where we teach a model to guess how many bikes get rented based on our data.

    **How do we set it up?**  
    - You pick a model from a dropdown—like Random Forest or Linear Regression.
    - There’s a checkbox to use the log version of rentals if you want.
    - We’ve got a little function that grabs your choice, trains it on the data, and keeps it quick with a caching trick.

    **Why’s it neat?**  
    - Different models are good at different things—Random Forest might catch tricky patterns, while Linear Regression keeps it simple.
    - Putting this up top means all the tabs use the same model, and the log option can help if rentals have wild swings.
""")
st.write("Global Model Settings (affects all tabs):")
model_choice = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", 
                                      "Gradient Boosting", "XGBoost", "SVR"], index=4, key="train_model",
                                      help="Pick your fighter! Random Forest and XGBoost are great for messy data; Linear Regression is straightforward.")
use_log = st.checkbox("Use Log Target", help="Tick this if you want to predict the smoothed-out version of rentals—it helps when there’s a few crazy high days.")

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

with tab0:
    st.subheader("Raw Data")
    st.markdown("""
        **What’s this tab?**  
        This is your first peek at the data—like flipping open the book to see what’s inside.

        **How’s it laid out?**  
        - We show the first 5 rows so you get a quick look.
        - Toss in some basic stats—like averages and ranges—for the numbers.
        - Then we’ve got histograms to see how everything shakes out.

        **Why check it out?**  
        - It’s like meeting the data—you see what you’re dealing with, like how many rentals or what the temps look like.
        - Histograms give you a heads-up on patterns, like if rentals are mostly low with a few big days.
    """)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    display_data = data.head()
    st.dataframe(display_data)
    st.markdown("**Data Summary:**")
    st.write(data.describe())
    st.markdown("**Feature Distributions (Histograms):**")
    st.write("These histograms are like a bar graph showing how often each number shows up—like how many days had 100 rentals or 20°C temps. Tall bars mean common values; long tails mean some wild ones.")
    for col in data.select_dtypes(include=np.number).columns:
        fig = px.histogram(data, x=col, nbins=50, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.subheader("Bike Rental Distribution")
    st.markdown("""
        **What’s up here?**  
        We’re zooming in on how many bikes get rented—like, are most days quiet or busy?

        **How do we show it?**  
        - A histogram with a fancy violin shape on top plots the rental counts.
        - There’s a little dropdown with extra stats if you’re curious.

        **Why’s it handy?**  
        - Seeing if rentals are mostly low or have big spikes helps us tweak the model—like maybe we need that log trick to calm things down.
    """)
    fig = px.histogram(data, x="Rented Bike Count", nbins=50, marginal="violin", 
                       title="Rental Distribution", color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Stats"):
        stats_df = data["Rented Bike Count"].describe().to_frame().T
        st.write(stats_df.style.format("{:.2f}"))

with tab2:
    st.subheader("Outlier Detection")
    st.markdown("""
        **What’s an outlier?**  
        Think of those crazy days—way more rentals than usual. We’re hunting those down here.

        **How do we spot them?**  
        - We figure out how far each rental count is from the average, in a stats way called z-scores.
        - You slide a bar to decide how far is “too far”—like 3 standard jumps from normal.
        - A boxplot shows the regular range and dots for the weird ones.

        **Why care?**  
        - Those oddballs can throw off the model—like predicting too many rentals all the time. Knowing they’re there helps us decide what to do about them.
    """)
    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, 
                          help="Slide this to pick how wild a day has to be to call it an outlier—bigger number, fewer get flagged.")
    z_scores = np.abs((data["Rented Bike Count"] - data["Rented Bike Count"].mean()) / 
                      data["Rented Bike Count"].std())
    outliers = data[z_scores > threshold]
    st.write(f"Outliers (z > {threshold}): {len(outliers)}")
    fig, ax = plt.subplots()
    sns.boxplot(x=data["Rented Bike Count"], color='#ff7f0e', ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Feature Correlations")
    st.markdown("""
        **What’s this about?**  
        We’re checking how much things like temperature or hour tie into rentals—like, does warm weather mean more bikes?

        **How’s it done?**  
        - We crunch numbers to see how everything lines up, from -1 (opposite) to 1 (perfect match).
        - A colorful heatmap shows it all, and we list the top 5 that matter most to rentals.

        **Why’s it useful?**  
        - If temperature’s a big deal, the model should lean on it. If two things overlap a lot, maybe we don’t need both—it’s like picking the best players for the team.
    """)
    corr = data.select_dtypes('number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', ax=ax)
    st.pyplot(fig)
    st.write("Top 5 Correlations with Rentals:")
    st.write(corr["Rented Bike Count"].sort_values(ascending=False)[1:6].to_frame().style.format("{:.2f}"))

with tab4:
    st.subheader("Hourly Trends")
    st.markdown("""
        **What’s the scoop?**  
        This is where we see how rentals change through the day—like morning rush versus late night.

        **How do we show it?**  
        - A line graph tracks rentals by hour.
        - You can pick seasons to zoom in—like just Spring or Winter.

        **Why’s it cool?**  
        - Spotting when people grab bikes (say, 8 AM or 5 PM) helps the model guess better, especially if seasons shake things up.
    """)
    season_cols = [col for col in data.columns if col.startswith("Seasons_")]
    season_names = [col.replace("Seasons_", "") for col in season_cols]
    season_filter = st.multiselect("Select Seasons", season_names, default=season_names, 
                                   help="Choose seasons to see how they change the hourly vibe—like Spring might be busier!")
    filtered = data[data[[f"Seasons_{s}" for s in season_filter]].eq(1).any(axis=1)] if season_filter else data
    fig = px.line(filtered, x="Hour", y="Rented Bike Count", title="Rentals by Hour")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Optuna Hyperparameter Tuning")
    st.markdown("""
        **What’s this fancy stuff?**  
        We’re fine-tuning the model—like tweaking a bike to ride smoother—finding the best settings for it.

        **How do we tweak it?**  
        - We use Optuna to try 50 different combos for Random Forest or XGBoost—like how many trees or how deep they go.
        - It checks how close predictions are with RMSE (lower’s better).
        - Then we show you the winning combo.

        **Why mess with it?**  
        - Out-of-the-box settings might be okay, but tuning can make predictions way sharper.
        - Heads up: This tunes a separate model—it doesn’t change the main one unless you match them up yourself.
    """)
    model_choice_optuna = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", 
                                        "Gradient Boosting", "XGBoost", "SVR"], index=4, key="optuna_model",
                                        help="Pick one to tune—only Random Forest and XGBoost work here, they’ve got the knobs we can twist!")
    
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
            return float('inf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    if st.button("Start Tuning"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        st.write("Best Parameters:", study.best_params)
        st.write("Best RMSE:", np.sqrt(study.best_value))
        st.write("Hey, just so you know—this tweaks a fresh model here. To use these settings everywhere, pick the same model up top!")

with tab6:
    st.subheader("Feature Importance")
    st.markdown("""
        **What’s this telling us?**  
        For Random Forest or XGBoost, we’re figuring out which things—like hour or temp—matter most for guessing rentals.

        **How do we see it?**  
        - The model spits out scores for each feature—higher means it’s a bigger deal.
        - We throw those into a sideways bar chart so you can see the top dogs.

        **Why’s it a big deal?**  
        - If temperature’s king, we know what drives rentals. It’s like knowing which teammate’s carrying the team—super helpful for understanding the model!
    """)
    if model_choice in ["Random Forest", "XGBoost"]:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model.")

with tab7:
    st.subheader("Cross Validation Scores")
    st.markdown("""
        **What’s this check?**  
        We’re testing the model’s skills by splitting the data five ways—training on four parts, testing on one, and doing that five times.

        **How do we run it?**  
        - We use a trick called `KFold` to divvy it up.
        - For each split, we get an RMSE score—how off our guesses are—and then average them.

        **Why’s it worth doing?**  
        - One test might be a fluke—maybe we got lucky or unlucky with the split. This gives us a solid idea of how the model holds up overall.
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

with tab8:
    st.subheader("Modeling")
    st.markdown("""
        **What’s the plan here?**  
        This is where we see how good the model is, make some predictions, and save everything for later.

        **How do we roll?**  
        - **Metrics**: We check MAE (average mistake), RMSE (bigger mistakes hurt more), and R² (how well it fits, 0 to 1).
        - **Prediction**: You plug in numbers—like hour or temp—and get a rental guess.
        - **Save/Export**: Keep the model or grab a CSV of predictions.

        **Why’s this the fun part?**  
        - Metrics tell us if the model’s any good on data it hasn’t seen.
        - Predicting’s like taking it for a spin yourself.
        - Saving means you can come back to it anytime.
    """)
    mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}", help="Average slip-up—how far off we are on a typical guess.")
    col2.metric("RMSE", f"{rmse:.2f}", help="Root mean squared error—big misses sting more here.")
    col3.metric("R²", f"{r2:.2f}", help="Fit score—1’s perfect, 0’s no good. How much of the rental vibe we nailed.")
    
    st.write("#### Prediction Form")
    st.markdown("""
        Play around with this! Toss in some numbers—like what time or how hot it is—and see how many bikes the model thinks will roll out.
    """)
    with st.form("predict"):
        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Hour", 0, 23, 12, help="What time of day? Maybe rush hour’s busier.")
            temp = st.slider("Temperature (°C)", -20.0, 40.0, 20.0, help="How warm is it? People love biking when it’s nice out.")
            humidity = st.slider("Humidity (%)", 0, 100, 50, help="Sweaty weather might slow things down.")
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
    st.markdown("Hit this to stash your model—like putting it on a shelf for next time.")
    if st.button("Save Model"):
        joblib.dump(model, "bike_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model saved")

    st.write("#### Export Predictions")
    st.markdown("Grab a file with what we guessed versus what really happened—great for checking our work.")
    csv = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(index=False)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")