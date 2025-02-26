# 🚲 Bike Sharing App

This is my Bike Sharing app, built to explore and predict bike rentals in Seoul using a big datase of rental counts, weather conditions, and time details. It’s powered by Streamlit, so you get an interactive experience with data visuals and prediction tools right in your browser.

## What’s Inside?
The app’s got a bunch of features to help you understand bike rental patterns:
- **Raw Data & EDA**: See the first chunk of data, basic stats, and histograms for things like rental counts or temperature.
- **Distribution**: A violin plot shows how rentals spread out—quiet days vs. busy ones.
- **Outliers**: Spot crazy high rental days with a boxplot and adjustable z-score slider.
- **Correlations**: A heatmap reveals what factors (like temp or humidity) tie to rentals, with a top-5 list.
- **Trends**: Line graphs track hourly rentals, filterable by season.
- **Feature Importance**: For models like Random Forest, see which inputs (e.g., hour, temp) matter most.
- **Tuning**: Optuna tweaks Random Forest or XGBoost with 50 trials to optimize predictions.
- **Cross-Validation**: Tests model reliability with 5-fold RMSE scores.
- **Prediction**: Pick conditions (hour, temp, etc.) and get a rental guess, plus save/export options.

## The Model Scoop
You can choose from six models to predict rentals:
- **Linear Regression**: Simple and fast, good for straight-line trends.
- **Decision Tree**: Splits data into branches, catching basic patterns.
- **Random Forest**: Tons of trees averaging out guesses—great for messy data.
- **Gradient Boosting**: Builds trees step-by-step, refining predictions.
- **XGBoost**: A turbo-charged booster, often super accurate.
- **SVR**: Support Vector Regression, handy for tricky curves.

I added a “Use Log Target” option to smooth out wild rental spikes, which helps some models perform better. The app caches training to keep things snappy.

## Data Details
The dataset (`SeoulBikeData.csv`) comes from Seoul’s bike-sharing system—hourly records with stuff like temperature, humidity, wind speed, and rental counts. I engineered features like `Temp_Humidity` (temp × humidity) and `Hour_Temp` (hour × temp) to catch combo effects, plus split dates into day, month, and more for time-based insights.

## How to Run It
- **Locally**: Grab this folder, run `pip install -r requirements.txt` in a terminal here, then `streamlit run BikeSharing_app.py`.
- **Online**: It’s live on Streamlit Cloud [here](https://app-projects-3dxuaexk7h53lyqcjytnx2.streamlit.app/).

## A Little Background
I started this to mess around with some data I found—it’s from Seoul’s bike-sharing system. The app’s grown into a fun way to see what drives rentals and test out prediction models. Hope you enjoy poking around as much as I did making it!
