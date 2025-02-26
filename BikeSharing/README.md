# Bike Sharing App

Hey there! Welcome to my Bike Sharing app. This little project is all about diggin into bike rental data from Seoul think weather, time of day, and how many bikes people grab. I built it with Streamlit to make it interactive, so you can play around with charts, tweak models and even predict rentals yourself.

## What’s It Do?
- **Explore Data**: Check out raw number, see how rentals spread out, and spot weird outliers.
- **Visualize Trends**: Graphs show hourly patterns or how temperature ties to rentals.
- **Predict Stuff**: Pick a model (like Random Forest or XGBoost), tweak it with Optuna, and guess how many bikes will roll out based on conditions you set.

## How to Run It Locally
1. **Get the Files**: Grab this folder (`BikeSharing/`) with `BikeSharing_app.py`, `SeoulBikeData.csv`, and `requirements.txt`.
2. **Set Up**: Make sure you’ve got Python installed. Open a terminal in this folder and run: pip install -r requirements.txt
3. **Fire It Up**: Then just type:
streamlit run BikeSharing_app.py
It’ll pop open in your browser—easy peasy!

## Try It Online
I’ve got it live on Streamlit Cloud too! Check it out [here](https://<your-app-url>.streamlit.app) (replace with the real URL once it’s up). No setup needed—just click and play.

## A Little Background
I started this to mess around with some data I found it’s from Kaggle. The app’s grown into a fun way to see what drives rentals and test out prediction models. Hope you enjoy poking around as much as I did making it!
