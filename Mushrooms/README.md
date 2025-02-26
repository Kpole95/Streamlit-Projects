🍄 # Mushrooms App

This is my Mushrooms app, designed to classify mushrooms as edible or poisonous based on traits like cap shape and odor. It’s a Streamlit app, so it’s fully interactive think data visuals, model tuning, and predictions at your fingertips.

## What’s Inside?
Here’s what you can do with it:
- **Raw Data & EDA**: Peek at the first rows, stats, and bar charts for traits like cap color or gill size.
- **Distribution**: A bar chart shows how many mushrooms are edible vs. poisonous.
- **Correlations**: Heatmap and top-5 list reveal which traits link to edibility.
- **Feature Importance**: See what matters most with a bar chart for Random Forest or XGBoost.
- **Tuning**: Optuna runs 50 trials to fine-tune Random Forest or XGBoost for better accuracy.
- **Prediction**: Pick traits and predict edibility, with save/export options.

## The Model Lowdown
You’ve got three models to choose from:
- **Random Forest**: Builds a forest of decision trees, averaging their votes—solid for complex patterns.
- **SVM**: Finds the best line (or curve) to split edible from poisonous—good for clear divides.
- **XGBoost**: Boosts trees one by one, often nailing high accuracy with tricky data.

The app caches model training to speed things up. Since it’s classification (edible = 0, poisonous = 1), accuracy and detailed reports (precision, recall) tell us how it’s doing.

## Data Details
The dataset (`mushrooms.csv`) is a classic mushroom collection—think cap shape, odor, spore print color, and more, all as text that gets turned into numbers with `LabelEncoder`. The target’s the `class` column, and I use all other features to predict it. No extra engineering here—just pure mushroom traits!

## How to Run It
- **Locally**: Fork this folder, run `pip install -r requirements.txt` in a terminal here, then `streamlit run mushrooms_app.py`.
- **Online**: It’s up on Streamlit Cloud [here](https://app-projects-epbrznvgpqhrvyzbz5yappg.streamlit.app/)—swap in the real link when it’s live.

## Why I Made This
I stumbled across this mushroom dataset and thought, “Hey, can I teach a model to spot the bad ones?”.
