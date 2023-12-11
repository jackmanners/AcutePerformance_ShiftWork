import pandas as pd
from joblib import load

# Load models
regressor_model = load('models/extra_trees_regressor.joblib')
classifier_model = load('models/extra_trees_classifier.joblib')

# Load input data
df = pd.read_csv('example_dataset.csv')

# Make predictions
reaction_time_predictions = regressor_model.predict(df)
lapse_predictions = classifier_model.predict(df)

# Append predictions to each row
df['reaction_time_predictions'] = reaction_time_predictions
df['5_orMore_lapse_predictions'] = lapse_predictions

# Save file to CSV, with predictions
df.to_csv('output/example_dataset_with_predictions.csv', index=False)