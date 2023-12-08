from joblib import load
import pandas as pd

# Define model paths
regressor_model_path = 'extra_trees_regressor.joblib'
classifier_model_path = 'extra_trees_classifier.joblib'

# Load models
regressor_model = load(regressor_model_path)
classifier_model = load(classifier_model_path)

# Assuming 'df' is a DataFrame with the appropriate features - see example_dataset.csv
df = pd.read_csv('example_dataset.csv')

# Make the predictions
reaction_time_predictions = regressor_model.predict(df)
lapse_predictions = classifier_model.predict(df)

# Append predictions to each row
df['reaction_time_predictions'] = reaction_time_predictions
df['5_orMore_lapse_predictions'] = lapse_predictions

# Save the updated DataFrame
df.to_csv('example_dataset_with_predictions.csv', index=False)