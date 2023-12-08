from joblib import load
import pandas as pd
from tkinter import Tk, filedialog

# Define model paths
regressor_model_path = 'extra_trees_regressor.joblib'
classifier_model_path = 'extra_trees_classifier.joblib'

# Load models
regressor_model = load(regressor_model_path)
classifier_model = load(classifier_model_path)

# Open file dialog to select CSV file
root = Tk()
root.withdraw()
csv_file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Make the predictions
reaction_time_predictions = regressor_model.predict(df)
lapse_predictions = classifier_model.predict(df)

# Append predictions to each row
df['reaction_time_predictions'] = reaction_time_predictions
df['5_orMore_lapse_predictions'] = lapse_predictions

# Save the updated DataFrame
df.to_csv('example_dataset_with_predictions.csv', index=False)