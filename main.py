import os
from joblib import load
import pandas as pd
from tkinter import Tk, filedialog
import os

# Define model paths
regressor_model_path = 'extra_trees_regressor.joblib'
classifier_model_path = 'extra_trees_classifier.joblib'

# Load models
regressor_model = load(regressor_model_path)
classifier_model = load(classifier_model_path)

# Get current working directory
cwd = os.getcwd()

# Open file dialog to select CSV file
root = Tk()
root.withdraw()
csv_file_path = filedialog.askopenfilename(initialdir=cwd, filetypes=[('CSV Files', '*.csv')])

# Check if the user closed the dialog without selecting a file
if not csv_file_path:
    raise ValueError("No file selected.")
filename = os.path.splitext(csv_file_path.split('/')[-1])[0]

# Read the CSV file
df = pd.read_csv(csv_file_path)
required_columns = ['nb_rem_episodes', 'sleep_efficiency', 'total_sleep_time', 'remsleepduration', 
                    'lightsleepduration', 'deepsleepduration', 'waso', 'hr_average', 'hr_min', 
                    'hr_max', 'rr_average', 'sleep_mp_int', 'nremProp', 'sleep_score', 'timeSince']

# Check if the required columns are present in the DataFrame
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

df = df[required_columns]

# Make the predictions
reaction_time_predictions = regressor_model.predict(df)
lapse_predictions = classifier_model.predict(df)

# Append predictions to each row
df['reaction_time_predictions'] = reaction_time_predictions
df['5_orMore_lapse_predictions'] = lapse_predictions

# Save the updated DataFrame
savename = f'{filename}_with_predictions.csv'
df.to_csv(savename, index=False)

print(f"Saved locally at {cwd}/{savename}")