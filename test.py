from huggingface_hub import hf_hub_url, hf_hub_download
from joblib import load
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

# Replace these with your Hugging Face repository ID and filename
REPO_ID = 'jackmanners/AcutePerformance_DaytimeSleep'
REGRESSOR_FILENAME = 'extra_trees_regressor.joblib'
CLASSIFIER_FILENAME = 'extra_trees_classifier.joblib'

# Download models
regressor_model_path = hf_hub_download(REPO_ID, REGRESSOR_FILENAME, token=token)
classifier_model_path = hf_hub_download(REPO_ID, CLASSIFIER_FILENAME, token=token)
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
