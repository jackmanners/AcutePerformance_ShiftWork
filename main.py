from calendar import c
import os
from joblib import load
import pandas as pd
from tkinter import Tk, filedialog

class PerformancePredictor:
    def __init__(self, regressor_model_path='extra_trees_regressor.joblib', classifier_model_path='extra_trees_classifier.joblib'):
        """
        Initialize the PerformancePredictor class.

        Args:
        - regressor_model_path: str, path to the regressor model file (default: 'extra_trees_regressor.joblib')
        - classifier_model_path: str, path to the classifier model file (default: 'extra_trees_classifier.joblib')
        """
        # Load models
        self.regressor_model = load(regressor_model_path)
        self.classifier_model = load(classifier_model_path)
        self.dataframe = None
        self.filepath = None
        self._required_columns = [
            'nb_rem_episodes', 'sleep_efficiency', 'total_sleep_time', 'remsleepduration',
            'lightsleepduration', 'deepsleepduration', 'waso', 'hr_average', 'hr_min',
            'hr_max', 'rr_average', 'sleep_mp_int', 'nremProp', 'sleep_score', 'timeSince']
        
    def _check_columns(self, df):
        """
        Check if the required columns are present in the DataFrame.

        Args:
        - df: pandas DataFrame, input DataFrame

        Raises:
        - ValueError: if any of the required columns are missing in the DataFrame
        """
        missing_columns = [col for col in self._required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

    def load_csv(self, csv_file_path) -> None:
        """
        Load and process the CSV file.

        Args:
        - csv_file_path: str, path to the CSV file
        """
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        self._check_columns(df)
        self.filepath = csv_file_path
        self.dataframe = df[self._required_columns]
        
    def predict(self, csv_file_path=None, savefile=False, savefolder=None) -> pd.DataFrame:
        """
        Make predictions based on the selected CSV file. Uses preloaded data if no file is selected.

        Args:
        - csv_file_path: str, path to the CSV file (default: None)
        - savefile: bool, whether to save the updated DataFrame (default: False)
        - savefolder: str, path to the folder where the updated DataFrame should be saved (default: None)

        Returns:
        - pd.DataFrame: DataFrame with predictions

        Raises:
        - ValueError: if no file is selected
        """
        if csv_file_path:
            # Read the CSV file
            self.load_csv(csv_file_path)

        # Filter the DataFrame to include only the required columns
        df = self.dataframe[self._required_columns]

        # Make the predictions
        reaction_time_predictions = self.regressor_model.predict(df)
        lapse_predictions = self.classifier_model.predict(df)

        # Append predictions to each row
        df['reaction_time_predictions'] = reaction_time_predictions
        df['5_orMore_lapse_predictions'] = lapse_predictions
        
        self.dataframe = df
        
        if savefile:            
            # Save the updated DataFrame
            filename = os.path.splitext(self.filepath.split('/')[-1])[0]
            savename = f'{filename}_with_predictions.csv'
            
            savepath = f'{savefolder}/{savename}' if savefolder else os.path.join(os.getcwd(), savename)
                
            df.to_csv(savepath, index=False)
            print(f"Saved locally at {savepath}")

        return df

# Example usage:
regressor_model_path = 'extra_trees_regressor.joblib'
classifier_model_path = 'extra_trees_classifier.joblib'

# Create an instance of PerformancePredictor
pp = PerformancePredictor(regressor_model_path, classifier_model_path)

# Open file dialog to select CSV file
root = Tk()
root.withdraw()
csv_file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

# Check if the user closed the dialog without selecting a file
if not csv_file_path:
    raise ValueError("No file selected.")

# Load and process the CSV file
pp.load_csv(csv_file_path)

# Predict based on the pre-loaded CSV file
predicted_df = pp.predict(savefile=True)

