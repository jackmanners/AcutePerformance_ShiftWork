import os
import sys
from joblib import load
import pandas as pd
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# Suppressing version warning
warnings.filterwarnings('ignore', message='is_categorical_dtype is deprecated')

class PerformancePredictor:
    def __init__(self, filepath=None, expand=True,
                 regressor_model_path='models/extra_trees_regressor.joblib',
                 classifier_model_path='models/extra_trees_classifier.joblib'):
        """
        Initialize the PerformancePredictor class.

        Args:
        - regressor_model_path: str, path to the regressor model file (default: 'models/extra_trees_regressor.joblib')
        - classifier_model_path: str, path to the classifier model file (default: 'models/extra_trees_classifier.joblib')
        """
        self._required_columns = [
            'nb_rem_episodes', 'sleep_efficiency', 'total_sleep_time', 'remsleepduration',
            'lightsleepduration', 'deepsleepduration', 'waso', 'hr_average', 'hr_min',
            'hr_max', 'rr_average', 'sleep_mp_int', 'nremProp', 'sleep_score', 'timeSince']
        
        # Load models
        self.regressor_model = load(regressor_model_path)
        self.classifier_model = load(classifier_model_path)
        self.filepath = filepath
        self.dataframe = self.load_csv(filepath)
        self.expanded_dataframe = self.expand_data() if expand else None
        
        print(f"Data loaded{' and expanded' if expand else ''} from: {self.filepath}")
     
    def _check_columns(self, df) -> None:
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


    def load_csv(self, csv_file_path=None) -> None:
        """
        Load and process the CSV file.

        Args:
        - csv_file_path: str, path to the CSV file. If not provided, a file dialog will open to select the CSV file.
        """
        if not csv_file_path:
            # Open file dialog to select CSV file
            root = Tk()
            root.withdraw()
            csv_file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

        # Read the CSV file
        try: 
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            sys.stderr.write(f"File not found: '{csv_file_path}'\n")
            sys.exit(1)

        self._check_columns(df)
        self.filepath = csv_file_path

        return df
    
    
    def predict(self, expanded=False, savefile=True, savefolder='output') -> pd.DataFrame:
        """
        Make predictions based on the selected CSV file. Uses preloaded data if no file is selected.

        Args:
        - csv_file_path: str, path to the CSV file (default: None)
        - expanded: bool, whether to use expanded data (default: False)
        - savefile: bool, whether to save the updated DataFrame (default: False)
        - savefolder: str, path to the folder where the updated DataFrame should be saved (default: None)

        Returns:
        - pd.DataFrame: DataFrame with predictions

        Raises:
        - ValueError: if no file is selected

        Example usage:
        ```
        predictor = PerformancePredictor()
        predictor.predict(csv_file_path='data.csv', savefile=True, savefolder='results')
        ```
        """
        print(f"Predicting {'expanded ' if expanded else ''}input data...")
        df = self.expanded_dataframe if expanded else self.dataframe
        df_required = df[self._required_columns].copy()

        # Make the predictions
        reaction_time_predictions = self.regressor_model.predict(df_required)
        lapse_predictions = self.classifier_model.predict(df_required)

        # Append predictions to each row
        df['reaction_time_predictions'] = reaction_time_predictions
        df['5_orMore_lapse_predictions'] = lapse_predictions

        if savefile:            
            # Save the updated DataFrame
            filename = os.path.splitext(self.filepath.split('/')[-1])[0]
            savename = f'{filename}_with_predictions{f"_expanded" if expanded else ""}.csv'
                
            savepath = f'{savefolder}/{savename}' if savefolder else savename
                    
            df.to_csv(savepath, index=False)
            print(f"Saved {'expanded ' if expanded else ''}data locally at {savepath}")

        return df
    
    
    def expand_data(self):
        """
        Expand the data for each row, where all values are held constant except timeSince is repeated from 200-800 with an interval of 10.
        """
        df = self.dataframe.copy()
        
        # Drop duplicate rows based on all columns except timeSince
        df.drop_duplicates(subset=df.columns.difference(['timeSince']), inplace=True)
        
        expanded_rows = []
        for time in range(300, 790, 10):
            expanded_row = df.copy()
            expanded_row['timeSince'] = time
            expanded_rows.append(expanded_row)
        
        expanded_df = pd.concat(expanded_rows, ignore_index=True)
        self.expanded_dataframe = expanded_df
        
        return expanded_df
            
    def plot_individual_data(self, identifier=0, identities=None, show=False, save=True, plt_kwargs={'figsize': (16, 4), 'dpi': 150}):
        """
        Plot each individual's data with the x-axis being df['timeSince'] and the y-axis being 'reaction_time_predictions',
        with the hue being the 'participant' column. Include the '5_orMore_lapse_predictions' column as well.

        Args:
            identifier (str or int): The column name or index to filter the data. The default is the index of the first column.
            identities (list): A list of values to filter the data. The default is None (all).
            show (bool): Whether to show the plot. The default is False.
            save (bool): Whether to save the plot. The default is True.
            plt_kwargs (dict): Additional keyword arguments to be passed to the plot function. E.g. {'figsize': (8, 10), 'dpi': 300}.

        Returns:
            None
        """
        plot_print = 'all participants (NOT RECOMMENDED)' if not identities else identities
        print(f"Plotting {plot_print}...")
        
        assert isinstance(identifier, (str, int)), "Identifier must be a string column name or an integer column index."
        
        if self.expanded_dataframe is None:
            expanded_df = self.predict(expanded=True)
        else: 
            expanded_df = self.expanded_dataframe.copy()

        if isinstance(identifier, str):
            if identities is None:
                df = expanded_df.copy()
            else:
                df = expanded_df[expanded_df[identifier].isin(identities)].copy()
        elif isinstance(identifier, int):
            if identities is None:
                df = expanded_df.iloc[:, identifier].copy()
            else:
                df = expanded_df[expanded_df.iloc[:, identifier].isin(identities)].copy()

        # Convert identifier column values to strings if necessary
        if not df[identifier].dtype == 'object':
            df.loc[:, identifier] = df.loc[:, identifier].astype(str)

        plt.figure(**plt_kwargs)
        sns.scatterplot(data=df, x='timeSince', y='reaction_time_predictions', hue=identifier, style='5_orMore_lapse_predictions', s=20, palette='deep')
        plt.xlabel('Time Since Wake')
        plt.ylabel('Predicted Reaction Time (ms)')

        # Set the subtitle
        plt.title(
            "Performance Predictions\n(Dot: <5 Predicted Lapses, Cross: <5 Predicted Lapses)",
            fontsize=12,
            color="black",  # Change the color to black
            fontdict={'verticalalignment': 'baseline', 'fontsize': 10}
        )
        
        handles, labels = plt.gca().get_legend_handles_labels()
        
        plt.legend(
            handles=handles[1::],
            title="Identities",  # Set a title for the legend
            title_fontsize=12,  # Set the legend title size
            fontsize=10,  # Set the fontsize of the legend labels
            frameon=False,  # Disable the legend border
            labels=identities,
        )
        
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        
        if save: 
            savepath = f'output/prediction_plot.png'
            plt.savefig(savepath, dpi=300)
            print(f"Saved plot at {savepath}")
        if show: plt.show()
        
