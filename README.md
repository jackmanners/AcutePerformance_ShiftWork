# Performance Estimation Models - Jack Manners

This repository hosts performance estimation models and execution code developed by Jack Manners. 

These models were constructed using scikit-learn, based on a dataset from approximately 24 individuals who participated in a simulated shift work study. In this study, participants were required to advance their sleep schedule 12 hours, followed by nighttime "shift-work" containing an array of performance tasks.

**Note:** The models were trained on data collected from an under-mattress sleep sensor, not Polysomnography (PSG).

The models are trained on and designed to predict performance on the Psychomotor Vigilance Task (PVT). Specifically, they estimate the mean reaction time under 500ms (non-lapse reaction time), and whether there would be 5 or more lapses (reaction time > 500ms) at a given time point, chosen as a marker of significant performance decline as a result of cumulative sleep deprivation.

**Note:** These models were trained on shifts running from midnight-8AM, with a mandated out of bed time of 7PM. However, the actual wake time detected by the sensor may have differed.

The models in this repository have been packaged into the `ppredict.PerformancePredictor` class to assist with data collation, data-checking and simple plotting features. However, an example of the most basic usage can be found in `ppredict/basic_usage.py`, if this is preferable.

## Installation

1. Clone the repository from GitHub:

    ```sh
    git clone https://github.com/jackmanners/AcutePerformance_ShiftWork.git
    ```

2. Navigate into the cloned repository:

    ```sh
    cd AcutePerformance_ShiftWork
    ```

3. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the model, execute the main script:

```sh
python main.py
```
This will open a file dialogue for you to select the data to be used for predictions.

Currently, the data **must** include the column names found in the `example_dataset.csv`. The exception is the 'participant' column, but identifier values are expected in the first column for plotting (see below).

The `main.py` script will, by default, make predictions on any data within the input file. The input file can also be specified in-code. The main script will also expand each record in order to make predictions for roughly the timeframe that the model has been trained on (5 - 13 hours post-wake). Data is, by default, saved to the 'output' folder, though an alternative folder can be specified using `savefolder`.

```python
pp = PerformancePredictor() # Opens a file dialogue
pp = PerformancePredictor(filepath='example_dataset.csv') # Directly opens specified file

predicted_df = pp.predict(savefile=False) # Returns the dataframe with predictions. Does not save to file.
predicted_expanded_df = pp.predict(expanded=True, savefolder='prediction_folder') # Returns the dataframe with predictions for all available timepoints. Saves .csv to folder 'prediction_folder/'.
```

### Plotting

The main script also includes some simple plotting, integrated useing the [seaborn](https://seaborn.pydata.org/) package.
This will run by default, but some additional arguments have been included in the `main.py` script.

```python
pp.plot_individual_data(
    identifier='participant', # Identifier column (e.g., participant ID, sleep ID, etc.). Defaults to the first column.
    identities=['PX_2', 'PX_3'], # Specific identities (based on identifier column) to plot, defaults to all. (All not recommended without adjusting plot parameters). Must be a list, but can be a list of one (e.g., ['PX_1']).
    plt_kwargs={'figsize': (16, 4), 'dpi': 150}, # Keywords arguments passed to plt.figure(**plt_kwargs).
    show=False, # Whether to show the plot
    save=True # Whether to save the plot (to 'output' folder)
)
``````
