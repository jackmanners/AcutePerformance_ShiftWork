from ppredict import PerformancePredictor

# Create an instance of PerformancePredictor
pp = PerformancePredictor('example_dataset.csv', expand=True)

# Predict based on the pre-loaded CSV file
predicted_df = pp.predict(savefile=True)
predicted_expanded_df = pp.predict(expanded=True, savefile=True)

pp.plot_individual_data(
    identifier='participant',
    identities=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    plt_kwargs={'figsize': (16, 4), 'dpi': 150},
    show=True
)