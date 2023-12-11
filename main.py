from ppredict import PerformancePredictor

# Create an instance of PerformancePredictor
pp = PerformancePredictor()

# Predict based on the pre-loaded CSV file
predicted_df = pp.predict(savefile=True)
predicted_expanded_df = pp.predict(expanded=True, savefile=True)

pp.plot_individual_data(
    identifier='participant',
    identities=['PX_2', 'PX_3'],
    plt_kwargs={'figsize': (16, 4), 'dpi': 150},
    show=False, save=True
)