# Explanations of files in the scripts folder

| File Name | Purpose|
|-----------|--------|
| posterior_median_plot.R | script creates plots of posterior predictive accuracy and median values of parameters and saves these values into a csv file for later use in JASP|
| modeling_call.R | script handles cognitive modeling (preprocessing of RNN test data and model fitting with stan)|
| modeling_functions.R | container for modeling helper functions|
| preprocess_human_data.R | script preprocesses human data for cognitive modeling|
| create_bandits.py | script that creates bandit tasks, which are saved in zip files to be later tested on trained RNN agents|
| model_comparison_plot.R | script plots cognitive model comparison with WAIC|
| create_Rdataframe_from_pickle.R | script converts multiple pickled dataframes to one RDataframe|
| bayesian_t_tests.R | script computes bayesian t-test based on the cumulative regret of agents and saves the effect size distribution |

