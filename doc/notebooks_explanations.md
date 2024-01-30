# Explanations of files in the notebooks folder


| File        | Purpose           |
| ------------- |---------------| 
| access_tensorboard_data.ipynb     | notebook plots and analyzes loss functions during training based on files saved in the tensorboard folder |
| convert_rnn_data_to_csv.ipynb        | notebook converts multiple rnn test data (pickled pandas.Dataframes) to a single csv file           |
| multiplot.ipynb        | notebook merges single plots to a grid (multiplot)           |
| hidden_unit_analysis.ipynb        | notebook performs PCA and targeted dimensionality reduction on hidden unit activity of RNN agents and saves related plots (3D, scatter, stay/switch accuracy)           |
| bandit_recency_plot.ipynb        | notebook plots bandit recency plot (% switches as a function of interswitch unique bandits)           |
| delta_posterior_plot.ipynb        | notebook plots posterior distribution of the effect size delta based on a bayesian t-test regarding cumulative regret of agents           |
| performance_plot.ipynb        | notebook creates performance plots (reward walks and choices) of a single agent           |
| regret_plot.ipynb        | notebook creates a barplot of regret by agent type (RNN vs Human)           |
| screeplot.ipynb        | notebook creates a screeplot (cumulative variance explained by number of principal components)           |
| switch_plot.ipynb       | notebook creates a violin plot of proportion of switches by agent type (RNN vs Human)           |
| rank_switch_target_analysis.ipynb        | notebook creates plots for rank analysis of switch targets (reward rank, uncertainty rank)           |
| regret_barplots.ipynb       | notebook creates a barplot of regret values by RNN architectures           |

