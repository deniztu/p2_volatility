# script to inspect fitted objects

library(shinystan)

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../data/intermediate_data/modeling/modeling_fits')

# CONFIG

STANFIT = 'stan_fit_m_2_d_rnn_n_f_rt_continuous_train_sd_meta_volatility_id_0_test_sd_0_32_t_2021_11_23_17_52_24.RData'
load(STANFIT)


# launch shinystan
launch_shinystan(stanfit$stanfit)
