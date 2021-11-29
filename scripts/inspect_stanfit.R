# script to inspect fitted objects

library(shinystan)

# CONFIG

STANFIT = 'stan_fit_m_2_d_rnn_n_f_rt_continuous_train_sd_meta_volatility_id_0_test_sd_0_32_t_2021_11_23_17_52_24.RData'



# launch shinystan
launch_shinystan(STANFIT)
