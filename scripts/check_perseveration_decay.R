# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

library(rstan)

# load stanfit
load("data/intermediate_data/modeling/modeling_fits/stan_fit_m_19_d_lstm_a2c_nh_48_lr_0_0001_n_n_p_0_ew_lin_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_1_test_b_daw_p_1.RData")
# load data
load("data/intermediate_data/modeling/preprocessed_data_for_modeling/pp_data_lstm_a2c_nh_48_lr_0_0001_n_n_p_0_ew_lin_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_1_test_b_daw_p_1.RData")

# extract recency weighted perseveration values
mcmc = rstan::extract(stanfit$stanfit, pars = 'h')
h_arr = as.array(mcmc)$h
mean_h = apply(h_arr, c(2,3), mean)

# extract choices
my_cols = c('black', 'red', 'lightgreen', 'lightblue')

col_points <- c()
for (a in res$choices[1,]){ col_points <- c(col_points, my_cols[a])}

# plot
matplot(mean_h, type = "l")
points(x = c(1:res$nTrials), rep(1, res$nTrials), col = col_points)



