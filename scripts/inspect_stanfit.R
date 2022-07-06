# script to inspect fitted objects

library(shinystan)

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../data/intermediate_data/modeling/modeling_fits')

# CONFIG

STANFIT = 'stan_fit_m_8_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_18_test_b_daw_p_1_id_1.RData'
load(STANFIT)


# launch shinystan
launch_shinystan(stanfit$stanfit)
