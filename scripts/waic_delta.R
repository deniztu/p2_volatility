###########################################
# script does model comparison with WAIC  #
###########################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(loo)

# CONFIG

path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
file_string = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.RData'



model_names = c("1LR","1LRP","1LRPU", "1LRPT", "1LRU", "1LRT",
                "1LRDP","1LRDPU","1LRDPT")



# 1: ms_ql_1lr.stan
# 2: ms_ql_1lr_p.stan
# 3: ms_ql_1lr_p_u.stan
# 4: ms_ql_1lr_p_t.stan
# 5: ms_ql_1lr_u.stan
# 6: ms_ql_1lr_t.stan
# 7: ms_ql_1lr_dp.stan
# 8: ms_ql_1lr_dp_u.stan
# 9: ms_ql_1lr_dp_t.stan

model_ids = c(1:9)
rnn_ids = c(0:19)
walk = 3



# get waic's

waic_matrix = matrix(NA, nrow = length(rnn_ids), ncol = length(model_ids))

for (m in model_ids){
  
  for (i in rnn_ids){
    
    file = sprintf(file_string ,m, i,walk)
    
    load(paste0(path_to_stanfit, file))
    
    ll = extract_log_lik(stanfit$stanfit, parameter_name = "log_lik", merge_chains = TRUE)
    
    waic_object = loo::waic(ll)
    
    waic_matrix[i+1, m] = waic_object$estimates['waic','Estimate']

    }
}

# get delta waic's

for (row in c(1:nrow(waic_matrix))){
  m_waic = min(waic_matrix[row,])
  
  waic_matrix[row,] = waic_matrix[row,] - m_waic

}

# plot

boxplot(waic_matrix, ylab = 'Delta(WAIC)', names = model_names)

# best model
apply(waic_matrix,2,median)
