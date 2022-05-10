######################################
# script handles cognitive modeling  #
######################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# source functions
source('modeling_functions.R')

# move up two directories
setwd('../')

#########################
# Preprocess test files #
#########################

preprocess_rnn_data_for_modeling(rnn_type = 'lstm_a2c'
                                 , file_string = '%s_nh_48_lr_0_0001_n_n_p_0_ew_lin_vw_0_5_dr_0_5_res_d_f_p_%s_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s'
                                 , is_noise = FALSE
                                 , num_instances = 5
                                 , train_sds = c('0_1')
                                 , sd_range = c(1,2,3)#seq(0.02, 0.32, 0.02)
                                 , path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'
                                 , reward_type = '')


################################
# Model Preprocessed Data      #
################################

fit_model_to_rnn_data(stan_models = c(1,2,3,4,5), preprocessed_file_name = 'pp_data_rnn_n_f_rt_continuous_train_sd_meta_volatility_id_%s_test_sd_%s.RData',
                      num_instances = 10,
                      sd_range = seq(0.02, 0.32, 0.02))

fit_model_to_rnn_data(stan_models = c(2), preprocessed_file_name = 'pp_data_rnn_n_f_rt_continuous_train_sd_.1_id_%s_test_sd_%s.RData',
                      num_instances = 10, 
                      sd_range = seq(0.1, 0.1, 0.02))

fit_model_to_rnn_data(stan_models = c(2), preprocessed_file_name = 'pp_data_rnn_n_f_rt_continuous_train_sd_.05_id_%s_test_sd_%s.RData',
                      num_instances = 10, 
                      sd_range = seq(0.02, 0.32, 0.02))

fit_model_to_rnn_data(stan_models = c(2), preprocessed_file_name = 'pp_data_lstm_ac_continuous_n_f_rt_continuous_train_sd_meta_volatility_id_%s_test_sd_%s.RData',
                      num_instances = 10, 
                      sd_range = c(0.32))

### testing model for lstm ew 0.05

#debugonce(fit_model_to_rnn_data)
for (i in c(1:4)){
  
  for (m in c(2,6,7)){
    
    for (w in c(1,2,3)){
      
      file = sprintf('pp_data_lstm_a2c_nh_48_lr_0_0001_n_n_p_0_ew_0_05_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s.RData', i,w)
      
      fit_model_to_rnn_data(stan_models = c(m), preprocessed_file_name = file,
                            num_instances = 1, 
                            sd_range = c(1))
      
    }
    
  }
  
}





##### CUT
# preprocess_rnn_data_for_modeling(path_to_test_zip = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s.zip',
#                                  zip_file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s_res_rt_bin_p_%s_n_300_run_0.csv',
#                                  num_instances = 10,
#                                  sd_range = seq(0.02, 0.32, 0.02))








