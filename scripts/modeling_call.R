######################################
# script handles cognitive modeling  #
######################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# source functions
source('modeling_functions.R')

# move up two directories
setwd('../')

################################
# Preprocess binary test files #
################################

preprocess_rnn_data_for_modeling(rnn_type = 'rnn'
                                 , is_noise = FALSE
                                 , num_instances = 3
                                 , train_sds = c('meta_volatility')
                                 , sd_range = seq(0.02, 0.32, 0.02)
                                 , path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'
                                 , reward_type = 'continuous')


################################
# Model Preprocessed Data      #
################################

fit_model_to_rnn_data(stan_models = c(2), preprocessed_file_name = 'pp_data_rnn_n_f_rt_continuous_train_sd_meta_volatility_id_%s_test_sd_%s.RData',
                      num_instances = 10, 
                      sd_range = seq(0.02, 0.32, 0.02))






##### CUT
# preprocess_rnn_data_for_modeling(path_to_test_zip = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s.zip',
#                                  zip_file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s_res_rt_bin_p_%s_n_300_run_0.csv',
#                                  num_instances = 10,
#                                  sd_range = seq(0.02, 0.32, 0.02))






