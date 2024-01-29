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
# function loads rnn test files and saves an r object with formatted data
# for Stan models under path specified by path_to_save_formatted_data

# for explanations of arguments see comments in modeling_functions.R

preprocess_rnn_data_for_modeling(rnn_type = 'lstm2_a2c'
                                 , file_string = '%s_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_%s_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s'
                                 , num_instances = 30
                                 , train_sds = c('0_1')
                                 , sd_range = c(1,2,3)#seq(0.02, 0.32, 0.02)
                                 , path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling')


#####################
# Fit RNN data      #
#####################
# function takes preprocessed test files and saves an r object with
# posterior samples of stan models under path specified by path_to_save_results

# stan models to run (see legend in function definition in modeling_functions.R)
STAN_MODELS = c(1:6, 10:17)
# RNN instances to model
IDS = c(0:29)
# Walk instances
WALKS = c(1:3)
# preprocessed file name (%s as placeholders for id, walk)
FILE_NAME = 'pp_data_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s.RData'

for (m in STAN_MODELS){
  
  for (i in IDS){
    
    for (w in WALKS){
      
      file = sprintf(FILE_NAME, i,w)
    
      fit_model_to_rnn_data(stan_models = c(m), preprocessed_file_name = file,
                            num_instances = 1, 
                            sd_range = c(1),
                            subject_ids = 1, 
                            n_iter = 2000, 
                            path_to_save_results = 'data/intermediate_data/modeling/modeling_fits')
    }
    
  }
}



########################
# fit human data       #
########################

# preprocessed file name
file = 'pp_data_chakroun_placebo_human_bandit_data.RData'

# loads object with name res
load(sprintf('data/intermediate_data/modeling/preprocessed_data_for_modeling/%s', file))

# nRuns = num_instances
subject_ids = c(1:res$nRuns)

for (m in STAN_MODELS){
  
      fit_model_to_rnn_data(stan_models = c(m), preprocessed_file_name = file,
                            num_instances = 1, 
                            sd_range = c(1),
                            subject_ids = subject_ids,
                            n_iter = 2000,
                            path_to_save_results = 'data/intermediate_data/modeling/modeling_fits')
}







##### CUT
# preprocess_rnn_data_for_modeling(path_to_test_zip = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s.zip',
#                                  zip_file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_%s_res_rt_bin_p_%s_n_300_run_0.csv',
#                                  num_instances = 10,
#                                  sd_range = seq(0.02, 0.32, 0.02))





