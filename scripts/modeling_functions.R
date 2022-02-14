#######################################
# Container for modeling scripts      #
#######################################

# load packages
library(rstan)
library(arrow)
library(stringr)
library(reticulate)
# use_condaenv(condaenv = 'rnn', required = TRUE)
# use_python("C:/Users/Deniz/anaconda3/envs/rnn/python.exe", required = TRUE)
# conda_install('pyarrow')
# py_config()

# stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

##########################################################################
# Script takes rnn test files and gives an r object with formatted data  #
# for Stan models                                                        #
##########################################################################

preprocess_rnn_data_for_modeling <- function(reward_type
                                             , rnn_type
                                             , is_noise
                                             , num_instances
                                             , train_sds
                                             , sd_range
                                             , path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'){

for (id_ in 0:(num_instances-1)){
      for (train_sd in train_sds){
          for (sd_ in sd_range){
            
            # convert decimal point to '_'
            sd_ = as.character(sd_)
            test_sd = gsub('[.]', '_', sd_)
            
            # load feathered python file in R
            is_noise = tolower(substring(is_noise, 1, 1))
            
            file_name = sprintf('%s_n_%s_rt_%s_train_sd_%s_id_%s_test_sd_%s',rnn_type, is_noise, reward_type, train_sd, id_, test_sd)
            df = arrow::read_feather(paste0(path_to_save_formatted_data,'/',file_name))
            
            ### format data for stan models
            
            # number of subjects = num_instances
            nRuns = length(unique(df$run))
            
            # number of trials, we divide by unique runs and unique reward instances (important for binary)
            nTrials = nrow(df)/(length(unique(df$run))*length(unique(df$reward_instance)))
            
            # get choices (add +1, because python indexes with 0)
            choices = df$choice+1
            
            
            # get rewards TODO check for binary vs continuous
            
            if (reward_type == 'continuous'){
              
              rewards = df[,c('p_rew_1', 'p_rew_2', 'p_rew_3', 'p_rew_4')]
              
              # get chosen rewards
              chosen_rewards = vector()
              
              for (row in c(1:nrow(rewards))){
                chosen_rewards = c(chosen_rewards, rewards[row, choices[row]])
              }
              
              chosen_rewards = unlist(chosen_rewards)
            }
            
            if (reward_type == 'binary'){
              
              rewards = df[,c('p_rew_1', 'p_rew_2', 'p_rew_3', 'p_rew_4')]
              
              # get chosen rewards
              chosen_rewards = vector()
              
              for (row in c(1:nrow(rewards))){
                chosen_rewards = c(chosen_rewards, rewards[row, choices[row]])
              }
              
              chosen_rewards = unlist(chosen_rewards)
            }
            
            # get choices in matrix[nRuns, nTrials]
            choices = matrix(choices, nrow = nRuns, ncol = nTrials, byrow = TRUE)
            
            # get chosen_rewards in matrix[nRuns, nTrials]
            chosen_rewards = matrix(chosen_rewards, nrow = nRuns, ncol = nTrials, byrow = TRUE)
            
            
            # get file name
            preprocessed_file_name_ = sprintf('%s_n_%s_rt_%s_train_sd_%s_id_%s_test_sd_%s', rnn_type, is_noise, reward_type, train_sd, id_, test_sd)
            
            # save formatted data
            res = list(model = preprocessed_file_name_, choices = choices, chosen_rewards = chosen_rewards, rewards = rewards
                       , nRuns = nRuns, nTrials = nTrials)
            # TODO: get name convention for preprocessed data
            save(file = sprintf('%s/pp_data_%s.RData', path_to_save_formatted_data, preprocessed_file_name_),res)
            
          }
        }
      }
}

  

##########################################################################
# Script takes preprocessed rnn test files and gives an r object with    #
# posterior samples of stan models                                       #
##########################################################################

# Input:
# stan_model: stan model to fit to data
# path_to_data: path_to_preprocessed_data
# Output: 
# save .RData object with stanfit object (posterior samples)

#--------------------------------------------------#
#               stan_model legend                  # 
#--------------------------------------------------#
# 1: ss_q_learning_model_seperate_lr.stan
# 2: ss_q_learning_model_single_lr.stan
# 3: BayesSM_ss_model.stan
# 4: BayesSMEP_ss_model.stan
# 5: BayesSME_ss_model.stan

fit_model_to_rnn_data <- function(stan_models # vector of integers according to stan_model legend
                                  , path_to_preprocessed_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'
                                  , preprocessed_file_name
                                  , path_to_save_results = 'data/intermediate_data/modeling/modeling_fits'
                                  , cognitive_model_directory = 'cognitive_models/'
                                  , num_instances
                                  , sd_range
){
  
  # inside R, source Python script
  # source_python("helpers.py")
  
# for (ins in c(3,4,5,6,7,9)){
for (ins in c(1)){
    for (sd_ in sd_range){
     
      # convert decimal point to '_'
      sd_ = as.character(sd_)
      sd_ = gsub('[.]', '_', sd_)
      
      # insert instance and sd
      preprocessed_file_name_ = sprintf(preprocessed_file_name, ins, sd_)
      
      # load preprocessed data
      full_preprocessed_file_path <- paste0(path_to_preprocessed_data,'/',preprocessed_file_name_)
      
      load(full_preprocessed_file_path)
      
      # get data into list
      choice <- res$choice
      reward <- res$chosen_rewards
      nTrials <- res$nTrials
      nSubjects <- res$nRuns
      my_data <- list(choice = choice, reward = reward, nTrials = nTrials, nSubjects = nSubjects)
      
      for (stan_model in stan_models){
        
        # get model_file and inits according to stan_model
        
        if(stan_model == 1){
          model_file = paste0(cognitive_model_directory, 'ms_q_learning_model_seperate_lr.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 10 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list(
          #     alpha_pos_rpe = 0.5,
          #     alpha_neg_rpe = 0.5,
          #     beta  = 1
          #   )}
        }
        
        if(stan_model == 2){
          model_file = paste0(cognitive_model_directory, 'ms_q_learning_model_single_lr.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 10 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list(
          #     alpha = 0.5,
          #     beta  = 1
          #   )}
        }
        
        if(stan_model == 3){
          model_file = paste0(cognitive_model_directory,'BayesSM_ms_model.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 1
          n_runs = 10 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list(
          #     beta  = 1
          #   )}
        }
        
        if(stan_model == 4){
          model_file = paste0(cognitive_model_directory,'BayesSMEP_ms_model.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 10 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list(
          #     beta  = 1,
          #     phi = 0,
          #     rho = 0
          #   )}
        }
        if(stan_model == 5){
          model_file = paste0(cognitive_model_directory,'BayesSME_ms_model.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 10 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list(
          #     beta  = 1,
          #     phi = 0
          #   )}
        }
        
        # posterior sampling
        my_samples <- sampling(my_stan_model
                               , my_data
                               , cores = 8
                               , chains = 1
                               , iter = 4000)
        
        # diagnostics (uncomment if needed)
        print(my_samples)
        # traceplot(my_samples, inc_warmup = TRUE)
        
        # get object name to be saved
        d = strsplit(full_preprocessed_file_path, 'pp_data_')[[1]][2]
        d = strsplit(d, '.RData')
        # t = strsplit(as.character(Sys.time()),' CEST')
        # t = gsub("-", "_", t)
        # t = gsub(":", "_", t)
        # t = gsub(" ", "_", t)
        result_name <- sprintf('stan_fit_m_%s_d_%s', stan_model, d)
        
        # get results list
        stanfit = list(stanfit = my_samples, data = full_preprocessed_file_path)
        
        # save result object (contains stan fit object + full_preprocessed_file_path)
        df <- as.data.frame(stanfit$stanfit)
        # save whole df as .RData
        save(df, file = sprintf('%s/%s.RData', path_to_save_results, result_name))
        # save only parameter columns as .feather (otherwise slow)
        
        df <- df[, 1:n_parameter_columns]
        write_feather(df, sprintf('%s/%s.feather', path_to_save_results, result_name))
        
        # delete my_samples
        rm(my_samples) 
        
      }
       
    }
    
  }
}



