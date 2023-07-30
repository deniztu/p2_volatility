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
                                             , file_string # added
                                             , is_noise
                                             , num_instances
                                             , train_sds
                                             , sd_range
                                             , path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'){
# for (id_ in c(0)){
for (id_ in c(0:(num_instances-1))){
      for (train_sd in train_sds){
          for (sd_ in sd_range){
            
            # convert decimal point to '_'
            sd_ = as.character(sd_)
            test_sd = gsub('[.]', '_', sd_)
            
            # load feathered python file in R
            is_noise = tolower(substring(is_noise, 1, 1))
            # is_noise = 'n' #added
            
            file_name = sprintf(file_string, rnn_type, train_sd, id_, test_sd)
            
            df = arrow::read_feather(paste0(path_to_save_formatted_data,'/',file_name))
            
            ### format data for stan models
            
            # number of subjects = num_instances
            nRuns = length(unique(df$run))
            
            # number of trials, we divide by unique runs and unique reward instances (important for binary)
            nTrials = nrow(df)/(length(unique(df$run))*length(unique(df$reward_instance)))
            
            # get choices (add +1, because python indexes with 0)
            choices = df$choice+1
            
            
            # get rewards TODO check for binary vs continuous
            
            if (reward_type == ''){ # changed
              
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
            preprocessed_file_name_ = file_name
            
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
# Function calculates the bandit heuristic predictor                     #
# returns a matrix with number of unique Bandits sampled between switches#
##########################################################################

get_bandit_heuristic_predictor = function(choices_of_run){
  
  # calculate matrix 
  result_matrix = matrix(0, length(choices_of_run), 4)
  
  a1 = which(choices_of_run == 1)
  a2 = which(choices_of_run == 2)
  a3 = which(choices_of_run == 3)
  a4 = which(choices_of_run == 4)
  
  b1 = rep(0, length(choices_of_run))
  b2 = rep(0, length(choices_of_run))
  b3 = rep(0, length(choices_of_run))
  b4 = rep(0, length(choices_of_run))
  
  
  # 1 to first occurence
  
  b1[1:(a1[1])] = (cumsum(as.numeric(!duplicated(choices_of_run[1:(a1[1])]))))
  b2[1:(a2[1])] = (cumsum(as.numeric(!duplicated(choices_of_run[1:(a2[1])]))))
  b3[1:(a3[1])] = (cumsum(as.numeric(!duplicated(choices_of_run[1:(a3[1])]))))
  b4[1:(a4[1])] = (cumsum(as.numeric(!duplicated(choices_of_run[1:(a4[1])]))))
  
  # occurence[i] to occurence[i+1]
  
  for(i in 2:length(a1)){
    b1[(a1[i-1]+1):(a1[i]-1)]<-cumsum(as.numeric(!duplicated(choices_of_run[(a1[i-1]+1):(a1[i]-1)])))
  }
  
  for(i in 2:length(a2)){
    b2[(a2[i-1]+1):(a2[i]-1)]<-cumsum(as.numeric(!duplicated(choices_of_run[(a2[i-1]+1):(a2[i]-1)])))
  }
  
  for(i in 2:length(a3)){
    b3[(a3[i-1]+1):(a3[i]-1)]<-cumsum(as.numeric(!duplicated(choices_of_run[(a3[i-1]+1):(a3[i]-1)])))
  }
  
  for(i in 2:length(a4)){
    b4[(a4[i-1]+1):(a4[i]-1)]<-cumsum(as.numeric(!duplicated(choices_of_run[(a4[i-1]+1):(a4[i]-1)])))
  }
  
  # last occurence to last trial
  if(a1[length(a1)]<length(choices_of_run)){
    b1[(a1[length(a1)]+1):(length(choices_of_run))]<-cumsum(as.numeric(!duplicated(choices_of_run[(a1[length(a1)]+1):(length(choices_of_run))])))
  }
  
  if(a2[length(a2)]<length(choices_of_run)){
    b2[(a2[length(a2)]+1):(length(choices_of_run))]<-cumsum(as.numeric(!duplicated(choices_of_run[(a2[length(a2)]+1):(length(choices_of_run))])))
  }
  
  if(a3[length(a3)]<length(choices_of_run)){
    b3[(a3[length(a3)]+1):(length(choices_of_run))]<-cumsum(as.numeric(!duplicated(choices_of_run[(a3[length(a3)]+1):(length(choices_of_run))])))
  }
  
  if(a4[length(a4)]<length(choices_of_run)){
    b4[(a4[length(a4)]+1):(length(choices_of_run))]<-cumsum(as.numeric(!duplicated(choices_of_run[(a4[length(a4)]+1):(length(choices_of_run))])))
  }
  
  # chosen options are 0
  b1[a1]<-0
  b2[a2]<-0
  b3[a3]<-0
  b4[a4]<-0
  
  # diagnostics
  cbind(choices_of_run, b1) # 1 is okay
  cbind(choices_of_run, b2) # 2 is okay
  cbind(choices_of_run, b3) # 3 is okay
  cbind(choices_of_run, b4) # 4 is okay
  
  # insert to result matrix
  result_matrix[,1] = b1
  result_matrix[,2] = b2
  result_matrix[,3] = b3
  result_matrix[,4] = b4
  
  return(result_matrix)
  
}

###############################################################################
# Function calculates the bandit trials not chosen predictor                  #
# returns a array with number of trials Bandits not sampled consecutive times #
###############################################################################

get_trials_not_chosen <- function(pp_file = res){
  
  NS=pp_file$nSubjects
  NT=pp_file$nTrials
  choices<-pp_file$choice
  trials_not_chosen<-array(0,c(NS,NT,4))
  for (s in 1:NS) {
    for (t in 1:NT) {
      if (choices[s,t]!=0){
        trials_not_chosen[s,t,1]<-ifelse(choices[s,t]==1, 0, t)
        if (trials_not_chosen[s,t,1]!=0 & max(which(trials_not_chosen[s,1:t,1]==0))>-Inf){
          trials_not_chosen[s,t,1]<-(t-max(which(trials_not_chosen[s,1:t,1]==0)))
        }
        trials_not_chosen[s,t,2]<-ifelse(choices[s,t]==2, 0, t)
        if (trials_not_chosen[s,t,2]!=0 & max(which(trials_not_chosen[s,1:t,2]==0))>-Inf){
          trials_not_chosen[s,t,2]<-(t-max(which(trials_not_chosen[s,1:t,2]==0)))
        }
        trials_not_chosen[s,t,3]<-ifelse(choices[s,t]==3, 0, t)
        if (trials_not_chosen[s,t,3]!=0 & max(which(trials_not_chosen[s,1:t,3]==0))>-Inf){
          trials_not_chosen[s,t,3]<-(t-max(which(trials_not_chosen[s,1:t,3]==0)))
        }
        trials_not_chosen[s,t,4]<-ifelse(choices[s,t]==4, 0, t)
        if (trials_not_chosen[s,t,4]!=0 & max(which(trials_not_chosen[s,1:t,4]==0))>-Inf){
          trials_not_chosen[s,t,4]<-(t-max(which(trials_not_chosen[s,1:t,4]==0)))
        }
      }
    }
  }
  
  return(trials_not_chosen)
  
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
# 1: ms_ql_1lr.stan
# 2: ms_ql_1lr_p.stan
# 3: ms_ql_1lr_p_u.stan
# 4: ms_ql_1lr_p_t.stan
# 5: ms_ql_1lr_u.stan
# 6: ms_ql_1lr_t.stan
# 7: ms_ql_1lr_dp.stan
# 8: ms_ql_1lr_dp_u.stan
# 9: ms_ql_1lr_dp_t.stan

# 10: ms_kalman_model.stan
# 11: ms_kalman_model_e.stan
# 12: ms_kalman_model_t.stan
# 13: ms_kalman_model_u.stan
# 14: ms_kalman_model_p.stan
# 15: ms_kalman_model_ep.stan
# 16: ms_kalman_model_tp.stan
# 17: ms_kalman_model_up.stan



fit_model_to_rnn_data <- function(stan_models # vector of integers according to stan_model legend
                                  , path_to_preprocessed_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'
                                  , preprocessed_file_name
                                  , path_to_save_results = 'data/intermediate_data/modeling/modeling_fits'
                                  , cognitive_model_directory = 'cognitive_models/'
                                  , num_instances
                                  , sd_range
                                  , subject_ids # which subjects/runs to model
                                  , n_iter = 4000 # number of iterations
                                  , n_chains = 2 # number of chains
){
  
  # inside R, source Python script
  # source_python("helpers.py")
  
for (ins in c(num_instances-1)){
  for (subject_id in subject_ids){
    for (sd_ in sd_range){
     
      # convert decimal point to '_'
      sd_ = as.character(sd_)
      sd_ = gsub('[.]', '_', sd_)
      
      # insert instance and sd
      # preprocessed_file_name_ = sprintf(preprocessed_file_name, ins, sd_)
      # file_name = 'pp_data_test_lstm_a2c_ew_0_05.RData'
      preprocessed_file_name_ = preprocessed_file_name
      
      
      # load preprocessed data
      full_preprocessed_file_path <- paste0(path_to_preprocessed_data,'/',preprocessed_file_name_)
      
      load(full_preprocessed_file_path)
      
      # get data into list
      
      choice <- t(as.matrix(res$choice[subject_id,]))
      reward <- t(as.matrix(res$chosen_rewards[subject_id,]))
      nTrials <- res$nTrials
      nSubjects <- 1 # always single subject
      my_data <- list(choice = choice, reward = reward, nTrials = nTrials, nSubjects = nSubjects)
      
      # rescale reward to 1-100 for RNNs trained on 0-1
      if (max(my_data$reward) < 1){
        my_data$reward = my_data$reward*100
      }
      
      for (stan_model in stan_models){
        
        # get model_file and inits according to stan_model
        
        if(stan_model == 1){
          model_file = paste0(cognitive_model_directory, 'ms_ql_1lr.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = nSubjects 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 2){
          model_file = paste0(cognitive_model_directory, 'ms_ql_1lr_p.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1 
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 3){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_p_u.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 4){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_p_t.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1 
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
          
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
          # my_inits <- function(){
          #   list()}
        }
        if(stan_model == 5){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_u.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 6){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_t.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
        
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 7){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_dp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        if(stan_model == 8){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_dp_u.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 5
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 9){
          model_file = paste0(cognitive_model_directory,'ms_ql_1lr_dp_t.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 5
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
          
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 10){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 1
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()
        }
        
        if(stan_model == 11){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_e.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()
        }
        
        if(stan_model == 12){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_t.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
          
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
        }
        
        if(stan_model == 13){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_u.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()
        }
        
        if(stan_model == 14){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_p.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 2
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 15){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_ep.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 16){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_tp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
          
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 17){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_up.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 18){
          model_file = paste0(cognitive_model_directory, 'ms_kalman_model_dp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 3
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        if(stan_model == 19){
          model_file = paste0(cognitive_model_directory, 'ms_kalman_model_u_dp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          uni = array(0, c(n_runs, nTrials, 4))
          
          for (s in 1:n_runs){
            uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
          }
          
          # append to data
          my_data$uni = uni
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 20){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_t_dp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # create bandit heuristic predictor
          trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
          
          # append to data
          my_data$trials_not_chosen = trials_not_chosen
          
          # my_inits <- function(){
          #   list()}
        }
        
        if(stan_model == 21){
          model_file = paste0(cognitive_model_directory,'ms_kalman_model_e_dp.stan')
          
          # make stan model a global variable
          my_stan_model <<- stan_model(model_file)
          
          n_parameters = 4
          n_runs = 1
          n_parameter_columns = n_parameters * n_runs
          
          # my_inits <- function(){
          #   list()}
        }
        
        # if(stan_model == 18){
        #   model_file = paste0(cognitive_model_directory,'ms_kalman_model_up.stan')
        #   
        #   # make stan model a global variable
        #   my_stan_model <<- stan_model(model_file)
        #   
        #   n_parameters = 3
        #   n_runs = 1
        #   n_parameter_columns = n_parameters * n_runs
        #   
        #   # create bandit heuristic predictor
        #   uni = array(0, c(n_runs, nTrials, 4))
        #   
        #   for (s in 1:n_runs){
        #     uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
        #   }
        #   
        #   # append to data
        #   my_data$uni = uni
        #   
        #   # my_inits <- function(){
        #   #   list()}
        # }
        
        # if(stan_model == 19){
        #   model_file = paste0(cognitive_model_directory,'ms_kalman_model_tp.stan')
        #   
        #   # make stan model a global variable
        #   my_stan_model <<- stan_model(model_file)
        #   
        #   n_parameters = 3
        #   n_runs = 1
        #   n_parameter_columns = n_parameters * n_runs
        #   
        #   # create bandit heuristic predictor
        #   
        #   trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
        #   
        #   # append to data
        #   my_data$trials_not_chosen = trials_not_chosen
        #   
        #   # my_inits <- function(){
        #   #   list()}
        # }
        # 
        # if(stan_model == 20){
        #   model_file = paste0(cognitive_model_directory,'ms_kalman_model_u.stan')
        #   
        #   # make stan model a global variable
        #   my_stan_model <<- stan_model(model_file)
        #   
        #   n_parameters = 2
        #   n_runs = 1
        #   n_parameter_columns = n_parameters * n_runs
        #   
        #   # create bandit heuristic predictor
        #   uni = array(0, c(n_runs, nTrials, 4))
        #   
        #   for (s in 1:n_runs){
        #     uni[s,,] = get_bandit_heuristic_predictor(choice[s,])
        #   }
        #   
        #   # append to data
        #   my_data$uni = uni
        #   
        #   # my_inits <- function(){
        #   #   list()}
        # }
        # 
        # if(stan_model == 21){
        #   model_file = paste0(cognitive_model_directory,'ms_kalman_model_t.stan')
        #   
        #   # make stan model a global variable
        #   my_stan_model <<- stan_model(model_file)
        #   
        #   n_parameters = 2
        #   n_runs = 1
        #   n_parameter_columns = n_parameters * n_runs
        #   
        #   # create bandit heuristic predictor
        #   
        #   trials_not_chosen = get_trials_not_chosen(pp_file = my_data)
        #   
        #   # append to data
        #   my_data$trials_not_chosen = trials_not_chosen
        #   
        #   # my_inits <- function(){
        #   #   list()}
        # }
        
        # posterior sampling
        my_samples <- sampling(my_stan_model
                               , my_data
                               , cores = getOption("mc.cores", 1L)
                               , chains = n_chains
                               , iter = n_iter)
        
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
        result_name <- sprintf('stan_fit_m_%s_d_%s_id_%s', stan_model, d, subject_id)
        
        # get results list
        stanfit = list(stanfit = my_samples, data = full_preprocessed_file_path)
        
        # save whole stanfit as .RData
        save(stanfit, file = sprintf('%s/%s.RData', path_to_save_results, result_name))

        # save only parameter columns as .feather (otherwise slow)
        df <- as.data.frame(stanfit$stanfit)
        df <- df[, 1:n_parameter_columns]
        df <- as.data.frame(df)
        write_feather(df, sprintf('%s/%s.feather', path_to_save_results, result_name))
        
        # delete my_samples
        rm(my_samples) 
        
        }
       
      }
    
    }
  }
}



