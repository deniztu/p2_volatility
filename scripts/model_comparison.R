###########################################
# script does model comparison with WAIC  #
###########################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(loo)
library(rstan)
library(Hmisc)

# CONFIG

path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
file_string = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'


# model_ids = c(1:6)
model_ids = c(1:9)

rnn_ids = c(1:31)
n_walks = 1
n_trials = 300
n_samples = 500

model_names = c("1LR","1LRP","1LRPU", "1LRPT", "1LRU", "1LRT", "1LRDP","1LRDPU", "1LRDPT")

# 1: ms_ql_1lr.stan
# 2: ms_ql_1lr_p.stan
# 3: ms_ql_1lr_p_u.stan
# 4: ms_ql_1lr_p_t.stan
# 5: ms_ql_1lr_u.stan
# 6: ms_ql_1lr_t.stan
# 7: ms_ql_1lr_dp.stan
# 8: ms_ql_1lr_dp_u.stan
# 9: ms_ql_1lr_dp_t.stan

# OLD below

# 1: ms_q_learning_model_single_lr.stan
# 2: ms_q_learning_model_single_lr_perseveration.stan
# 3: ms_q_learning_model_single_lr_perseveration_unique_bandits_heuristic.stan
# 4: ms_q_learning_model_single_lr_perseveration_trials_not_chosen_heuristic.stan
# 5: ms_q_learning_model_single_lr_unique_bandits_heuristic.stan
# 6: ms_q_learning_model_single_lr_trials_not_chosen_heuristic.stan
# 7: ms_q_learning_model_seperate_lr.stan
# 8: ms_q_learning_model_seperate_lr_perseveration.stan
# 9: ms_q_learning_model_seperate_lr_perseveration_unique_bandits_heuristic.stan
# 10: ms_q_learning_model_seperate_lr_perseveration_trials_not_chosen_heuristic.stan
# 11: ms_q_learning_model_seperate_lr_unique_bandits_heuristic.stan
# 12: ms_q_learning_model_seperate_lr_trials_not_chosen_heuristic.stan
# 13: ms_kalman_model.stan
# 14: ms_kalman_model_p.stan
# 15: ms_kalman_model_up.stan
# 16: ms_kalman_model_tp.stan
# 17: ms_kalman_model_u.stan
# 18: ms_kalman_model_t.stan




#################
# get waic's    #
#################

waic_matrix = matrix(NA, nrow = n_walks*length(rnn_ids), ncol = length(model_ids))

for (m in c(1:length(model_ids))){
  
  # for (walk in c(1:n_walks)){
    
    for (i in rnn_ids){
      
      file = sprintf(file_string ,model_ids[m], i)#,walk)
      
      load(paste0(path_to_stanfit, file))
      
      ll = extract_log_lik(stanfit$stanfit, parameter_name = "log_lik", merge_chains = TRUE)
      
      # insert 0 for na
      ll[is.na(ll)]=0
      
      waic_object = loo::waic(ll)
  
      # waic_matrix[(walk-1)*length(rnn_ids)+i+1, m] = waic_object$estimates['waic','Estimate']
      waic_matrix[i, m] = waic_object$estimates['waic','Estimate']
      
    }
    
  }

#}

# get delta waic's

for (row in c(1:nrow(waic_matrix))){
  m_waic = min(waic_matrix[row,])
  
  waic_matrix[row,] = waic_matrix[row,] - m_waic

}

###################
# delta waic plot #
###################

boxplot(waic_matrix, ylab = 'Delta(WAIC)', names = model_names)


#################################
# % best model plot             #
#################################

best_model_matrix = waic_matrix == 0 

perc_best <- apply(best_model_matrix, 2, mean)

barplot(perc_best, ylab = '% Best model', names.arg = model_names,  cex.names=0.7)

###############################
# choice predictive accuracy  #
###############################

acc_matrix = array(NA ,dim = c(n_walks*length(rnn_ids), length(model_ids), n_samples, n_trials))

for (m in c(1:length(model_ids))){
  
  for (walk in c(1:n_walks)){
    
    for (i in rnn_ids){
      
      # load stanfit and get predicted choices
      file = sprintf(file_string ,model_ids[m], i)#,walk)
      load(paste0(path_to_stanfit, file))
      
      pred_choices = rstan::extract(stanfit$stanfit, pars = "predicted_choices")
      
      # drop chain dimension
      pred_choices = pred_choices$predicted_choices[,1,]
      
      # load data and get observed choices
      load(stanfit$data)
      obs_choices = res$choices
      
      # get accuracy for each sample
      # acc_im = apply(pred_choices, 1, function(x) x==obs_choices[1,]) FOR RNNs!
      acc_im = apply(pred_choices, 1, function(x) x==obs_choices[i,])
      
      
      # transpose to matrix with dim: n_samples X n_trials
      acc_im = t(acc_im)
      
      # mean accuracy for each sample
      #mean_acc_im = apply(acc_im, 1, mean)
      
      #acc_matrix[(walk-1)*length(rnn_ids)+i+1, m, ,] = acc_im 
      
      acc_matrix[i, m, ,] = acc_im 
    } 
    
  }
}

# accuracy for each id and each model over n_trials & over n_samples
acc_matrix_im = apply(acc_matrix, c(1,2), mean)

###############################
# plot accuary for each model #
###############################

boxplot(acc_matrix_im, ylab = '% predicted choice accuracy', names = model_names)

#############################
# plot accuracy over trials #
#############################
# 
# par(mfrow=c(4,1))
# 
# acc_matrix_t = apply(acc_matrix, c(2,4), mean)
# 
# plot(acc_matrix_t[2,], type = 'l')
# 
# lines(acc_matrix_t[3,], col = m_i)
# lines(acc_matrix_t[4,], col = m_i)
# 
# plot(acc_matrix_t[3,]-acc_matrix_t[2,], main = 'KMTP-KMP', type = 'l')
# abline(h = 0, col = 'red')
# 
# plot(acc_matrix_t[4,]-acc_matrix_t[2,], main = 'KMUP-KMP', type = 'l')
# abline(h = 0, col = 'red')
# 
# 
# plot(acc_matrix_t[3,]-acc_matrix_t[4,], main = 'KMTP-KMUP', type = 'l')
# abline(h = 0, col = 'red')
# 
# 
# for (m_i in 2:length(model_ids)){
# 
#   lines(acc_matrix_t[m_i,], col = m_i)
# 
# }

################################
# plot accuracy stay vs switch #
################################

switch_accuracy_array = array(NA ,dim = c(n_walks*length(rnn_ids), length(model_ids)))
stay_accuracy_array = array(NA ,dim = c(n_walks*length(rnn_ids), length(model_ids)))

for (m in c(1:length(model_ids))){
  
  for (walk in c(1:n_walks)){
   
    for (i in rnn_ids){
      
      # load stanfit and get predicted choices
      file = sprintf(file_string ,model_ids[m], i,walk)
      load(paste0(path_to_stanfit, file))
      
      pred_choices = rstan::extract(stanfit$stanfit, pars = "predicted_choices")
      
      # drop chain dimension
      pred_choices = pred_choices$predicted_choices[,1,]
      
      # load data and get observed choices
      load(stanfit$data)
      obs_choices = res$choices
      
      # get accuracy for each sample
      acc_im = apply(pred_choices, 1, function(x) x==obs_choices[1,])
      
      # transpose to matrix with dim: n_samples X n_trials
      acc_im = t(acc_im)
      
      # get indices of switch trials
      lag_1_obs_ch = Lag(obs_choices[1,], 1) 
      switch_indices = obs_choices[1,]!=lag_1_obs_ch
      # first is non switch
      switch_indices[1] = FALSE
      # index switch trials
      acc_switch = acc_im[,switch_indices]
      # mean accuracy over samples over trials
      acc_switch = mean(acc_switch)
      
      switch_accuracy_array[i+1, m] = acc_switch
      
      # get stay trials
      stay_indices = !switch_indices
      # first non stay
      stay_indices[1] = FALSE
      # index stay trials
      acc_stay = acc_im[,stay_indices]
      # mean accuracy over samples over trials
      acc_stay = mean(acc_stay)
      
      stay_accuracy_array[(walk-1)*length(rnn_ids)+i+1, m] = acc_stay
    }
  }
}

# length(model_ids), n_samples,
par(mfrow=c(2,1))

# # stay boxplots
# stay_acc = apply(acc_matrix[,,,stay_indices], c(1,2), mean)
boxplot(stay_accuracy_array, ylab = '% predicted choice accuracy', names = model_names, main = 'stay trials')

# switch boxplots
# switch_acc = apply(acc_matrix[,,,switch_indices], c(1,2), mean)
boxplot(switch_accuracy_array, ylab = '% predicted choice accuracy', names = model_names, main = 'switch trials')


###############
# get params  #
###############

phi_mat = matrix(NA, nrow =n_samples, ncol = length(rnn_ids))

for (m in c(6)){

  for (i in rnn_ids){

    file = sprintf(file_string ,model_ids[m], i,walk)

    load(paste0(path_to_stanfit, file))

    # print(boxplot(unlist(extract(stanfit$stanfit, pars = 'phi')), main =  sprintf('%s',i)))
    #
    phi_mat[,i] = unlist(extract(stanfit$stanfit, pars = 'phi'))

  }
}

boxplot(phi_mat, horizontal = TRUE, main = 'phi')
abline(v = 0, lty = 2, col = 'red')
