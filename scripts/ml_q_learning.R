# clear all objects in environment
rm(list = ls(all.names = TRUE))

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

source('ml_q_learning_functions.R')
library('reticulate')

# move up two directories
setwd('../')

### CONFIG

DATA_FILES = 'rnn_data_train_sd_0_1'#c('rnn_data_train_sd_0_05', 'rnn_data_train_sd_0_1', 'rnn_data_train_sd_met')
DATA_PATH = 'data/intermediate_data/modeling/preprocessed_data_for_modeling/pp_data_merged_%s.RData'
MODEL = 'q_learning'
SAVE_FILE_NAME = 'stanfit_%s.npy'
PATH_TO_SAVE = 'data/intermediate_data/modeling/modeling_fits/'

###


for (d in seq_along(DATA_FILES)){
  
  # load preprocessed data (object name 'res' assumed)
  load(sprintf(DATA_PATH, DATA_FILES[d]))
  
  n_test_sds = res$n_test_sds
  n_runs = res$n_runs
  n_ids = res$n_ids
  n_trials = res$n_trials
  all_choices = res$choice
  all_rewards = res$reward
  
  if (MODEL=='q_learning'){
    
    #----------------#
    # fit q-Learning #
    #----------------#
    alphas = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    betas = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    for (id in 1:n_ids){
      for(test_sd in 1:n_test_sds){
        for (run in 1:n_runs){
          
          choice = all_choices[,run,test_sd,id]
          reward = all_rewards[,run,test_sd,id]
          df = data.frame(list(choice = choice, reward = reward))
          est = fit(model = MODEL, inits = c(0.5, 0.5), 
                    df = df, upper_bounds = c(1, 200))
          
          alphas[run,test_sd,id] = est$par[1]
          betas[run,test_sd,id] = est$par[2]
        }
      }
    }
    
    n_pars = length(est$par)
    pars_array = array(NA, dim = c(n_pars, dim(alphas)))
    
    pars_array[1,,,] = alphas
    pars_array[2,,,] = betas
    
  }
  
  if (MODEL=='q_learning_hop'){
    #------------------------------------------------#
    # fit q-Learning with higher-order perseveration #
    #------------------------------------------------#
    alphas = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    alpha_hs = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    rhos = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    betas = array(NA, dim = c(n_runs, n_test_sds, n_ids))
    
    for (id in 1:n_ids){
      for(test_sd in 1:n_test_sds){
        for (run in 1:n_runs){
          
          choice = all_choices[,run,test_sd,id]
          reward = all_rewards[,run,test_sd,id]
          df = data.frame(list(choice = choice, reward = reward))
          est = fit(model = MODEL, inits = c(0.5, 0.5, 0.1, 0.5), df = df,
                    lower_bounds = c(0.1,0.1,-200, 0), 
                    upper_bounds = c(0.99, 0.99, 200, 200))
          
          alphas[run,test_sd,id] = est$par[1]
          alpha_hs[run,test_sd,id] = est$par[2]
          rhos[run,test_sd,id] = est$par[3]
          betas[run,test_sd,id] = est$par[4]
        }
      }
    }
    
    n_pars = length(est$par)
    pars_array = array(NA, dim = c(n_pars, dim(alphas)))
    
    pars_array[1,,,] = alphas
    pars_array[2,,,] = alpha_hs
    pars_array[3,,,] = rhos
    pars_array[4,,,] = betas
    
    
    
    
  }
  if (MODEL=='q_learning_hop_e'){

  }
  

  

  

  
  #save
  np = import("numpy") 
  np$save(file.path(PATH_TO_SAVE, sprintf(SAVE_FILE_NAME, DATA_FILES[d])), r_to_py(pars_array))
}


#-----------------------------------------------------#
# Sliding window modeling
#-----------------------------------------------------#

# get one data set

run = 1
test_sd = 5
id = 1

load(sprintf(DATA_PATH, DATA_FILES[1]))
n_test_sds = res$n_test_sds
n_runs = res$n_runs
n_ids = res$n_ids
n_trials = res$n_trials
all_choices = res$choice
all_rewards = res$reward

choice = all_choices[,run,test_sd,id]
reward = all_rewards[,run,test_sd,id]

# sliding window modeling
start_t = 1
max_t = 20
n_trials = 300

indices = c(0:(n_trials - max_t))

alphas = array(NA, dim = c(n_trials))
betas = array(NA, dim = c(n_trials))

for (i in indices){
  choice_i = choice[(start_t + i):(max_t + i)]
  reward_i = reward[(start_t + i):(max_t + i)]
  reward_i = reward_i * 100
  
  # initial values for alpha and beta
  inits = c(0.5, 0.1)
  if (i > 0){
    inits = c(est$par[1], est$par[2])
  }
  
  df = data.frame(list(choice = choice_i, reward = reward_i))
  est = fit(model = MODEL, inits = c(inits), # first iteration different inits
            df = df, lower_bounds = c(0.001, 0.001), upper_bounds = c(1, 7))
  
  alphas[i] = est$par[1]
  betas[i] = est$par[2]

}

is_switch = choice[1:(n_trials-1)] != choice[2:(n_trials)]

plot(alphas, type = 'l', col = 'red', lwd = 2)
points(c(2:n_trials), as.numeric(is_switch))

plot(betas, type = 'l', col = 'green', lwd = 2)
points(c(2:n_trials), as.numeric(is_switch)*7)



lines(alphas*700, type = 'l', col = 'red', lty = 2, lwd = 3)


plot(betas, type = 'l', lwd = 3)



cor(alphas[!is.na(alphas)], betas[!is.na(betas)])










# debugging nan in q-learning hop
# alpha = 0.99
# alpha_h = 0.99
# rho = 98.50261
# beta = 79.8552
# 
# 
# vals = ql_hop(choice, reward, alpha, alpha_h)
# debugonce(softmax_ql_hop)
# p = softmax_ql_hop(v = vals$v, h = vals$h, rho, beta) # problem
# 
# 
# # fix
# v = vals$v
# h = vals$h
# inputs = beta*(v+rho*h)
# normalize <- function(x, na.rm = TRUE) {
#   return((x- min(x)) /(max(x)-min(x)))
# }
# norm_inputs = normalize(inputs)
# 
# prob = exp(norm_inputs)
# prob = prob/rowSums(prob)
#   
#   
#   
# 
# 
# lik = p[cbind(seq_along(choice), choice)]
# neg_log_lik = -sum(log(lik))