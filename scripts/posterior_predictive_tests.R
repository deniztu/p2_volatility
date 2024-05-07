#############################################################
# This script creates plots of posterior predictive checks  #
#############################################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../')

library(ggplot2)
library(cowplot)
library(BayesFactor)
library(ggdist)
library(reticulate)
# import numpy
np = import("numpy")



# CONFIG

STAN_PATH = 'data/intermediate_data/modeling/modeling_fits/'

# specify cognitive model (see stan_model legend in scripts/modeling_function.R)
MY_MODEL =  21 # 18 =  SM+DP, 21 = SM+EDP

# colors used in the plot
my_clrs_yct <- c("#808080", "#ADD8E6", "#808080", "#ADD8E6")

RNN_INSTANCES = 30
N_WALKS = 3
N_SUBS = 31
N_TRIALS = 300


# file name with placeholders (%s) for model (int according to stan_model legend), id and bandit parameter
RNN_STRING = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.Rdata'

# file name with placeholders (%s) for model (int according to stan_model legend) and id
HUMAN_STRING = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'

STAN_PATH = 'data/intermediate_data/modeling/modeling_fits/'


#####################################
### get posterior predictive checks##
#####################################

### Simulating Figure 2C

# RNNs
rnn_switch_probs = c()

for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){
    
    
    load(paste0(STAN_PATH, sprintf(RNN_STRING, MY_MODEL, id, w)))
    mcmc = rstan::extract(stanfit$stanfit, pars = c("predicted_choices"))
    mcmc = mcmc$predicted_choices
    n_sims = dim(mcmc)[1]
    ppreds = array(dim = n_sims)
    
    for (s in c(1:n_sims)){
      choice_t_1 = mcmc[s, 1, 1:299]
      choice_t = mcmc[s, 1, 2:300]
      switch_prob = sum(choice_t_1 != choice_t)/299
      ppreds[s] = switch_prob
    }
    
    rnn_switch_probs = c(rnn_switch_probs, median(ppreds))
    
  }
}

# Humans
human_switch_probs = c()

for (id in c(1:(N_SUBS))){
    
    load(paste0(STAN_PATH, sprintf(HUMAN_STRING, MY_MODEL, id)))
    mcmc = rstan::extract(stanfit$stanfit, pars = c("predicted_choices"))
    mcmc = mcmc$predicted_choices
    n_sims = dim(mcmc)[1]
    ppreds = array(dim = n_sims)
    
    for (s in c(1:n_sims)){
      choice_t_1 = mcmc[s, 1, 1:299]
      choice_t = mcmc[s, 1, 2:300]
      switch_prob = sum(choice_t_1 != choice_t)/299
      ppreds[s] = switch_prob
    }
    
    human_switch_probs = c(human_switch_probs, median(ppreds))
    
}

# make the plot

plot_df = data.frame(matrix(NA, nrow = length(rnn_switch_probs)+ length(human_switch_probs), ncol = 2))

plot_df[,1] = c(rnn_switch_probs, human_switch_probs)
plot_df[,2] = c(rep('RNN', length(rnn_switch_probs)), rep('Human', length(human_switch_probs)))

colnames(plot_df) = c('switch_prob', 'type')

ggplot(data = plot_df, aes(x = factor(type), y = switch_prob, fill = factor(type))) +
  
  # add half-violin from {ggdist} package
  stat_halfeye(
    # adjust bandwidth
    adjust = 0.5,
    # move to the right
    justification = -0.2,
    # remove the slub interval
    .width = 0,
    point_colour = NA
  ) +
  geom_boxplot(
    width = 0.12,
    # removing outliers
    outlier.color = NA,
    alpha = 0.5
  )

# save csv files
write.csv(plot_df, './data/intermediate_data/modeling/ppred_fig_2_switch_probs.csv')


### Export predicted choices for reproduction of Figure 6A in Python


# export predicted choices of RNNs
rnn_predicted_choices = array(NA, dim = c(n_sims, RNN_INSTANCES*N_WALKS, N_TRIALS))

for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){
    
    load(paste0(STAN_PATH, sprintf(RNN_STRING, MY_MODEL, id, w)))
    mcmc = rstan::extract(stanfit$stanfit, pars = c("predicted_choices"))
    mcmc = mcmc$predicted_choices
    rnn_predicted_choices[,N_WALKS*id+w,] = mcmc

  }
}

np$save("./data/intermediate_data/modeling/rnn_ppred_choices.npy",r_to_py(rnn_predicted_choices))

# export predicted choices of Humans
human_predicted_choices = array(NA, dim = c(n_sims, N_SUBS, N_TRIALS))


for (id in c(1:(N_SUBS))){
  load(paste0(STAN_PATH, sprintf(HUMAN_STRING, MY_MODEL, id)))
  mcmc = rstan::extract(stanfit$stanfit, pars = c("predicted_choices"))
  mcmc = mcmc$predicted_choices
  human_predicted_choices[,id,] = mcmc
}

np$save("./data/intermediate_data/modeling/human_ppred_choices.npy",r_to_py(human_predicted_choices))





