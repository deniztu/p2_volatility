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
library(ggplot2)
library(ggpattern)
library(tidyverse)
library(ggpubr)

# CONFIG

path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
human_file_string = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'
rnn_file_string = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.RData'

model_ids = c(1:6, 10:17)

human_ids = c(1:31)
rnn_ids = c(0:29)
n_walks = 3
n_trials = 300

# model_names = c("SM","1LRP","1LRPU", "1LRPT", "1LRU", "1LRT")

# model_names = c("SM","SM+P","SM+BP", "SM+TP", "SM+B", "SM+T", 
#                 "SM_KM", "SM+E_KM", "SM+T_KM", "SM+B_KM", "SM+P_KM", "SM+EP_KM", "SM+TP_KM", "SM+BP_KM")

model_names = as.character(c(1:14))
# 1: ms_ql_1lr.stan
# 2: ms_ql_1lr_p.stan
# 3: ms_ql_1lr_p_u.stan
# 4: ms_ql_1lr_p_t.stan
# 5: ms_ql_1lr_u.stan
# 6: ms_ql_1lr_t.stan


# 10: ms_kalman_model.stan
# 11: ms_kalman_model_e.stan
# 12: ms_kalman_model_t.stan
# 13: ms_kalman_model_u.stan
# 14: ms_kalman_model_p.stan
# 15: ms_kalman_model_ep.stan
# 16: ms_kalman_model_tp.stan
# 17: ms_kalman_model_up.stan

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

### human

human_waic_matrix = matrix(NA, nrow = length(human_ids), ncol = length(model_ids))

for (m in c(1:length(model_ids))){
  
  for (i in human_ids){
    
    file = sprintf(human_file_string ,model_ids[m], i)
    
    load(paste0(path_to_stanfit, file))
    
    ll = extract_log_lik(stanfit$stanfit, parameter_name = "log_lik", merge_chains = TRUE)
    
    # insert 0 for na
    ll[is.na(ll)]=0
    
    waic_object = loo::waic(ll)
    
    human_waic_matrix[i, m] = waic_object$estimates['waic','Estimate']
  

  }
  
}



### RNN

rnn_waic_matrix = matrix(NA, nrow = n_walks*length(rnn_ids), ncol = length(model_ids))

for (m in c(1:length(model_ids))){
  
  for (walk in c(1:n_walks)){
    
    for (i in rnn_ids){
      
      file = sprintf(rnn_file_string ,model_ids[m], i,walk)
      
      load(paste0(path_to_stanfit, file))
      
      ll = extract_log_lik(stanfit$stanfit, parameter_name = "log_lik", merge_chains = TRUE)
      
      # insert 0 for na
      ll[is.na(ll)]=0
      
      waic_object = loo::waic(ll)
      
      rnn_waic_matrix[(walk-1)*length(rnn_ids)+i+1, m] = waic_object$estimates['waic','Estimate']
      
    }
  }
  
}

# get delta waic's

### human

human_waic_delta = colSums(human_waic_matrix) - min(colSums(human_waic_matrix))

### rnn

rnn_waic_delta = colSums(rnn_waic_matrix) - min(colSums(rnn_waic_matrix))

###################
# delta waic plot #
###################


### RNN

# create color
my_clrs_yct <- c("blue","#ADD8E6")

# create df
df1 = data.frame(delta_waic = as.vector(rnn_waic_delta),
                type = rep('RNN', length(as.vector(rnn_waic_delta))),
                lr = c(rep('Delta-Rule', length(as.vector(rnn_waic_delta[c(1:6)]))),
                       rep('Bayesian-Learner', length(as.vector(rnn_waic_delta[c(7:14)])))),
                model = model_names
)

# create grouped df
# df1 = df %>% 
#   group_by(model, lr) %>% 
#   summarise(mean = mean(delta_waic), sd = sd(delta_waic))

# reorder x-axis
df1$model <- factor(df1$model, levels = as.character(c(1:14)))

# save plot in object
p1 <- ggplot(df1,aes(x = model, y = delta_waic, fill = lr, group = lr)) +
  geom_col(alpha = 0.4, position = position_dodge(width = 10)) +
  #geom_linerange(aes(ymin=mean, ymax=mean + sd, color = lr), alpha = 0.4, size=1.2) + 
  theme_bw() +
  scale_fill_manual(values=my_clrs_yct) + 
  scale_colour_manual(values=my_clrs_yct)+
  theme(legend.position = c(0.9, 0.8),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.title = element_blank(),
        axis.title.x=element_blank(),
        axis.text.x =element_text(size=12, face="bold"),
        axis.title.y = element_text(size=12, face="bold")) + 
  ylab("Delta WAIC") + 
  scale_x_discrete(labels=c("1" = "", "2" = "",
                            "3" = "", "4" = "", "5" = "",
                            "6" = "", "7" = "", "8" = "", 
                            "9" = "", "10"="","11"="", 
                            "12"="", "13"="", "14" ="")) +
  ggtitle('RNN')



### human

# create df
df2 = data.frame(delta_waic = as.vector(human_waic_delta),
                type = rep('Human', length(as.vector(human_waic_delta))),
                lr = c(rep('Delta-Rule', length(as.vector(human_waic_delta[c(1:6)]))),
                       rep('Bayesian-Learner', length(as.vector(human_waic_delta[c(7:14)])))),
                model = model_names
)

# df2 = df %>% 
#   group_by(model, lr) %>% 
#   summarise(mean = mean(delta_waic), sd = sd(delta_waic))

# reorder x-axis
df2$model <- factor(df1$model, levels = as.character(c(1:14)))



# create color
my_clrs_yct <- c("black","#808080")


p2 <- ggplot(df2,aes(x = model, y = delta_waic, fill = lr, group = lr)) +
  geom_col(alpha = 0.4, position = position_dodge(width = 10)) +
  theme_bw() +
  scale_fill_manual(values=my_clrs_yct) + 
  scale_colour_manual(values=my_clrs_yct)+
  theme(legend.position = c(0.9, 0.8),
        plot.title = element_text(hjust = 0.5,face="bold"),
        legend.title = element_blank(),
        axis.title.x=element_blank(),
        axis.text.x =element_text(size=12, face="bold"),
        axis.title.y = element_text(size=12, face="bold")) + 
  ylab("Delta WAIC") + 
  scale_x_discrete(labels=c("1" = "SM", "2" = "SM+P",
                            "3" = "SM+BP", "4" = "SM+TP", "5" = "SM+B",
                            "6" = "SM+T", "7" = "SM", "8" = "SM+E",
                            "9" = "SM+T", "10"="SM+B","11"="SM+P",
                            "12"="SM+EP", "13"="SM+TP", "14" ="SM+BP")) +
  ggtitle('Human')

# two plots in one
p3 = ggarrange(p1, p2, nrow = 2, ncol = 1)

ggsave('plots/model_comparison.png', dpi = 600,  width = 10, height = 6)
#ggsave('plots/model_comparison.png', width = 20, height = 15, units = "in")


model_names1 = c("SM","SM+P","SM+BP", "SM+TP", "SM+B", "SM+T", 
                                  "SM_KM", "SM+E", "SM+T", "SM+B", "SM+P", "SM+EP", "SM+TP", "SM+BP")


# check stats
df1$model = model_names1
print('RNN')
print(df1)

df2$model = model_names1
print('Human')
print(df2)
