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

###########
# CONFIG  #
###########
path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
human_file_string = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'
rnn_file_string = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.RData'

# see modeling_functions for mappings of numbers to cognitive models
model_ids = c(1:21)

human_ids = c(1:31)
rnn_ids = c(0:29)
n_walks = 3
n_trials = 300

# model_names = c("SM","1LRP","1LRPU", "1LRPT", "1LRU", "1LRT")

# model_names = c("SM","SM+P","SM+BP", "SM+TP", "SM+B", "SM+T", 
#                 "SM_KM", "SM+E_KM", "SM+T_KM", "SM+B_KM", "SM+P_KM", "SM+EP_KM", "SM+TP_KM", "SM+BP_KM")

model_names = as.character(model_ids)

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

####################
# get delta waic's #
####################

### human
human_waic_delta = colSums(human_waic_matrix) - min(colSums(human_waic_matrix))

### rnn
rnn_waic_delta = colSums(rnn_waic_matrix) - min(colSums(rnn_waic_matrix))

###################
# delta waic plot #
###################

# function for plotting 
delta_waic_plot <- function(df, my_colors, my_ylab, my_x_ticks, my_title){
  
  p <- ggplot(df,aes(x = model, y = delta_waic, fill = lr, group = lr)) +
    
    geom_col(alpha = 0.4, position = position_dodge(width = 10)) +
    theme_bw() +
    scale_fill_manual(values=my_colors) + 
    scale_colour_manual(values=my_colors)+
    theme(legend.position = c(0.9, 0.8),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.title = element_blank(),
          axis.title.x=element_blank(),
          axis.text.x =element_text(size=7, face="bold"),
          axis.title.y = element_text(size=8, face="bold")) + 
    ylab(my_ylab) + 
    scale_x_discrete(labels = my_x_ticks) +
    ggtitle(my_title)
  
  return(p)
}


### RNN

# create color
rnn_colors <- c("blue","#ADD8E6")

# create df
df1 = data.frame(delta_waic = as.vector(rnn_waic_delta),
                type = rep('RNN', length(as.vector(rnn_waic_delta))),
                lr = c(rep('Delta-Rule', length(as.vector(rnn_waic_delta[c(1:9)]))),
                       rep('Bayesian-Learner', length(as.vector(rnn_waic_delta[c(10:21)])))),
                model = model_names
)

# reorder x-axis
df1$model <- factor(df1$model, levels = model_names)

# save plot in object
plot1_x_ticks = c("1" = "", "2" = "", "3" = "", "4" = "",
                  "5" = "", "6" = "", "7" = "", "8" = "", 
                  "9" = "", "10"="","11"="", "12"="", 
                  "13"="", "14" ="", "15"="", "16"="",
                  "17"="", "18"="", "19"="", "20"="", 
                  "21"="")

p1 <- delta_waic_plot(df = df1, my_colors = rnn_colors, my_ylab = "Delta WAIC",
                my_x_ticks = plot1_x_ticks, my_title = 'RNN')


### human

# create df
df2 = data.frame(delta_waic = as.vector(human_waic_delta),
                type = rep('Human', length(as.vector(human_waic_delta))),
                lr = c(rep('Delta-Rule', length(as.vector(human_waic_delta[c(1:9)]))),
                       rep('Bayesian-Learner', length(as.vector(human_waic_delta[c(10:21)])))),
                model = model_names
)

# reorder x-axis
df2$model <- factor(df1$model, levels = model_names)

# create color
human_colors <- c("black","#808080")

plot2_x_ticks <- c(
  "1" = "SM", "2" = "SM+P",
  "3" = "SM+BP", "4" = "SM+TP",
  "5" = "SM+B", "6" = "SM+T",
  "7" = "SM+DP", "8" = "SM+BDP",
  "9" = "SM+TDP", "10"="SM",
  "11"="SM+E", "12"="SM+T",
  "13"="SM+B", "14" ="SM+P",
  "15"="SM+EP", "16" ="SM+TP",
  "17"="SM+BP", "18" ="SM+DP",
  "19"="SM+BDP", "20" ="SM+TDP",
  "21"="SM+EDP"
  )


p2 <- delta_waic_plot(df = df2, my_colors = human_colors, my_ylab = "Delta WAIC",
                my_x_ticks = plot2_x_ticks, my_title = 'Human')



### two plots in one
p3 = ggarrange(p1, p2, nrow = 2, ncol = 1)

# save plot
ggsave('plots/model_comparison_dp.png', dpi = 600,  width = 10, height = 6)
#ggsave('plots/model_comparison.png', width = 20, height = 15, units = "in")



# check stats
df1$model = plot2_x_ticks
print('RNN')
print(df1)

df2$model = plot2_x_ticks
print('Human')
print(df2)
