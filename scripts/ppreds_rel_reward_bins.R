############################################################################################
# Script does posterior predictions for 4-armed restless bandits with relative reward bins #
############################################################################################

library(ggplot2)

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# CONFIG
path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
file_string = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'
data_string = 'data/human_raw_data/data_chakroun_wiehler.csv'

model_ids = c(1:9)

num_subjects = 31

model_names = c(
  'ms_ql_1lr',
  'ms_ql_1lr_p',
  'ms_ql_1lr_p_u',
  'ms_ql_1lr_p_t',
  'ms_ql_1lr_u',
  'ms_ql_1lr_t',
  'ms_ql_1lr_dp',
  'ms_ql_1lr_dp_u',
  'ms_ql_1lr_dp_t'
)

# 1: ms_ql_1lr.stan
# 2: ms_ql_1lr_p.stan
# 3: ms_ql_1lr_p_u.stan
# 4: ms_ql_1lr_p_t.stan
# 5: ms_ql_1lr_u.stan
# 6: ms_ql_1lr_t.stan
# 7: ms_ql_1lr_dp.stan
# 8: ms_ql_1lr_dp_u.stan
# 9: ms_ql_1lr_dp_t.stan



######################################
# choice accuracy by relative reward #
######################################

# lists will be filled in the loop
mean_acc_list = c()
rel_reward_list = c()
lower_list = c() # 5% percentile
upper_list = c() # 95% percentile
model_list = c() # contains model name
sub_list = c() # contains subject number


for (i in 1:num_subjects){
  for (m in model_ids){
    
    # load data and get observed choices
    res = read.csv(data_string)
    res = res[res$vp==i,]
    filter = which(res$choice!=0) 
    filtered_res = res[filter, ] # delete invalid trials
    obs_choices = filtered_res$choice
    rewards = filtered_res[, c('reward_b1', 'reward_b2', 'reward_b3', 'reward_b4')]
    
    # load stanfit and get predicted choices
    file = sprintf(file_string ,model_ids[m], i)#,walk)
    load(paste0(path_to_stanfit, file))
    pred_choices = rstan::extract(stanfit$stanfit, pars = "predicted_choices")
    pred_choices = pred_choices$predicted_choices[,1,] # drop chain dimension
    pred_choices = pred_choices[,filter] # delete invalid trials
    
    # calculate average reward
    mean_rewards = apply(rewards, 1, mean)
    
    # calculate best reward
    best_reward = c()
    for (t in c(1:length(obs_choices))){
      best_reward = c(best_reward, max(rewards[t,]))
    }
    
    
    # get chosen rewards
    # chosen_rewards = c()
    # for (t in c(1:length(obs_choices))){
    #   chosen_rewards = c(chosen_rewards, rewards[t, obs_choices[t]])
    # }
    
    # get relative reward
    rel_rewards = as.numeric(best_reward) - as.numeric(mean_rewards)
    
    # bin relative rewards
    my_breaks = seq(min(rel_rewards)-0.01, max(rel_rewards), 5)
    my_breaks[length(my_breaks)] = max(rel_rewards)+0.01 # to include max value
    rel_reward_bins = cut(rel_rewards, breaks = my_breaks)
    
    # get accuracy for each sample
    acc_im = apply(pred_choices, 1, function(x) x==obs_choices)
    # transpose to matrix with dim: n_samples X n_trials
    acc_im = t(acc_im)
    
    # get mean for each trial over each mcmc sample 
    mean_acc = colMeans(acc_im)
    mean_acc_and_bin = as.data.frame(
      
      list(mean_acc = mean_acc,
           rel_reward_bins = rel_reward_bins)
      
    )
    
    # mean for each bin
    mean_acc_by_bin = aggregate(mean_acc, by = list(rel_reward_bins), mean)
    
    # 5 and 95 percentile for each bin
    percentiles = aggregate(mean_acc ~ rel_reward_bins, mean_acc_and_bin, function(x) quantile(x, prob=c(.05,.95)))
    
    mean_acc_by_bin['lower'] = percentiles['mean_acc']$mean_acc[,'5%']
    mean_acc_by_bin['upper'] = percentiles['mean_acc']$mean_acc[,'95%']
    
    mean_acc_by_bin['model'] = model_names[m]
    mean_acc_by_bin['subject'] = i
    
    # append to vectors
    mean_acc_list = c(mean_acc_list, mean_acc_by_bin$x)
    rel_reward_list = c(rel_reward_list, mean_acc_by_bin$Group.1)
    lower_list = c(lower_list, mean_acc_by_bin$lower) # 5% percentile
    upper_list = c(upper_list, mean_acc_by_bin$upper) # 95% percentile
    model_list = c(model_list, mean_acc_by_bin$model) # contains model name
    sub_list = c(sub_list, mean_acc_by_bin$subject) # contains subject number
    
  }
}

############
# plotting #
############

plot_df = as.data.frame(list(mean_acc = mean_acc_list,
                   rel_reward_bins = rel_reward_list,
                   lower = lower_list, 
                   upper = upper_list, 
                   model = model_list, 
                   sub = sub_list))


###############
# per subject #
###############

for (i in 1:num_subjects){
  
  plot_df_i = plot_df[plot_df$sub==i,]
  
  # Basic line plot with points
  p = ggplot(data=plot_df_i, aes(x=rel_reward_bins, y=mean_acc, group = model, color=model)) +
    geom_line()+
    geom_point() + 
    scale_color_manual(values=c("black", "lightblue", "blue",
                                "darkblue", "lightgreen", "green",
                                "darkgreen", "grey", "lightgrey"))+
    ggtitle(sprintf('%s', i))
  
  print(p)
  
}







#ggsave(plot = p, width = 10, height = 10, dpi = 300, filename = "single_sub.pdf")


# aggregate plot over all subjects
aggr_plot_df = aggregate(mean_acc ~ rel_reward_bins + model, data = plot_df,FUN = mean)

p = ggplot(data=aggr_plot_df, aes(x=rel_reward_bins, y=mean_acc, group = model, color=model)) +
  geom_line()+
  geom_point() + 
  scale_color_manual(values=c("black", "lightblue", "blue",
                              "darkblue", "lightgreen", "green",
                              "darkgreen", "grey", "lightgrey"))+
  ggtitle(sprintf('aggregated'))

p
#ggsave(plot = p, width = 10, height = 10, dpi = 300, filename = "aggr.pdf")

########################
# pairwise comparisons #
########################

pw_plot_df= aggr_plot_df[aggr_plot_df$model == 'ms_ql_1lr' | aggr_plot_df$model == 'ms_ql_1lr_dp' | aggr_plot_df$model == 'ms_ql_1lr_dp_t',]

p = ggplot(data=pw_plot_df, aes(x=rel_reward_bins, y=mean_acc, group = model, color=model)) +
  geom_line()+
  geom_point() + 
  scale_color_manual(values=c("black", "lightblue", "blue",
                              "darkblue", "lightgreen", "green",
                              "darkgreen", "grey", "lightgrey"))+
  ggtitle(sprintf('aggregated'))

p


# get optimal choices
# optimal_choices = c()
# for (t in c(1:length(obs_choices))){
#   optimal_choices = c(optimal_choices, which(rewards[t,] == max(rewards[t,])))
# }
# 
# # scatterplot relative reward vs percent optimal with observed choices
# rel_reward_bins = cut(rel_rewards, breaks = seq(min(rel_rewards), max(rel_rewards), 15))
# perc_optimal = as.data.frame(list(is_optimal = obs_choices==optimal_choices, rel_reward_bins = rel_reward_bins))
# perc_optimal = aggregate(perc_optimal$is_optimal, by = list(perc_optimal$rel_reward_bins), mean)
# obviously not optimal with negative rel. reward
# plot(perc_optimal$Group.1, perc_optimal$x)
