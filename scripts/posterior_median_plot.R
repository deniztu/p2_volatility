###############################################################################################
# This script creates plots of posterior predictive accuracy and median values of parameters  #
# and saves these values into a csv file for later use in JASP                                #
###############################################################################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../')

library(ggplot2)
library(cowplot)
library(BayesFactor)

# CONFIG

STAN_PATH = 'data/intermediate_data/modeling/modeling_fits/'

# specify cognitive model (see stan_model legend in scripts/modeling_function.R)
MY_MODEL =  21 # 18 =  SM+DP, 21 = SM+EDP

# colors used in the plot
my_clrs_yct <- c("#808080", "#ADD8E6", "#808080", "#ADD8E6")

RNN_INSTANCES = 30
N_WALKS = 3
N_SUBS = 31

MY_PARS = c('beta', 'rho', 'alpha_h', 'phi') # must be the same name as pars in stanfit

# Number of samples for posterior predictives
N_SAMPLES = 500 

# x-axis label under the first plot
MY_LABELS = c('SM+EDP')

# file name with placeholders (%s) for model (int according to stan_model legend), id and bandit parameter
RNN_STRING = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.Rdata'

# file name with placeholders (%s) for model (int according to stan_model legend) and id
HUMAN_STRING = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'

######################################
### get posterior median values    ###
######################################

# RNNS
post_medians_rnn = matrix(99, nrow = RNN_INSTANCES*N_WALKS, ncol = length(MY_PARS))
colnames(post_medians_rnn) = MY_PARS

for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){
      
      load(paste0(STAN_PATH, sprintf(RNN_STRING, MY_MODEL, id, w)))
      mcmc = rstan::extract(stanfit$stanfit, pars = MY_PARS)
      post_medians_rnn[(w-1)*RNN_INSTANCES + id + 1, ] = unlist(lapply(mcmc, median))
      
  }
}

# Humans
post_medians_human = matrix(99, nrow = N_SUBS, ncol = length(MY_PARS))
colnames(post_medians_human) = MY_PARS


for (id in c(1:(N_SUBS))){
    load(paste0(STAN_PATH, sprintf(HUMAN_STRING, MY_MODEL, id)))
    
    mcmc = rstan::extract(stanfit$stanfit, pars = MY_PARS)
    post_medians_human[id, ] = unlist(lapply(mcmc, median))
}

######################################
### create median parameter plots  ###
######################################

# create plotting df

combined_post_medians = rbind(post_medians_rnn, post_medians_human)
posterior_medians = c(combined_post_medians)

# create agent labels
agent = c(rep("RNN", nrow(post_medians_rnn)), rep("Human", nrow(post_medians_human)))
agent = rep(agent, length(MY_PARS))

# create parameter labels 
par = rep(MY_PARS, each = nrow(combined_post_medians))

# merge to df
df <- data.frame(posterior_medians, par, agent)

# define function for plotting
plot_posterior_medians <- function(df, ylabel = 'Posterior Median', xlabel = '', legend = TRUE, my_tag = ''){
  
  if (legend){
    legend = c(0.8, 0.8)
  }
  else{
    legend = 'none' 
  }
  
  ggplot(df) +
    aes(x = par,
        y = posterior_medians,
        fill = agent) + 
    geom_point(aes(fill = agent), 
               position = position_jitterdodge(jitter.width = .15, # to sepreate jitter into their group.
                                               dodge.width = .3), 
               size = 3, 
               alpha = 0.9,
               show.legend = F,
               colour = 'white',
               pch = 21,
               stroke = 2
               ) +
    geom_boxplot(aes(color = agent),
                 width = .3, 
                 outlier.shape = NA,
                 alpha = 0,
                 colour = c('black', 'blue'),
                 lwd = 1) +
    theme_bw() +
    scale_fill_manual(values=my_clrs_yct)+
    theme(legend.position = legend,
          legend.title = element_blank(),
          legend.text = element_text(size=20, face = "bold"),
          axis.title.x=element_text(size=20, face="bold"),
          axis.text.x =element_blank(),
          axis.title.y = element_text(size=20, face="bold"),
          axis.text.y = element_text(size = 20, face="bold"),
          plot.tag = element_text(size = 40, face = "bold")) + 
    ylab(ylabel) +
    xlab(xlabel) +
    labs(tag = my_tag)
} 

# parameter plots
p1 = plot_posterior_medians(df[df$par == 'beta',], legend = FALSE,
                            my_tag = 'b', xlabel = bquote(beta))
p2 = plot_posterior_medians(df[df$par == 'rho',], ylabel = '',
                            legend = FALSE, my_tag = 'c',
                            xlabel = bquote(rho))
p3 = plot_posterior_medians(df[df$par == 'alpha_h',], ylabel = '', legend = FALSE,
                            my_tag = 'd', xlabel = bquote(alpha[h]))
p4 = plot_posterior_medians(df[df$par == 'phi',], ylabel = '', legend = FALSE,
                            my_tag = 'e', xlabel = expression(phi))

# quick look at median values 
df %>%
  group_by(agent, par) %>%
  summarise(median = median(posterior_medians)
            , min = min(posterior_medians)
            , max = max(posterior_medians))

# # bayesian t-test (two samples)
# ttestBF(formula = posterior_medians ~ agent, data = df[df$par=='beta',])
# ttestBF(formula = posterior_medians ~ agent, data = df[df$par=='alpha_h',])
# ttestBF(formula = posterior_medians ~ agent, data = df[df$par=='rho',])
# ttestBF(formula = posterior_medians ~ agent, data = df[df$par=='phi',])
# 
# # bayesian t-test (one samples)
# ttestBF(df[df$par=='phi'& df$agent=='RNN' ,'posterior_medians'], mu = 0)


######################################
### get posterior predictive values###
######################################

# initialize matrix
rnn_accuracy = rep(NA, RNN_INSTANCES * N_WALKS)
human_accuracy = rep(NA, N_SUBS)

# RNNS

for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){

      # load stanfit
      load(paste0(STAN_PATH, sprintf(RNN_STRING, MY_MODEL, id, w)))
      # load predicted choices
      df = as.data.frame(stanfit$stanfit)
      pred_choices = as.matrix(df[,startsWith(names(df), 'predicted_choices')])
      # set dimnaes attribute to null for calculation of is_predicted
      attr(pred_choices, 'dimnames') = NULL 
      # load observed data
      load(stanfit$data)
      # get observed choices
      obs_choices = as.integer(res$choices)
      # transform predict choice matrix to boolean: True is predicted, else False
      is_predicted = t(pred_choices) == obs_choices # transpose needed
      is_predicted = t(is_predicted) # reverse transpose
      # get accuracy over all posterior samples and trials
      rnn_accuracy[(w-1)*RNN_INSTANCES + id + 1] = mean(is_predicted)
  }
}

# Human

for (sub in c(1:(N_SUBS))){
  # for (m in c(1:length(MY_MODELS))){
    
    # load stanfit
    load(paste0(STAN_PATH, sprintf(HUMAN_STRING, MY_MODEL, sub)))
    # load predicted choices
    df = as.data.frame(stanfit$stanfit)
    pred_choices = as.matrix(df[,startsWith(names(df), 'predicted_choices')])
    # set dimnaes attribute to null for calculation of is_predicted
    attr(pred_choices, 'dimnames') = NULL 
    # load observed data
    load(stanfit$data)
    # get observed choices
    obs_choices = as.integer(res$choices[sub,])
    # transform predict choice matrix to boolean: True is predicted, else False
    is_predicted = t(pred_choices) == obs_choices # transpose needed
    is_predicted = t(is_predicted) # reverse transpose
    # insert NAN for missing values (choice = 0)
    is_predicted[obs_choices==0] = NaN
    # get accuracy over all posterior samples and trials
    human_accuracy[sub] = mean(is_predicted, na.rm = TRUE)
  # }
}

#################################################
### create posterior predictive accuracy plot ###
#################################################

# create plotting df
accuracy = c(human_accuracy,rnn_accuracy)
agent = c(rep('Human', N_SUBS),rep('RNN', RNN_INSTANCES*N_WALKS))
model = c(rep(MY_MODEL, each=N_SUBS), rep(MY_MODEL, times = N_WALKS*RNN_INSTANCES))
model = as.factor(model)
levels(model) = MY_LABELS
# merge to df
df <- data.frame(accuracy, agent, model)

# plotting function
ppred_plot <- ggplot(df) +
  aes(x = model,
      y = accuracy,
      fill = agent) + 
  geom_point(aes(fill = agent), 
             position = position_jitterdodge(jitter.width = .15, # to sepreate jitter into their group.
                                             dodge.width = .3), 
             size = 3, 
             alpha = 0.9,
             show.legend = F,
             colour = 'white',
             pch = 21,
             stroke = 2
  ) +
  geom_boxplot(aes(color = agent),
               width = .3, 
               outlier.shape = NA,
               alpha = 0,
               #colour = c('black', 'blue'),
               lwd = 1) + 
  theme_bw() +
  scale_fill_manual(values=my_clrs_yct)+
  scale_color_manual(values=c('black', 'blue'))+
  theme(legend.position = c(0.7, 0.2),
        legend.title = element_blank(),
        legend.text = element_text(size=15, face = "bold"),
        axis.title.x=element_blank(),
        axis.text.x =element_text(size=20, face="bold"),
        axis.title.y = element_text(size=20, face="bold"),
        axis.text.y = element_text(size = 20, face="bold"),
        plot.tag = element_text(size = 40, face = "bold")) + 
  ylab('Predictive accuracy') + 
  xlab(MY_LABELS) + labs(tag = "a")

# quick look at mean values 
df %>%
  group_by(agent, model) %>%
  summarise(mean = mean(accuracy), sd = sd(accuracy))

# # bayesian t-test
# ttestBF(formula = accuracy ~ agent, data = df)

########################
### create multiplot ###
########################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

setwd('../')

# plot into grid
plot_grid(ppred_plot, p1, p2,p3,p4, labels = c('', '', ''), nrow = 1)

# save
ggsave("plots/figure_4_SM_EDP.svg",   dpi = 1000,  width = 20, height = 4, unit = 'in')

##########################
### save df for jasp   ###
##########################

agent = c(rep("RNN", nrow(post_medians_rnn)), rep("Human", nrow(post_medians_human)))


jasp_list = list(accuracy= c(rnn_accuracy,human_accuracy), 
                 beta = combined_post_medians[,'beta'],
                 rho = combined_post_medians[,'rho'],
                 alpha_h = combined_post_medians[,'alpha_h'],
                 phi = combined_post_medians[,'phi'],
                 agent = agent)

jasp_df = data.frame(jasp_list)

write.csv(jasp_df, 'data/intermediate_data/jasp_analysis/post_median_data.csv',
          row.names = F)




