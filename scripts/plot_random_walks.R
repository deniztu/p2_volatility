# clear all objects in environment
rm(list = ls(all.names = TRUE))

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# load libraries
library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggnewscale)
library(cowplot)
library('grid')

### Plot function
plot_function <- function(df, my_title = 'Run = 0', my_subtitle = ''){
  
  p <- ggplot(data = df, aes(x = trial2, y = reward, colour = bandit_rew))+
    geom_line( size = 0.5)+
    scale_y_continuous(name = "Reward")+
    scale_x_continuous(name = "Trial")+
    scale_color_manual(labels = c("b1", "b2", "b3", "b4"), values = c("steelblue1", "limegreen", "tan1", "violetred1"), name = "Bandit")+
    new_scale_colour()+
    # geom_point(aes(x = trial2, y = choice_y, colour = choice_c), size = .8)+
    scale_color_manual(labels = c("a1", "a2", "a3", "a4"), values = c("steelblue1", "limegreen", "tan1", "violetred1"), name = "Choice", guide = "none")+
    ggtitle(label = my_title, subtitle = my_subtitle)+
    theme(plot.title = element_text(hjust = 0.5, face = "bold"), plot.subtitle = element_text(hjust = 0.5))+
    ylim(0,100)
  
  return(p)
  
}


### load my run data
load("../data/rnn_raw_data/rnn_data_nh_factors_80s_trained.Rdata")

# subset my run
my_runs_df = subset(rnn_data, id == 0 & run==0 & n_hidden == 48)
my_runs_df = my_runs_df[,c("id", "n_hidden", "run", "p_rew_1", "p_rew_2", "p_rew_3", "p_rew_4", "choice", "is_switch")]

# long format
rnn_run_long <- pivot_longer(data = my_runs_df, cols = c("p_rew_1", "p_rew_2", "p_rew_3", "p_rew_4"), names_to = "bandit_rew")

# some specifications
rnn_run_long <- rnn_run_long %>% mutate(choice_y = case_when(choice == 0 ~ 97, choice == 1 ~ 98, choice == 2 ~ 99, TRUE ~ 100))
rnn_run_long$choice_c <- as.factor(rnn_run_long$choice)
rnn_run_long$reward = rnn_run_long$value * 100
rnn_run_long$trial2 = rep(1:300, each = 4)

### load daw run
load("../data/rnn_raw_data/rnn_data_nh_192_daw_walks.Rdata")

# subset my run
daw_df = subset(rnn_data, id == 0 & n_hidden == 192)

# create trial index
daw_df$trial2 = rep(c(1:300), 3)

# create run variable NEXT
daw_df$run = rep(c(1:3), each = 300)

daw_df = daw_df[,c("id", "n_hidden", "run", "p_rew_1", "p_rew_2", "p_rew_3", "p_rew_4", "choice", "is_switch")]

# long format
daw_long <- pivot_longer(data = daw_df, cols = c("p_rew_1", "p_rew_2", "p_rew_3", "p_rew_4"), names_to = "bandit_rew")

# some specifications
daw_long <- daw_long %>% mutate(choice_y = case_when(choice == 0 ~ 97, choice == 1 ~ 98, choice == 2 ~ 99, TRUE ~ 100))
daw_long$choice_c <- as.factor(daw_long$choice)
daw_long$reward = daw_long$value * 100
daw_long$trial2 = rep(rep(1:300, each = 4), 3)
daw_long = subset(daw_long, run == 2)


### plot both
p1 = plot_function(rnn_run_long, 'run = 10')
p2 = plot_function(daw_long, my_title = 'daw walk = 2')


ggsave(p1, filename = '../plots/run_10_plot.png', dpi = 300, width = 4, height = 4)
ggsave(p2, filename = '../plots/daw_walk_2_plot.png', dpi = 300, width = 4, height = 4)
