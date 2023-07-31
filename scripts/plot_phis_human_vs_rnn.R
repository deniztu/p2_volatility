######################################
# script handles cognitive modeling  #
######################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../data/intermediate_data/modeling/modeling_fits/')

library(ggplot2)
library(PupillometryR)


# 3, 4, 8, 9 
# 
# "SM+BP", "SM+TP", "SM+B", "SM+T"
# 
# 11, 12, 13, 15, 16, 17
# 
# "SM+E", "SM+T", "SM+U", "SM+EP", "SM+TP", "SM+BP"


# if q_learning
MY_MODELS = c(3, 4, 8, 9)
MODEL_NAMES = c("SM+BP", "SM+TP", "SM+B", "SM+T")
my_clrs_yct <- c("#808080", "#ADD8E6", "#808080", "#ADD8E6")



# if kalman filter
#MY_MODELS = c(11, 12, 13, 15, 16, 17)
#MODEL_NAMES = c("SM+E", "SM+T", "SM+B", "SM+EP", "SM+TP", "SM+BP")
#my_clrs_yct <- c("black", "blue", "black", "blue")


RNN_INSTANCES = 20
N_WALKS = 3
N_SUBS = 31

RNN_STRING = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.Rdata'
HUMAN_STRING = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'

# initialize matrix
phi_lstm = matrix(99, nrow = RNN_INSTANCES*N_WALKS, ncol = length(MY_MODELS))
phi_human = matrix(0, nrow = N_SUBS, ncol = length(MY_MODELS))

### get posterior parameters

# RNNS
for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){
    for (m in c(1:length(MY_MODELS))){
      
      load(sprintf(RNN_STRING, MY_MODELS[m], id, w))
      
      df = as.data.frame(stanfit$stanfit)
      
      phi_lstm[(w-1)*RNN_INSTANCES + id + 1, m] = median(df$`phi[1]`)
      
    }
  }
}

# Humans
for (id in c(1:(N_SUBS))){
  for (m in c(1:length(MY_MODELS))){
      
      load(sprintf(HUMAN_STRING, MY_MODELS[m], id))
      
      df = as.data.frame(stanfit$stanfit)
      
      phi_human[id, m] = median(df$`phi[1]`)
      
  }
}


### plot

# create plotting df
phi = c(as.vector(phi_lstm),as.vector(phi_human))

model = c()

for (m in MODEL_NAMES){
  model = c(model, c(rep(m, nrow(phi_lstm))))
}

for (m in MODEL_NAMES){
  model = c(model, c(rep(m, nrow(phi_human))))
}


# 
# model = c(rep(MODEL_NAMES[1], nrow(phi_lstm)), rep(MODEL_NAMES[2], nrow(phi_lstm)),
#   rep(MODEL_NAMES[1], nrow(phi_human)), rep(MODEL_NAMES[2], nrow(phi_human)))

agent = c(rep("RNN", length(MODEL_NAMES)*nrow(phi_lstm)), rep("Human", length(MODEL_NAMES)*nrow(phi_human)))

df <- data.frame(phi, model, agent)

# ATTENTION: temporary solution for outlier
df = df[df$phi > -100,]




#df1 = df[df$model == 'SM+B' | df$model == 'SM+BP',]
#df1 = df[df$model == 'SM+T' | df$model == 'SM+TP',]
df1 = df[df$model == 'SM+E' | df$model == 'SM+EP',]





p <- ggplot(df1) +
  aes(x = model,
      y = phi,
      fill = agent) + 
  geom_point(aes(color = agent), 
             position = position_jitterdodge(jitter.width = .15, # to sepreate jitter into their group.
                                             dodge.width = .3), 
             size = 2, 
             alpha = 0.4,
             show.legend = F) +
  geom_boxplot(aes(color = agent),
               width = .3, 
               outlier.shape = NA,
               alpha = 0) + 
  #geom_flat_violin(position = position_nudge(x = .5, y =0), alpha = .5)+
  theme_bw() +
  scale_colour_manual(values=my_clrs_yct)+
  theme(legend.position = c(0.9, 0.8),
        legend.title = element_blank(),
        legend.text = element_text(size=8, face = "bold"),
        axis.title.x=element_blank(),
        axis.text.x =element_text(size=10, face="bold"),
        axis.title.y = element_text(size=10, face="bold")) + 
  ylab("Phi") 

ggsave("phi_comparison_human_lstm_delta_rule3.png",   width = 5, height = 4, dpi = 300)


# to look at median values 

df %>%
  group_by(model, agent) %>%
  summarise(median = median(phi), sd = sd(phi))


