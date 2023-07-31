# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
#setwd('../data/intermediate_data/modeling/modeling_fits/')

setwd('../')

# load packages
library(rstan)
library(Hmisc)

# CONFIG

MY_MODELS = c(15)
MY_LABELS = c('SM+EP')

# colors
my_clrs_yct <- c("#808080", "#ADD8E6", "#808080", "#ADD8E6")

RNN_INSTANCES = 20
N_WALKS = 3
N_SUBS = 31

N_SAMPLES = 500
N_TRIALS = 300

STAN_PATH = ('data/intermediate_data/modeling/modeling_fits/')
RNN_STRING = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.Rdata'
HUMAN_STRING = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'

### get posterior predictive accuracy

# initialize matrix
rnn_accuracy = matrix(NA, nrow = RNN_INSTANCES * N_WALKS, ncol = length(MY_MODELS))
human_accuracy = matrix(NA, nrow = N_SUBS, ncol = length(MY_MODELS))

# RNNS

for (id in c(0:(RNN_INSTANCES-1))){
  for (w in c(1:N_WALKS)){
    for (m in c(1:length(MY_MODELS))){
      
      # load stanfit
      load(paste0(STAN_PATH, sprintf(RNN_STRING, MY_MODELS[m], id, w)))
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
      rnn_accuracy[(w-1)*RNN_INSTANCES + id + 1, m] = mean(is_predicted)
    }
  }
}

# Human

for (sub in c(1:(N_SUBS))){
    for (m in c(1:length(MY_MODELS))){
      
      # load stanfit
      load(paste0(STAN_PATH, sprintf(HUMAN_STRING, MY_MODELS[m], sub)))
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
      # get accuracy over all posterior samples and trials
      human_accuracy[sub, m] = mean(is_predicted)
    }
}

### plotting

# create plotting df
accuracy = c(as.vector(human_accuracy),as.vector(rnn_accuracy))
agent = c(rep('Human', N_SUBS*length(MY_MODELS)),rep('RNN', RNN_INSTANCES*N_WALKS*length(MY_MODELS)))
model = c(rep(MY_MODELS, each=N_SUBS), rep(MY_MODELS, times = N_WALKS*RNN_INSTANCES))
model = as.factor(model)
levels(model) = MY_LABELS
# merge to df
df <- data.frame(accuracy, agent, model)

ggplot(df) +
  aes(x = model,
      y = accuracy,
      fill = agent) + 
  geom_point(aes(color = agent), 
             position = position_jitterdodge(jitter.width = .15, # to sepreate jitter into their group.
                                             dodge.width = .3), 
             size = 5, 
             alpha = 0.2,
             show.legend = F) +
  geom_boxplot(aes(color = agent),
               width = .3, 
               outlier.shape = NA,
               alpha = 0) + 
  theme_bw() +
  scale_colour_manual(values=my_clrs_yct)+
  theme(legend.position = c(0.8, 0.8),
        legend.title = element_blank(),
        legend.text = element_text(size=20, face = "bold"),
        axis.title.x=element_blank(),
        axis.text.x =element_text(size=20, face="bold"),
        axis.title.y = element_text(size=20, face="bold"),
        axis.text.y = element_text(size = 20, face="bold")) + 
  ylab('Predictive accuracy') + 
  xlab(MY_LABELS)

# change directory and save plot
ggsave("plots/predictive_accuracy.png",   width = 7, height = 7, dpi = 600)

# to look at median values 
df %>%
  group_by(agent, model) %>%
  summarise(mean = mean(accuracy), sd = sd(accuracy))






