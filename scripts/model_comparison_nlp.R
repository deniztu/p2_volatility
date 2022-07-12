##############################################################
# script does model comparison with negative log-probability #
##############################################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(loo)
library(rstan)
library(Hmisc)
library(dplyr)
library(tidyverse)
library(PupillometryR)

# CONFIG

path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
file_string = 'stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'

model_ids = c(1:9)
ids = c(1:31)
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


########################################
# get negative log-probability         #
########################################

# prepare matrix with cols: nlp, id, model
nlp_matrix = array(NA ,dim = c(n_walks*length(ids)*length(model_ids), 3, n_samples, n_trials))
# prepare vectors
nlp = c()



for (m in c(1:length(model_ids))){
  
  for (walk in c(1:n_walks)){
    
    for (i in ids){
      
      # load stanfit and get predicted choices
      file = sprintf(file_string ,model_ids[m], i)#,walk)
      load(paste0(path_to_stanfit, file))
      
      # get log-likelihood
      ll = extract_log_lik(stanfit$stanfit, parameter_name = "log_lik", merge_chains = TRUE)
      
      # negative log-likelihood
      neg_ll = -1*ll
      
      # median of ll over samples for each trial
      medians = apply(neg_ll,2, median, na.rm=TRUE)
      
      nlp = c(nlp, medians)
      
      } 
  }
}

# create vectors for id and model cols
id =  rep(rep(ids, each=n_trials), times = length(model_ids))
model = rep(model_names, each=n_trials * length(ids))
# create final df for plotting
nlp_df = as.data.frame(list(nlp = nlp, id = id, model = model))

##################################
# plot median nlp for each model #
##################################

# calculate mean nlp for each model and each id (over trials)

medians_by_id_m= nlp_df %>% group_by(id,model)  %>% summarise(median(nlp, na.rm=TRUE))

medians_by_id_m

names(medians_by_id_m)[3] = 'nlp'

# nlp plot
ggplot(medians_by_id_m, aes(x=model, y=nlp, fill = model)) + 
  geom_flat_violin(position = position_nudge(x = .2, y = 0)) +
  geom_jitter(width=0.15)+ stat_summary(fun=median, geom="point", size=2, color="red")


###########################################################
# plot median probability of chosen action for each model #
###########################################################

# exp(-1*nlp) = porb of chosen action
medians_by_id_m['prob_chosen_action'] = exp(-1*medians_by_id_m['nlp'])

# plot
ggplot(medians_by_id_m, aes(x=model, y=prob_chosen_action, fill = model)) + 
  geom_flat_violin(position = position_nudge(x = .2, y = 0)) +
  geom_jitter(width=0.15)+ stat_summary(fun=median, geom="point", size=2, color="red")

