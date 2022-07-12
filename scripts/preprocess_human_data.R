###################################
# script preprocesses human data  #
###################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

#####################
# Wiehler data      #
#####################

preprocessed_file_name_ = 'wiehler_control_human_bandit_data'
path_to_save_formatted_data = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'

load('data/human_raw_data/pgbandit_bandit_data_controls.Rdata')

# nRuns = number of subjects
nRuns = length(bandit_data_controls$subject_id)

# choices 
choices = bandit_data_controls$choices

# chosen rewards
chosen_rewards = bandit_data_controls$rewards
# transform for rnn compatibility (change later)
chosen_rewards = chosen_rewards/100

# nTrials
nTrials = dim(bandit_data_controls$choices)[2]

# rewards with noise
# rewards = bandit_data_controls$payouts

res = list(model = preprocessed_file_name_, choices = choices, chosen_rewards = chosen_rewards
           , nRuns = nRuns, nTrials = nTrials)

save(file = sprintf('%s/pp_data_%s.RData', path_to_save_formatted_data, preprocessed_file_name_),res)

#####################
# Chakroun data     #
#####################

preprocessed_file_name_ = 'chakroun_placebo_human_bandit_data'

load("data/human_raw_data/Reg_BayesSMEP_mp_hrch_3hp_pFix_Karima01_Plac_KC.Rdata")

# nRuns = number of subjects
nRuns = dim(Reg$ch)[1]

# choices
choices = Reg$ch

# chosen rewards
chosen_rewards = Reg$rew
# transform for rnn compatibility (change later)
chosen_rewards = chosen_rewards/100

# nTrials
nTrials = dim(Reg$ch)[2]

res = list(model = preprocessed_file_name_, choices = choices, chosen_rewards = chosen_rewards
           , nRuns = nRuns, nTrials = nTrials)

save(file = sprintf('%s/pp_data_%s.RData', path_to_save_formatted_data, preprocessed_file_name_),res)

