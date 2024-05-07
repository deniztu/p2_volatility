#####################################
# Parameter recovery                #
#####################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(rstan)
library(truncnorm)
library(ggplot2)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# set seed for reproducibility
set.seed(1245433)


##########
# CONFIG #
##########

nSubjects = 5
nTrials = 300

# load daw walks
daw_walk_1 = read.csv('classes/bandits/Daw2006_payoffs1.csv')[,-1]
# daw_walk_2 = read.csv('classes/bandits/Daw2006_payoffs2.csv')[,-1]
# daw_walk_3 = read.csv('classes/bandits/Daw2006_payoffs3.csv')[,-1]

R = daw_walk_1[c(1:nTrials),]

##############
# Simulation #
##############

# parameter range for simulations (eyballing from RNNExplore Figure 4)
beta_in = runif(nSubjects, min = 0.03, max = 0.35)
phi_in = runif(nSubjects, min = -2, max = 6)
rho_in = runif(nSubjects, min = -4, max = 30)
alpha_h_in = runif(nSubjects, min = 0.03, max = 0.95)

# create data input
data_in <- list(nSubjects = nSubjects, nTrials= nTrials,
                beta= as.array(beta_in),
                phi= as.array(phi_in),
                rho = as.array(rho_in),
                alpha_h = as.array(alpha_h_in),
                reward = R)

save(data_in, file = 'data/intermediate_data/modeling/parameter_recovery_inputs.RData')

# simulate data
SMEDP<- rstan::stan_model(file= 'cognitive_models/ms_kalman_model_e_dp_fixed_param.stan') 
simSMEDP <- rstan::sampling(SMEDP, data = data_in, chains = 1, iter = 1, algorithm='Fixed_param')

# get choices and rewards
ex.simSMEDP <- extract(simSMEDP, permuted = TRUE)
choice <- ex.simSMEDP$choice[1,,]
reward <- ex.simSMEDP$reward_obt[1,,]

##############
# Fitting    #
##############

data_fit <- list(nSubjects = nSubjects, nTrials= nTrials, choice= choice, reward= reward)

# fit
fit_SMEDP <- stan(file= 'cognitive_models/ms_kalman_model_e_dp_no_gen.stan',
                  data = data_fit,
                  chains = 2,
                  cores = getOption("mc.cores", 1L),
                  iter = 2000)

save(fit_SMEDP, file = 'data/intermediate_data/modeling/parameter_recovery_fits.RData')




