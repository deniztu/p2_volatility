# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(rstan)
library(Hmisc)

# CONFIG

path_to_stanfit = 'data/intermediate_data/modeling/modeling_fits/'
file_string = 'stan_fit_m_8_d_lstm_a2c_nh_48_lr_0_0001_n_n_p_0_ew_0_05_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_8_test_b_daw_p_1.RData'

load(paste0(path_to_stanfit, file_string))

### plot performance and modeling variables

# load data
load(stanfit$data)

par(mfrow=c(3, 1))

# plot 1: reward walks & choices

my_cols = c('blue', 'red', 'green', 'orange')

col_points <- c()

for (a in res$choices[1,]){
  
  col_points <- c(col_points, my_cols[a]) 
}

plot(x = c(1:res$nTrials) , y = res$rewards$p_rew_1, type = 'l', col = my_cols[1], ylim = c(0,1), xlab = '', ylab = 'reward')
lines(x = c(1:res$nTrials) , y = res$rewards$p_rew_2, col = my_cols[2])
lines(x = c(1:res$nTrials) , y = res$rewards$p_rew_3, col = my_cols[3])
lines(x = c(1:res$nTrials) , y = res$rewards$p_rew_4, col = my_cols[4])
points(x = c(1:res$nTrials), rep(1, res$nTrials), col = col_points)

# plot 2 q-value

q_vals = rstan::extract(stanfit$stanfit, pars = 'v')
mean_q_vals = apply(q_vals$v[,,], c(2,3), mean)
mean_q_vals = mean_q_vals[c(1:res$nTrials), ]

plot(x = c(1:res$nTrials) , y = mean_q_vals[,1], type = 'l', col = my_cols[1], ylim = c(0,1), xlab = '', ylab = 'q value')
lines(x = c(1:res$nTrials) , y = mean_q_vals[,2], col = my_cols[2])
lines(x = c(1:res$nTrials) , y = mean_q_vals[,3], col = my_cols[3])
lines(x = c(1:res$nTrials) , y = mean_q_vals[,4], col = my_cols[4])
points(x = c(1:res$nTrials), rep(1, res$nTrials), col = col_points)


# plot 3 prediction error & switches
pes = rstan::extract(stanfit$stanfit, 'pe')
mean_pes = apply(pes$pe[,1,], 2, mean)

# get switch trials
res$choices[1,]
is_switch <- c(0)
t = 1
for (a in res$choices[1,]){
  if (t > 1){
    is_switch <- c(is_switch, prev_a != a)
  }
  prev_a = a
  t = t+1
}

lag1_pe = Lag(mean_pes, 1)

plot(x = c(1:res$nTrials) , y = Lag(lag1_pe, 1), type = 'l', lwd = 3, ylab = 'lag1 RPE')
abline(v = which(is_switch == 1), col = 'red', lwd = 0.5)

# plot(lag1_pe, is_switch)
# 
neg_pos = ifelse(lag1_pe < 0, '- rpe', '+ rpe')

df = data.frame(cbind(is_switch, neg_pos))
barplot(table(df)[2,], main = 'light grey = switch trials')


# plot 3 uncertainty

# eb = rstan::extract(stanfit$stanfit, pars = 'eb')
# mean_eb = apply(ev$v[,,], c(2,3), mean)
# mean_q_vals = mean_q_vals[c(1:res$nTrials), ]
# 
# plot(x = c(1:res$nTrials) , y = mean_q_vals[,1], type = 'l', col = my_cols[1], ylim = c(0,1), xlab = '', ylab = 'reward')
# lines(x = c(1:res$nTrials) , y = mean_q_vals[,2], col = my_cols[2])
# lines(x = c(1:res$nTrials) , y = mean_q_vals[,3], col = my_cols[3])
# lines(x = c(1:res$nTrials) , y = mean_q_vals[,4], col = my_cols[4])
# points(x = c(1:res$nTrials), rep(1, res$nTrials), col = col_points)



