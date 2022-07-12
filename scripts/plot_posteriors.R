# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load libraries
library('bayesplot')
library('ggplot2')

####################
# plot posteriors  #
####################

m = 9
subjects = c(1:31)
my_stanfit = 'C:/Users/Bio_Psych/Desktop/Deniz/tf-cpu/p1_generalization/data/intermediate_data/modeling/modeling_fits/stan_fit_m_%s_d_chakroun_placebo_human_bandit_data_id_%s.RData'
my_par = 'alpha_h'

plot_ss_posteriors <- function(my_stanfit, my_par, subjects, m, colapse_walks = 0){
  
  i = 0
  if (colapse_walks > 0){
    
    for (w in c(1:colapse_walks)){
      
      for (s in subjects){
        
        # load stanfit
        load(sprintf(my_stanfit, m, s, w))
        
        # extract
        mcmc = rstan::extract(stanfit$stanfit, pars = my_par)
        mcmc = as.matrix(unlist(mcmc))
        colnames(mcmc) = paste0(my_par, sprintf('_id_%s%s', s, w))
        
        tmp = mcmc_intervals_data(mcmc)
        tmp$model = sprintf('id_%s_w_%s', s, w)
        
        if (i == 0) {combined <- tmp}
        
        else {
          combined = rbind(combined, tmp)
        }
        
        i = i+1
      }
      
    }
  }
  
  else{
    
    for (s in subjects){
      
      # load stanfit
      load(sprintf(my_stanfit, m, s))
      
      # extract
      mcmc = rstan::extract(stanfit$stanfit, pars = my_par)
      mcmc = as.matrix(unlist(mcmc))
      colnames(mcmc) = paste0(my_par, sprintf('_id_%s', s))
      
      tmp = mcmc_intervals_data(mcmc)
      tmp$model = sprintf('id_%s', s)
      
      if (i == 0) {combined <- tmp}
      
      else {
        combined = rbind(combined, tmp)
      }
      
      i = i+1
    }
  }
  
  
  
  theme_set(bayesplot::theme_default())
  pos <- position_nudge(y = ifelse(combined$model == "Model 2", 0, 0.1))
  p = ggplot(combined, aes(x = m, y = parameter, color = model), show.legend = FALSE) + 
    geom_point(position = pos, show.legend = FALSE) +
    geom_linerange(aes(xmin = ll, xmax = hh), position = pos, show.legend = FALSE) +
    xlab(my_par) + 
    ylab('id')
  
  # no y_axis names
  p= p + theme(axis.text.y=element_blank())
  
  print(p)
  
  return(combined)
}

############
# call     #
############


my_stanfit_noisy_ew_lin = 'C:/Users/Bio_Psych/Desktop/Deniz/tf-cpu/p1_generalization/data/intermediate_data/modeling/modeling_fits/stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_lin_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.RData'
my_stanfit_noisy_ew_0 = 'C:/Users/Bio_Psych/Desktop/Deniz/tf-cpu/p1_generalization/data/intermediate_data/modeling/modeling_fits/stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.RData'


m1 = plot_ss_posteriors(my_stanfit, 'beta', subjects, m)
m2 = plot_ss_posteriors(my_stanfit_noisy_ew_lin, 'beta', c(0:19), m, colapse_walks = 3)
m3 = plot_ss_posteriors(my_stanfit_noisy_ew_0, 'beta', c(0:19), m, colapse_walks = 3)



#################
# violin plots  #
################

combined = rbind(m1,m2, m3)
combined$model = c(rep(c('placebo'), each = nrow(m1)), rep(c('noisy ew lin', 'noisy'), each = nrow(m2)))

p <- ggplot(combined, aes(x=model, y=m)) + 
  geom_violin(trim=FALSE) + ylab('median beta') + xlab('')

# violin plot with dot plot
p + geom_dotplot(binaxis='y', stackdir='center', dotsize=1) + stat_summary(fun.y=median, geom="point", size=5, color="red")

