###################################
# Scatterplots Parameter Recovery #
###################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move up two directories
setwd('../')

# load packages
library(rstan)
library(ggplot2)
library(cowplot)


#############
# Config    #
#############

label_size = 10
my_y_lab = 'Estimated'
my_x_lab = 'True'

#############
# load data #
#############

# loads 'data_in' object
load('data/intermediate_data/modeling/parameter_recovery_inputs.RData')
# loads 'fit_SMEDP' object
load('data/intermediate_data/modeling/parameter_recovery_fits.RData')

##############
# plotting   #
##############

mcmc = extract(fit_SMEDP, permuted = TRUE)

fitted_beta = colMeans(mcmc$beta)
fitted_phi = colMeans(mcmc$phi)
fitted_rho = colMeans(mcmc$rho)
fitted_alpha_h = colMeans(mcmc$alpha_h)

true_beta = data_in$beta
true_phi = data_in$phi
true_rho = data_in$rho
true_alpha_h = data_in$alpha_h


df = data.frame(fitted_beta,
                fitted_phi,
                fitted_rho,
                fitted_alpha_h,
                true_beta,
                true_phi,
                true_rho,
                true_alpha_h)

# correlations
round(cor(true_beta, fitted_beta), 3)
round(cor(true_phi, fitted_phi), 3)
round(cor(true_rho, fitted_rho), 3)
round(cor(true_alpha_h, fitted_alpha_h), 3)

# comparing beta
p1 = ggplot(df, aes(x = true_beta, y = fitted_beta))+ 
  geom_point(size=2.5)+
  geom_abline(intercept = 0, slope = 1, colour= 'red') + 
  xlim(min(true_beta, fitted_beta), max(true_beta, fitted_beta)) + 
  ylim(min(true_beta, fitted_beta), max(true_beta, fitted_beta))+
  theme_bw() +
  theme(plot.title =element_text(hjust = 0.5, size=label_size, face="bold"), 
        axis.title.x = element_text(size=label_size, face="bold"),
        axis.text.x =element_text(size=label_size, face="bold"),
        axis.title.y = element_text(size=label_size, face="bold"),
        axis.text.y = element_text(size = label_size, face="bold"))+
  ylab(my_y_lab)+
  xlab(my_x_lab)+ 
  ggtitle('Beta')

# comparing phi
p2 = ggplot(df, aes(x = true_phi, y = fitted_phi))+ 
  geom_point(size=2.5)+
  geom_abline(intercept = 0, slope = 1, colour= 'red') + 
  xlim(min(true_phi, fitted_phi), max(true_phi, fitted_phi)) + 
  ylim(min(true_phi, fitted_phi), max(true_phi, fitted_phi))+
  theme_bw() +
  theme(plot.title =element_text(hjust = 0.5, size=label_size, face="bold"), 
        axis.title.x = element_text(size=label_size, face="bold"),
        axis.text.x =element_text(size=label_size, face="bold"),
        axis.title.y = element_text(size=label_size, face="bold"),
        axis.text.y = element_text(size = label_size, face="bold"))+
  ylab(my_y_lab)+
  xlab(my_x_lab)+ 
  ggtitle('Phi')

# comparing rho
p3 = ggplot(df, aes(x = true_rho, y = fitted_rho))+ 
  geom_point(size=2.5)+
  geom_abline(intercept = 0, slope = 1, colour= 'red') + 
  xlim(min(true_rho, fitted_rho), max(true_rho, fitted_rho)) + 
  ylim(min(true_rho, fitted_rho), max(true_rho, fitted_rho))+
  theme_bw() +
  theme(plot.title =element_text(hjust = 0.5, size=label_size, face="bold"), 
        axis.title.x = element_text(size=label_size, face="bold"),
        axis.text.x =element_text(size=label_size, face="bold"),
        axis.title.y = element_text(size=label_size, face="bold"),
        axis.text.y = element_text(size = label_size, face="bold"))+
  ylab(my_y_lab)+
  xlab(my_x_lab)+ 
  ggtitle('Rho')

# comparing alpha_h
p4 = ggplot(df, aes(x = true_alpha_h, y = fitted_alpha_h))+ 
  geom_point(size=2.5)+
  geom_abline(intercept = 0, slope = 1, colour= 'red') + 
  xlim(min(true_alpha_h, fitted_alpha_h), max(true_alpha_h, fitted_alpha_h)) + 
  ylim(min(true_alpha_h, fitted_alpha_h), max(true_alpha_h, fitted_alpha_h))+
  theme_bw() +
  theme(plot.title =element_text(hjust = 0.5, size=label_size, face="bold"), 
        axis.title.x = element_text(size=label_size, face="bold"),
        axis.text.x =element_text(size=label_size, face="bold"),
        axis.title.y = element_text(size=label_size, face="bold"),
        axis.text.y = element_text(size = label_size, face="bold"))+
  ylab(my_y_lab)+
  xlab(my_x_lab)+ 
  ggtitle('Alpha H')

# plot into grid
plot_grid(p1, p2,p3,p4, nrow = 2)

