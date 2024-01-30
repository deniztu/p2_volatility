#############################################################################
# script computes bayesian t-test based on the cumulative regret of agents  #
# and and saves the effect size distribution                                #
#############################################################################

# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../data/intermediate_data/jasp_analysis/')

library(ggplot2)
library(BayesFactor)

# set global text size in plots
theme_update(text = element_text(size=20))


### t-test: regret test
df = read.csv(file = 'regret_t_test_data.csv')

ttestBF(formula = cum_reg ~ agent, data = df)

samples = ttestBF(formula = cum_reg ~ agent, data = df, posterior = TRUE,  iterations = 4000)

# plot posterior of effect size delta
# p = ggplot(data.frame(samples), aes(x=delta))+
#   geom_density(color = 'grey', fill="grey") + 
#   geom_vline(xintercept = 0, linetype="dashed", 
#              color = 'black', size=1) +
#   ylab("Posterior Density") +
#   xlab(bquote('Effect size '~.(bquote(delta)))) +
#   theme_bw() + 
#   theme(text = element_text(size = 20),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black"))
# 
# ggsave(filename = '../../../plots/figure_2_B.png', p,
#        width = 7, height = 7,
#        dpi = 600, units = "in",
#        device='png')

# write csv file
df = data.frame(samples[,'delta'])
write.csv(df, "regret_t_test_delta_posterior.csv")

# median
# median(samples[,'delta'])

