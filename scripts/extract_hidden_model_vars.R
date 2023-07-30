# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../data/intermediate_data/modeling/modeling_fits/')

### CONFIG

RNN_INSTANCES = 30
N_WALKS = 3
MY_MODELS = c(15)
RNN_STRING = 'stan_fit_m_%s_d_lstm2_a2c_nh_48_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_daw_p_%s_id_1.Rdata'

###

for (id in c(10)){
  for (w in c(1:N_WALKS)){
    for (m in c(1:length(MY_MODELS))){
      
      load(sprintf(RNN_STRING, MY_MODELS[m], id, w))
      
      x = as.data.frame(stanfit$stanfit)
      
      mean_paras = colMeans(x)
      
      ### collect variables
      res = matrix(NA, nrow = 300, ncol = 11)
      
      for (t in c(1:300)){
       
        res[t,1] = mean_paras[sprintf("v[1,1,%s]", t)]
        res[t,2] = mean_paras[sprintf("v[1,2,%s]", t)]
        res[t,3] = mean_paras[sprintf("v[1,3,%s]", t)]
        res[t,4] = mean_paras[sprintf("v[1,4,%s]", t)]
        
        res[t,5] = mean_paras[sprintf("sig[1,1,%s]", t)]
        res[t,6] = mean_paras[sprintf("sig[1,2,%s]", t)]
        res[t,7] = mean_paras[sprintf("sig[1,3,%s]", t)]
        res[t,8] = mean_paras[sprintf("sig[1,4,%s]", t)]
      }
      
      res[,9] = mean(x$`beta[1]`)
      res[,10] = mean(x$`phi[1]`)
      res[,11] = mean(x$`rho[1]`)
      
      df = data.frame(v1 = res[,1], v2 = res[,2], v3 = res[,3], v4 = res[,4],
                      sig1 = res[,5], sig2 = res[,6], sig3 = res[,7], sig4 = res[,8],
                      mean_beta = res[,9], mean_phi = res[,10], mean_rho = res[,11])
      
      write.csv(df, sprintf('id_%s_w_%s_m_%s_cog_vars.csv', id, w, MY_MODELS[m]))
      
    }
  }
}


