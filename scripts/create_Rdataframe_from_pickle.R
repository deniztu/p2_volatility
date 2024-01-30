#####################################################################
### script converts multiple pickled dataframes to one RDataframe ###
#####################################################################


# set working dir to dir where R-file resides
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path)))

# move to relevant directory
setwd('../data/rnn_raw_data/')

require("reticulate")
require("plyr")

pd <- import("pandas")

### CONFIG

file_name <- 'lstm2_a2c_nh_%s_lr_0_0001_n_u_p_0_5_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_%s_test_b_res_p_0_1'



# hidden unit factors
nh_factors <- c(48, 64, 80, 96)
#n_walks <- c(1:30)
ids <- c(0:29)

###

pickle_to_df <- function(file_name, id, nh){
  
  pickle_file <- sprintf(file_name, nh, id)
  pickle_data <- pd$read_pickle(pickle_file)
  pickle_data$id <- rep(id, nrow(pickle_data))
  pickle_data$n_hidden <- rep(nh, nrow(pickle_data))
  # add run indicator
  pickle_data$run <- rep(c(0:29), each = 300)
  
  
  return(pickle_data)
} 

# initialize empty dataframe
max_df <- pickle_to_df(file_name, id = 1, nh = max(nh_factors))
my_ncols <- dim(max_df)[2]
my_colnames <- colnames(max_df)

rnn_data = data.frame(matrix(nrow = 0, ncol = my_ncols)) 
colnames(rnn_data) <- my_colnames



for (nh in nh_factors){
  for (id in ids){
    #for (w in n_walks){
    
      
      rnn_data <- rbind.fill(rnn_data, pickle_to_df(file_name, id, nh))
      # else{
      #   df <- pickle_to_df(file_name, id, 96, w) 
      # }
      #     
      # i = i + 1  
    #}
  }
}

save(rnn_data, file = 'rnn_data_nh_factors_80s_trained.Rdata')

