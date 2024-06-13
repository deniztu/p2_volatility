### functions to fit maximum-likelihood RL models

#---------------------------#
# single subject q learning #
#---------------------------#
ql = function(choice, reward, alpha=0.5, v0 = 50, n_actions = 4){
  
  n_trials = length(choice)
  v = matrix(v0, ncol = n_actions, nrow = n_trials+1)
  pe = rep(0, n_trials)
  
  for(t in 1:n_trials){
    # calculate prediction error
    pe[t] = reward[t] - v[t, choice[t]]
    
    # value updating
    v[t+1,] = v[t,]
    
    v[t+1, choice[t]] = v[t, choice[t]] + alpha * pe[t]
  }
  
  return(list(v = v, pe = pe))
}

# softmax function for q_learning
softmax_ql = function(v, beta=0.5){
  
  prob = exp(beta*v)
  prob = prob/rowSums(prob)

  return(prob)
}

# negative loglikelihood for q_learning

sm_neg_log_lik_ql = function(par, data){
  
  alpha = par[1]
  beta = par[2]
  choice = data$choice
  reward = data$reward
  vals = ql(choice, reward, alpha)
  p = softmax_ql(v = vals$v, beta)
  lik = p[cbind(seq_along(choice), choice)]
  neg_log_lik = -sum(log(lik))
  
  # print(neg_log_lik)
  # print(alpha)
  # print(beta)
  
  return(neg_log_lik)
}


#-----------------------------------------------------------#
# single subject q learning with higher-order perseveration #
#-----------------------------------------------------------#

ql_hop = function(choice, reward, alpha=0.5, alpha_h = 0.5,
                          v0 = 0.5, h0 = 0, n_actions = 4){
  
  n_trials = length(choice)
  v = matrix(v0, ncol = n_actions, nrow = n_trials+1)
  h = matrix(h0, ncol = n_actions, nrow = n_trials+1)
  pe = rep(0, n_trials)
  
  for(t in 1:n_trials){
    
    
    # calculate prediction error
    pe[t] = reward[t] - v[t, choice[t]]
    
    # value updating
    v[t+1,] = v[t,]
    
    v[t+1, choice[t]] = v[t, choice[t]] + alpha * pe[t]
    
    # higher-order perseveration
    h[t+1,] = h[t,]
    pb = rep(0, n_actions) 
    pb[[choice[t]]] = 1
    h[t+1,] = h[t,] + alpha_h * (pb - h[t,])
  }
  
  return(list(v = v, h = h, pe = pe))
}

# softmax function for q_learning with higher-order perseveration
softmax_ql_hop = function(v, h, rho = 0, beta=0.5){
  
  prob = exp(beta*(v+rho*h))
  prob = prob/rowSums(prob)

  return(prob)
}

# negative loglikelihood for q_learning with higher-order perseveration

sm_neg_log_lik_ql_hop = function(par, data){
  
  alpha = par[1]
  alpha_h = par[2]
  rho = par[3]
  beta = par[4]
  choice = data$choice
  reward = data$reward
  vals = ql_hop(choice, reward, alpha, alpha_h)
  p = softmax_ql_hop(v = vals$v, h = vals$h, rho, beta)
  lik = p[cbind(seq_along(choice), choice)]
  neg_log_lik = -sum(log(lik))
  
  # print('alpha')
  # print(alpha)
  # print('alpha_h')
  # print(alpha_h)
  # print('rho')
  # print(rho)
  # print('beta')
  # print(beta)
  # print('neg_log_lik')
  # print(neg_log_lik)

  return(neg_log_lik)
}

# optimization
fit = function(model, inits = c(0.5, 0.5), df, lower_bounds= c(0, 0), upper_bounds = c(1, Inf)){
  
  if (model=='q_learning'){
    loss = sm_neg_log_lik_ql
  }
  if (model=='q_learning_hop'){
    loss = sm_neg_log_lik_ql_hop
  }
  if (model=='q_learning_hop_e'){
    loss = sm_neg_log_lik_ql_hop_e
  }
  
  est = optim(par = inits, loss, data = df, lower = lower_bounds, upper = upper_bounds)
  return(est)
}

