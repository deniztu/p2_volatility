data {
  int<lower=0> N;
  vector<lower=0>[N] x;
  vector<lower=0>[N] y;
}

parameters {
  real<lower=0> y0;
  real<lower=0> y_max;
  real k;
  real<lower=0> tau;
}

transformed parameters {
  real sigma;
  real m[N];
  for (i in 1:N) 
    m[i] = y0 + (y_max-y0)*(1-exp(-k*x[i]));
  
  sigma = 1 / sqrt(tau);
  
}


model {
  y0 ~ normal(0,10)T[0,];
  y_max ~ normal(0,10)T[0,];
  k ~ normal(0,10);
  tau ~ gamma(.0001, .0001);
  
  y ~ normal(m, sigma);
}

generated quantities{
  
  real Y_mean[N]; 
  real Y_pred[N]; 
  
  for(i in 1:N){
    // Posterior parameter distribution of the mean
    Y_mean[i] = y0 + (y_max-y0)*(1-exp(-k*x[i]));
    // Posterior predictive distribution
    Y_pred[i] = normal_rng(Y_mean[i], sigma);
    }
}

