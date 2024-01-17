data {
  int<lower=0> N;
  vector<lower=0>[N] x;
  vector<lower=0>[N] y;
}

parameters {
  real<lower=0, upper=1> y0;
  real<lower=0, upper=1> a;
  real<lower=0> k;
  real<lower=0> tau;
}

transformed parameters {
  real sigma;
  real m[N];
  for (i in 1:N) 
    m[i] = (y0 - a)*exp(-k*x[i]) + a;
  
  sigma = 1 / sqrt(tau);
  
}


model {
  y0 ~ uniform(0,1);
  a ~ uniform(0,1);
  k ~ normal(0,10)T[0,];
  tau ~ gamma(.0001, .0001);
  
  y ~ normal(m, sigma);
}

// generated quantities{
//   
//   real Y_mean[N]; 
//   real Y_pred[N]; 
//   
//   for(i in 1:N){
//     // Posterior parameter distribution of the mean
//     Y_mean[i] = (y0 - a)*exp(-k*x[i]) + a;
//     // Posterior predictive distribution
//     Y_pred[i] = normal_rng(Y_mean[i], sigma);
//     }
// }

