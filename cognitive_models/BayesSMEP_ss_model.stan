data {
  int<lower=1> nTrials;               
  int<lower=0,upper=4> choice[nTrials];     
  real<lower=0, upper=100> reward[nTrials]; 
  }

transformed data {
  real<lower=0, upper=100> v1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  v1 = 50.0;
  sig1 = 4.0;
  sigO = 4.0;
  sigD = 2.8;
  decay = 0.9836;
  decay_center = 50;
}

parameters {
  real<lower=0,upper=3> beta;
  real phi;
  real rho;
}

model {
  
  vector[4] v;   // value (mu)
  vector[4] sig; // sigma
  real pe;       // prediction error
  real Kgain;    // Kalman gain
  vector[4] eb;  // exploration bonus
  vector[4] pb;  // perseveration bonus
  
  phi ~ uniform(-10,10); //added by deniz
  rho ~ uniform(-10,10); //added by deniz

  v = rep_vector(v1, 4);
  sig = rep_vector(sig1, 4);

  for (t in 1:nTrials) {        
  
    if (choice[t] != 0) {
      
      // phi: exploration bonus
      eb = phi * sig;
      
      // rho: perseveration bonus
      pb = rep_vector(0.0, 4);
      
      if (t>1) {
        if (choice[t-1] !=0) {
          pb[choice[t-1]] = rho;
        } 
      }
      
      choice[t] ~ categorical_logit( beta * (v + eb + pb));  // compute action probabilities
      
      pe = reward[t] - v[choice[t]];  # prediction error 
      Kgain = sig[choice[t]]^2 / (sig[choice[t]]^2 + sigO^2); # Kalman gain
      
      v[choice[t]] = v[choice[t]] + Kgain * pe;  # value/mu updating (learning)
      sig[choice[t]] = sqrt( (1-Kgain) * sig[choice[t]]^2 ); # sigma updating
    }
  
  v = decay * v + (1-decay) * decay_center;  
  for (j in 1:4) 
    sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
  #sig = sqrt( decay^2 * sig^2 + sigD^2 );  # no elementwise exponentiation in STAN!
  }
}

generated quantities{
  vector[nTrials] log_lik;
  int predicted_choices[nTrials];
    
  vector[4] v;   # value (mu)
  vector[4] sig; # sigma
  real pe;       # prediction error
  real Kgain;    # Kalman gain
  vector[4] eb;  // exploration bonus
  vector[4] pb;  // perseveration bonus

  v = rep_vector(v1, 4);
  sig = rep_vector(sig1, 4);

  for (t in 1:nTrials) {        
  
  if (choice[t] != 0) {
    
      // phi: exploration bonus
      eb = phi * sig;
      
      // rho: perseveration bonus
      pb = rep_vector(0.0, 4);
      
      if (t>1) {
        if (choice[t-1] !=0) {
          pb[choice[t-1]] = rho;
        } 
      }
    
    log_lik[t] = categorical_logit_lpmf(choice[t] | beta * (v + eb + pb));
    predicted_choices[t] = categorical_logit_rng(beta * (v + eb + pb));

    pe = reward[t] - v[choice[t]];  # prediction error 
    Kgain = sig[choice[t]]^2 / (sig[choice[t]]^2 + sigO^2); # Kalman gain
    
    v[choice[t]] = v[choice[t]] + Kgain * pe;  # value/mu updating (learning)
    sig[choice[t]] = sqrt( (1-Kgain) * sig[choice[t]]^2 ); # sigma updating
  }
  
  v = decay * v + (1-decay) * decay_center;  
  for (j in 1:4) 
    sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
  #sig = sqrt( decay^2 * sig^2 + sigD^2 );  # no elementwise exponentiation in STAN!
}
}

