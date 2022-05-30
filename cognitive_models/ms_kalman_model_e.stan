data {
	int nTrials; //number of trials
	int<lower=1> nSubjects; // number of subjects
	int choice[nSubjects, nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nSubjects, nTrials]; // vector of rewards
}

transformed data {
  real<lower=0, upper=100> v1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  v1 = 50;
  sig1 = 4;
  sigO = 4;
  sigD =  2.8;
  decay = 0.9836;
  decay_center = 50;
}

parameters {
  real beta[nSubjects];
  real phi[nSubjects];
}

model {
  
  vector[4] v;   // value (mu)
  vector[4] sig; // sigma
  real pe[nSubjects, nTrials];       // prediction error
  real Kgain;    // Kalman gain
  vector[4] eb;  // exploration bonus

  for (s in 1:nSubjects){

    v = rep_vector(v1, 4);
    sig = rep_vector(sig1, 4);
  
    for (t in 1:nTrials) {        
    
      if (choice[s, t] != 0) {
        
        // phi: exploration bonus
        eb = phi[s] * sig;
        
        choice[s, t] ~ categorical_logit( beta[s] * (v + eb));  // compute action probabilities
        
        pe[s, t] = reward[s, t] - v[choice[s, t]];  # prediction error 
        Kgain = sig[choice[s, t]]^2 / (sig[choice[s, t]]^2 + sigO^2); # Kalman gain
        
        v[choice[s, t]] = v[choice[s, t]] + Kgain * pe[s, t];  # value/mu updating (learning)
        sig[choice[s, t]] = sqrt( (1-Kgain) * sig[choice[s, t]]^2 ); # sigma updating
      }
    
    v = decay * v + (1-decay) * decay_center;  
    for (j in 1:4) 
      sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
    #sig = sqrt( decay^2 * sig^2 + sigD^2 );  # no elementwise exponentiation in STAN!
    }
  }
}

generated quantities{
  real log_lik[nSubjects, nTrials];
  int predicted_choices[nSubjects, nTrials];
    
  vector[4] v;   # value (mu)
  vector[4] sig; # sigma
  real pe[nSubjects, nTrials];       // prediction error
  real Kgain;    # Kalman gain
  vector[4] eb;  // exploration bonus
  
  for (s in 1:nSubjects){
  
    v = rep_vector(v1, 4);
    sig = rep_vector(sig1, 4);
  
    for (t in 1:nTrials) {        
    
    if (choice[s, t] != 0) {
    
      // phi: exploration bonus
      eb = phi[s] * sig;
        
      log_lik[s, t] = categorical_logit_lpmf(choice[s, t] | beta[s] * (v + eb));
      predicted_choices[s, t] = categorical_logit_rng(beta[s] * (v + eb));
  
      pe[s, t] = reward[s, t] - v[choice[s, t]];  // prediction error 
      Kgain = sig[choice[s, t]]^2 / (sig[choice[s, t]]^2 + sigO^2); # Kalman gain
      
      v[choice[s, t]] = v[choice[s, t]] + Kgain * pe[s, t];  # value/mu updating (learning)
      sig[choice[s, t]] = sqrt( (1-Kgain) * sig[choice[s, t]]^2 ); # sigma updating
    }
    
    v = decay * v + (1-decay) * decay_center;  
    for (j in 1:4) 
      sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
    #sig = sqrt( decay^2 * sig^2 + sigD^2 );  # no elementwise exponentiation in STAN!
  }
  }
}

