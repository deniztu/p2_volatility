data {
	int nTrials; //number of trials
	int<lower=1> nSubjects; // number of subjects
	real<lower=0, upper=100> reward[nTrials,4]; // vector of rewards
	  
	real<lower=0, upper=3> beta[nSubjects];  
  real phi[nSubjects];
  real rho[nSubjects];
  real<lower=0, upper=1> alpha_h[nSubjects];
}

transformed data {
  real<lower=0, upper=100> v1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  real initH;  // initial values for H (recency weighted average) for each arm
  
  v1 = 50;
  sig1 = 4;
  sigO = 4;
  sigD =  2.8;
  decay = 0.9836;
  decay_center = 50;
  initH = 0;
}

parameters {

}

model {
  

}

generated quantities{
  
  int choice[nSubjects, nTrials];
  real reward_obt[nSubjects, nTrials];
  
  vector[4] v;   // value (mu)
  vector[4] h; // recency weighted perseveration
  vector[4] sig; // sigma
  real pe;       // prediction error
  real Kgain;    // Kalman gain
  vector[4] eb;  // exploration bonus
  vector[4] pb;  // perseveration bonus
  
  for (s in 1:nSubjects){
  
    v = rep_vector(v1, 4);
    h = rep_vector(initH, 4);
    sig = rep_vector(sig1, 4);
    
    for (t in 1:nTrials) {        
    
      if (choice[s, t] != 0) {
        
        // phi: exploration bonus
        eb = phi[s] * sig;
        
        choice[s, t] = categorical_logit_rng( beta[s] * (v + eb + rho[s]*h));  // compute action probabilities
        reward_obt[s,t] = reward[t,choice[s,t]];
        
        pe = reward_obt[s,t] - v[choice[s, t]];  // prediction error 
        Kgain = sig[choice[s, t]]^2 / (sig[choice[s, t]]^2 + sigO^2); // Kalman gain
        
        v[choice[s, t]] = v[choice[s, t]] + Kgain * pe;  // value/mu updating (learning)
        sig[choice[s, t]] = sqrt( (1-Kgain) * sig[choice[s, t]]^2 ); // sigma updating
      
        // recency weighted perseveration
        pb = rep_vector(0.0, 4);
        pb[choice[s, t]] = 1;
        
        h = h + alpha_h[s]*(pb - h);
      
      }
    
    v = decay * v + (1-decay) * decay_center;  
    for (j in 1:4) 
      sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
    //sig = sqrt( decay^2 * sig^2 + sigD^2 );  // no elementwise exponentiation in STAN!
    }
  }

}

