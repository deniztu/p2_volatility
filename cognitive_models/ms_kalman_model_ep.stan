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
  real<lower=0> beta[nSubjects];
  real phi[nSubjects];
  real rho[nSubjects];
}

model {
  
  vector[4] v;   // value (mu)
  vector[4] sig; // sigma
  real pe;       // prediction error
  real Kgain;    // Kalman gain
  vector[4] eb;  // exploration bonus
  vector[4] pb;  // perseveration bonus
  
  for (s in 1:nSubjects){
  
    v = rep_vector(v1, 4);
    sig = rep_vector(sig1, 4);
  
    for (t in 1:nTrials) {        
    
      if (choice[s, t] != 0) {
        
        // phi: exploration bonus
        eb = phi[s] * sig;
        
        // rho: perseveration bonus
        pb = rep_vector(0.0, 4);
        
        if (t>1) {
          if (choice[s, t-1] !=0) {
            pb[choice[s, t-1]] = rho[s];
          } 
        }
        
        choice[s, t] ~ categorical_logit( beta[s] * (v + eb + pb));  // compute action probabilities
        
        pe = reward[s, t] - v[choice[s, t]];  // prediction error 
        Kgain = sig[choice[s, t]]^2 / (sig[choice[s, t]]^2 + sigO^2); // Kalman gain
        
        v[choice[s, t]] = v[choice[s, t]] + Kgain * pe;  // value/mu updating (learning)
        sig[choice[s, t]] = sqrt( (1-Kgain) * sig[choice[s, t]]^2 ); // sigma updating
      }
    
    v = decay * v + (1-decay) * decay_center;  
    for (j in 1:4) 
      sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
    //sig = sqrt( decay^2 * sig^2 + sigD^2 );  // no elementwise exponentiation in STAN!
    }
  }
}

generated quantities{
  real log_lik[nSubjects, nTrials];
  int predicted_choices[nSubjects, nTrials];

  matrix[4, nTrials+1] v[nSubjects];   // value (mu)
  matrix[4, nTrials+1] sig[nSubjects]; // sigma
  real pe[nSubjects, nTrials];       // prediction error
  real Kgain;    // Kalman gain
  vector[4] eb;  // exploration bonus
  vector[4] pb;  // perseveration bonus

	for (s in 1:nSubjects){

    v[s][,1] = rep_vector(v1, 4);
    sig[s][,1] = rep_vector(sig1, 4);

    for (t in 1:nTrials) {

    if (choice[s, t] != 0) {

        // phi: exploration bonus
        eb = phi[s] * sig[s][,t];

        // rho: perseveration bonus
        pb = rep_vector(0.0, 4);

        if (t>1) {
          if (choice[s, t-1] !=0) {
            pb[choice[s, t-1]] = rho[s];
          }
        }
      
      log_lik[s, t] = categorical_logit_lpmf(choice[s, t] | beta[s] * (v[s][,t] + eb + pb));
      
      // new softmax functions
      // print("sm")
      // print(softmax(beta[s] * (v[s][,t] + eb + pb)))
      // print("sm log")
      // print(log_softmax(beta[s] * (v[s][,t] + eb + pb)))
      
      
      predicted_choices[s, t] = categorical_logit_rng(beta[s] * (v[s][,t] + eb + pb));

      pe[s, t] = reward[s, t] - v[s][choice[s, t],t];  // prediction error
      
      Kgain = sig[s][choice[s, t],t]^2 / (sig[s][choice[s, t],t]^2 + sigO^2); // Kalman gain
      
      v[s][,t+1] = v[s][,t]; // move current v to next trial 
      v[s][choice[s, t], t+1] = v[s][choice[s, t],t] + Kgain * pe[s, t];  // value/mu updating (learning)
      
      sig[s][,t+1] = sig[s][,t]; // move current sig to next trial
      sig[s][choice[s, t], t+1] = sqrt( (1-Kgain) * sig[s][choice[s, t],t]^2 ); // sigma updating
      
    }

    v[s][,t+1] = decay * v[s][,t+1] + (1-decay) * decay_center;
    for (j in 1:4)
      sig[s][j,t+1] = sqrt( decay^2 * sig[s][j,t+1]^2 + sigD^2 );

  }
  }
}

