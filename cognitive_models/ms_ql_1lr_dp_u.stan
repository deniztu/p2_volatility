data {
	int nTrials; //number of trials
	int<lower=1> nSubjects; // number of subjects
	int choice[nSubjects, nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nSubjects, nTrials]; // vector of rewards
	real<lower=0> uni[nSubjects,nTrials,4]; // unique bandit predictor
}

transformed data {
  vector[4] initV;  // initial values for V for each arm
  vector[4] initH;  // initial values for H (recency weighted average) for each arm
  initV = rep_vector(50, 4);
  initH = rep_vector(0, 4);
}

parameters {
  
  // learning rate
  real<lower=0,upper=1> alpha[nSubjects];

	// inverse temperature 
	real <lower=0> beta[nSubjects];
	
	// perseveration stepsize
	real <lower=0,upper=1> alpha_h[nSubjects];
	
	// perseveration weight
	real rho[nSubjects];
	  
	// exploration bonus weight
	real phi[nSubjects];
}

model {
  
  phi[nSubjects] ~ normal(0,10);
  rho[nSubjects] ~ normal(0,10);
  
  for (s in 1:nSubjects){
    
    vector[4] v[nTrials+1]; // value
    real pe[nSubjects, nTrials];       // prediction error
    vector[4] h[nTrials+1]; // recency weighted perseveration
    vector[4] pb;  // perseveration bonus
    vector[4] eb;  // exploration bonus
  
  
	  v[1] = initV;
	  h[1] = initH;
	  eb = rep_vector(0, 4);
	
	  for (t in 1:nTrials){
	    
	    if (choice[s,t] != 0) {

  	    if (t>1) {
  	      for(i in 1:4){
  	        eb[i] = phi[s] * uni[s,t-1,i];
  	        }
	        }
  	    
    	  // choice 
    		choice[s, t] ~ categorical_logit(beta[s] * (v[t] + eb + rho[s]*h[t]));
    		 	
    		// prediction error
    		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
  		
	    }
	    	    
  	  // value updating (learning) 
      v[t+1] = v[t];
      h[t+1] = h[t];
      
      if (choice[s,t] != 0) {
          
        v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
        
        // recency weighted perseveration
        pb = rep_vector(0.0, 4);
        pb[choice[s, t]] = 1;
        
        h[t+1] = h[t] + alpha_h[s]*(pb - h[t]);
          
      }
    }
  }
}

generated quantities {
  real log_lik[nSubjects, nTrials];
  int predicted_choices[nSubjects, nTrials];
  vector[4] v[nTrials+1]; // value
  real pe[nSubjects, nTrials];       // prediction error
  vector[4] h[nTrials+1]; // recency weighted perseveration
  vector[4] pb;  // perseveration bonus
  vector[4] eb;  // exploration bonus


	for (s in 1:nSubjects){

  	v[1] = initV;
  	h[1] = initH;
  	eb = rep_vector(0, 4);

  	for (t in 1:nTrials){
  	  
  	  if (choice[s,t] != 0) {

        if (t>1) {
          for(i in 1:4){
            eb[i] = phi[s] * uni[s,t-1,i];
          }
        }

  	  // choice
  		log_lik[s, t] = categorical_logit_lpmf(choice[s, t] | beta[s] * (v[t] + eb + rho[s]*h[t]));
  		predicted_choices[s, t] = categorical_logit_rng(beta[s] * (v[t] + eb + rho[s]*h[t]));

  		// prediction error
  		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
  		
  	  }

  	  // value updating (learning) 
      v[t+1] = v[t];
      h[t+1] = h[t];
      
      if (choice[s,t] != 0) {
          
        v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
        
        // recency weighted perseveration
        pb = rep_vector(0.0, 4);
        pb[choice[s, t]] = 1;
        
        h[t+1] = h[t] + alpha_h[s]*(pb - h[t]);
          
      }
  	}
  }
}
