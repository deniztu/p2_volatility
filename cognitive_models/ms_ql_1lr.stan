data {
	int nTrials; //number of trials
	int<lower=1> nSubjects; // number of subjects
	int choice[nSubjects, nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nSubjects, nTrials]; // vector of rewards
}

transformed data {
  vector[4] initV;  // initial values for V for each arm
  initV = rep_vector(50, 4);
}

parameters {
  
  // learning rate
  real<lower=0,upper=1> alpha[nSubjects];

	// inverse temperature 
	real <lower=0> beta[nSubjects];
}


model {
  
  for (s in 1:nSubjects){
    
    vector[4] v[nTrials+1]; // value
    real pe[nSubjects, nTrials];       // prediction error
  
	  v[1] = initV;
	
	  for (t in 1:nTrials){
	    
	    if (choice[s,t] != 0) {
  	    
    	  // choice 
    		choice[s, t] ~ categorical_logit(beta[s] * v[t]);
    		 	
    		// prediction error
    		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
    		
	    }
  		
  	  // value updating (learning) 
      v[t+1] = v[t];
      
      if (choice[s,t] != 0) {
      
        v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
      
      }
	  }
  }
}
  

generated quantities {
  real log_lik[nSubjects, nTrials];
  int predicted_choices[nSubjects, nTrials];
  vector[4] v[nTrials+1]; // value
  real pe[nSubjects, nTrials];       // prediction error

	for (s in 1:nSubjects){
	  
  	v[1] = initV;

  	for (t in 1:nTrials){
  	  
  	  if (choice[s,t] != 0) {
  	  
    	  // choice 
    		log_lik[s, t] = categorical_logit_lpmf(choice[s, t] | beta[s] * v[t]);
    		predicted_choices[s, t] = categorical_logit_rng(beta[s] * v[t]);
    		 	
    		// prediction error
    		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
  	  
  	  }	
  		
  	  // value updating (learning) 
      v[t+1] = v[t];
      
      if (choice[s,t] != 0) {
      
      v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
  	  
  	  }
  	}
  }
}

