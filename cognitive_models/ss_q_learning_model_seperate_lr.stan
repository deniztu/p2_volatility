data {
	int nTrials; //number of trials
	int choice[nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nTrials]; // vector of rewards
}

transformed data {
  vector[4] initV;  // initial values for V for each arm
  initV = rep_vector(50.0, 4);
}

parameters {
  
  // learning rate for positive RPEs
  real<lower=0,upper=1> alpha_pos_rpe;
  
  // learning rate for negative RPEs
  real<lower=0,upper=1> alpha_neg_rpe;
	
	// inverse temperature 
	real beta;
}

model {
  vector[4] v[nTrials+1]; // value
  real pe[nTrials];       // prediction error
  
  // priors
  alpha_pos_rpe ~ beta(2,2);
  alpha_neg_rpe ~ beta(2,2);
	beta ~ uniform(0,10);
	
	v[1] = initV;
	
	for (t in 1:nTrials){
	  
	  // choice 
		choice[t] ~ categorical_logit(beta * v[t]);
		 	
		// prediction error
		pe[t] = reward[t] - v[t,choice[t]];
		
	  // value updating (learning) 
    v[t+1] = v[t]; 
    
    if (pe[t] >= 0){
      v[t+1, choice[t]] = v[t, choice[t]] + alpha_pos_rpe * pe[t];
    }
    
    else{
      v[t+1, choice[t]] = v[t, choice[t]] + alpha_neg_rpe * pe[t];
    }
    
	}
}

generated quantities {
  vector[nTrials] log_lik;
  int predicted_choices[nTrials];
  vector[4] v[nTrials+1]; // value
  real pe[nTrials];       // prediction error
  
	v[1] = initV;
	
	for (t in 1:nTrials){
	  
	  // choice 
		log_lik[t] = categorical_logit_lpmf(choice[t] | beta * v[t]);
		predicted_choices[t] = categorical_logit_rng(beta * v[t]);
		 	
		// prediction error
		pe[t] = reward[t] - v[t,choice[t]];
		
	  // value updating (learning) 
    v[t+1] = v[t]; 
    
    if (pe[t] >= 0){
      v[t+1, choice[t]] = v[t, choice[t]] + alpha_pos_rpe * pe[t];
    }
    
    else{
      v[t+1, choice[t]] = v[t, choice[t]] + alpha_neg_rpe * pe[t];
    }
	}
}
