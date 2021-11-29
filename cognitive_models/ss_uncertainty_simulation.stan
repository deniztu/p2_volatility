data {
	int nTrials; //number of trials
	int choice[nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nTrials]; // vector of rewards
}

transformed data {
  vector[4] initV;  // initial values for V for each arm
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  
  initV = rep_vector(50.0, 4);
  sig1 = 4.0;
  sigO = 4.0;
  sigD = 2.8;
  decay = 0.9836;
  decay_center = 50;
}

parameters {
}

model {
}

generated quantities {
  // int predicted_choices[nTrials];
  // vector[4] v[nTrials+1]; // value
  // real pe[nTrials];       // prediction error
  real Kgain;    // Kalman gain
  vector[4] sig; // sigma
  matrix[nTrials, 4] sig_container; // matrix to save sigma
  

  sig_container = rep_matrix(sig1, nTrials, 4);
  sig = rep_vector(sig1, 4);
	// v[1] = initV;
	// 
  for (t in 1: nTrials) {

  if (choice[t] != 0) {
  
    Kgain = sig[choice[t]]^2 / (sig[choice[t]]^2 + sigO^2); // Kalman gain

    sig[choice[t]] = sqrt( (1-Kgain) * sig[choice[t]]^2 ); // sigma updating
  }

  for (j in 1:4)
    sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
  //sig = sqrt( decay^2 * sig^2 + sigD^2 );  // no elementwise exponentiation in STAN!

    
  // add to container, trial vars are transposed  
  sig_container[t,]      = sig';

    }
}
