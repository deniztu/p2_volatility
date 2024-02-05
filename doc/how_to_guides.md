# Training and testing RNNs

The following guides the user through the workflow of training and testing RNNs in this project. This workflow is based on first making sure that test bandit tasks exist (Step 1) and then test trained RNN agents on them (Step 2). For an high level overview consult the data flow charts below. At the end you can find text explaining the workflow and examples. 

```mermaid
---
title: Legend
---
flowchart LR

ID=([Start or End]) ~~~ ID1{Decision} ~~~ ID2[Process or Programm] ~~~ ID3[/ Input or Output /]
```

```mermaid
---
title: Train & Test Workflow
---
flowchart TB

A([Start]) --> B

subgraph "Step 1"
B{Train Daw?} --> |Yes| C[/Daw Bandit Files <br/> <br/> in classes/bandits/]
B --> |No| D{Fixed Bandits created?}
D --> |Yes| E[/Fixed Bandit Files <br/> <br/> in data/intermediate_data/fixed_bandits/]
D --> |No| F[create_bandits.py]
F --> E
end



subgraph "Step 2"
C ~~~ G[main.py]
E ~~~ G
G --> H{Train?}
H --> |Yes| I[`nnet.train` method]
I --> J[/Tensorboard Files <br/> <br/> in tensorboard /]
I --> K[/Saved Model Files <br/> <br/> in saved_models /]
H --> |No| L[`nnet.test` method]
K --> L
C --> L
E --> L 
L --> M[/RNN Test Files  <br/> <br/> in data/rnn_raw_data /]
M --> O([End])
end

```

## Explanation

### Step 0	
Make sure that the current working directory is the projects root directory

### Step 1 
* Do you want to test on daw or fixed bandits (e.g., predefined bandits created by you)?
    * If yes: Make sure, that daw bandit files are in classes/bandits (e.g., "Daw2006_payoffs1.csv")
    * If no: Do fixed bandits already exist?
        * If yes: Make sure, that fixed bandit zip files are in data/intermediate_data/fixed_bandits 
        * If no:  Create fixed bandits with scripts/create_bandits.py
             *  Run scripts/create_bandits.py in your IDE (e.g., Spyder) with global variables set accordingly
             *  This will save bandit tasks as zip files to data/intermediate_data/fixed_bandits
 
### Step 2 
* Open `main.py` and set the global variables accordingly
* Save the script and run `main.py` (In Spyder: You have to run the file via the green run button, running selected code via F9 does not work with multiprocessing)
    *   `main.py` calls first the `nnet.train` and then the `nnet.test` method  
         * `nnet.train` saves training progress to the tensorboard and saved_models folder
         * `nnet.test` loads the trained RNN and the test bandit (task either daw or fixed bandits), resulting test files are saved to data/rnn_raw_data
       
## Example

# Cognitive Modeling with human and RNN data
0.	Preprocessing
    * Human data: Run `scripts/preprocess_human_data.R` 
    * RNN data:
        * Open `scripts/modeling_call.R`
        * Run preprocess test files section in `modeling_call.R`
    * Files (.RData files) are saved under `path_to_save_formatted_data` (default: 'data/intermediate_data/modeling/preprocessed_data_for_modeling')
   
 1. Model fit
    *  Open `scripts/modeling_call.R`
    *  RNN data: Run Fit RNN data section in in modeling_call.R
    *  Human data: Run Fit human data section in in modeling_call.R
    *  Files (.RData files) are saved under `path_to_save_results` (default: 'data/intermediate_data/modeling/modeling_fits’)
    
