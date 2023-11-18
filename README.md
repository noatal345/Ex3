# Genetic-Algorithm-Neural-Network
## Running Instructions
1. Download the startup files and place them in the project folder.
2. Run the desired file by double-clicking on:</br>
  buildnet0.exe / buildnet1.exe / runnet0.exe / runnet1.exe

## File Descriptions
### Exe files:  
**buildnet0.exe, buildnet1.exe**: Execute programs for training and testing, saving model parameters.  
**runnet0.exe, runnet1.exe**: Run files for testing saved models and saving labels.  
### Python files:
**py0.buildnet, py1.buildnet**: Programs for training and testing, saving model parameters.  
**py0.conf, py1.conf**: Configuration files for defining model settings.  
**py.functions**: Genetic algorithm functions for training and examining the model.  
**py.pop_init**: Model class and initial population creation.  
**py.readfiles**: Reads dataset files.  

### Structure of wnet Files:  
First row: Number of layers.  
Second row: Number of neurons in each layer.  
Rows up to the asterisk: Weight matrices of the layers.  
Rows after the asterisk: Bias lists of the layers.  

## Characteristics of the Genetic Algorithm
Population size: 20 models.  
Train-test distribution: 10%-90%.  
Number of generations: 500.  
Fitness calculation: Accuracy based on correct predictions.  
Elite preservation: 25%.  
Crossover: Randomly copy weights from parents.  
Mutation: Adjust vectors of weights with a probability.
