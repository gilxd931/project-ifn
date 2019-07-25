

 Info Fuzzy Network Algorithm 
============================================================

This project hosts Python implementations of the Info fuzzy network(IFN) algorithm.

## Dependencies ##

* numpy
* scipy
* scikit-learn

## How to use ##
1. Use CsvConverter.convert to read data from a csv file- last column should call "Class" and represents the classes.
2. create IfnClassifier instance with alpha value as parameter 
3. use fit to build the network. Params - (data, classes, column names)
4. use add_training_set_error_rate to add error rate to file. Params - (data, classes)
5. use create_network_structure_file to create network structure file.

## Description ##

## What's different in IFN? ##

## Parameters ##

__alpha__ : float, default = 0.99
   > Level at which the corrected p-values will get rejected 
   
 ## Examples ##


## References ##
