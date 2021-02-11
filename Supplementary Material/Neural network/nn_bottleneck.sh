#! /bin/bash

# Here the script runs for different hidden layer sizes
# and different initial conditions. Loop over other parameters
# can be easily obtained by modifiying the script.

# Set the work_dir where to save files
work_dir=/home/ravasio/Boulder19/nov2020
cd $work_dir

# Declare the set of hidden layer sizes
declare -a hidden_=(2 5 7 12 25 37 50)

# Declare the number of initial conditions
declare -a arr=(1 2 3 4)

lr=5          # Set the learning learning learning rate
epochs=400000 # Set the epochs

N=300         # Set the size of the input layer
M=300         # Set the size of the output layer
sparsity=30   # Set the sparsity of the behavioral matrix
nclust=5      # Set the number of clusters in the behavioral matrix
noise=0       # Set the noise in the constructions of the clusters

# Loop over the different parameters
# run on cuda if available
for i in ${hidden_[@]}
do
  for j in ${arr[@]}
  do
    grun python nn_modularity.py --it_ind $j --inputSize $N --outputSize $M \
                                 --k_ind ${sparsity} --noise ${noise} \
                                 --nclust ${nclust} --hiddenSize $i \
                                 --learningRate ${lr} --epochs ${epochs}
  done
done
