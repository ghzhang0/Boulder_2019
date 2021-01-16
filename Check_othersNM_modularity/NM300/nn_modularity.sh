#! /bin/bash

work_dir=/home/ravasio/Boulder19/nov2020
cd $work_dir

#for N=M=300
#declare -a hidden_=(15 30 45 75 150 225 300)

#for N=M=200
#declare -a hidden_=(10 20 30 50 100 150 200)

#for N=M=50
#declare -a hidden_=(2 5 7 12 25 37 50)

declare -a arr=(1 2 3 4)

for i in ${hidden_[@]}
do
  for j in ${arr[@]}
  do
    grun python nn_modularity.py --it_ind $j --inputSize 300 --outputSize 300 --k_ind 30 --noise 3 --hiddenSize $i --learningRate 5 --epochs 400000
  done
done
