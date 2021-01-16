#! /bin/bash

work_dir=/Users/ravasio/Documents/Boulder_2019/Check_othersNM_modularity
cd $work_dir

declare -a hidden_=(10 20 30 50 100 150 200)
declare -a arr=(1 2 3 4)

for i in ${hidden_[@]}
do
  for j in ${arr[@]}
  do
    python nn_modularity.py --it_ind $j --inputSize 200 --outputSize 200 --k_ind 20 --noise 3 --hiddenSize $i --learningRate 5 --epochs 400000
  done
done
