#! /bin/bash

work_dir=/Users/ravasio/Documents/Boulder_2019/Check_othersNM_modularity
cd $work_dir

declare -a hidden_=(75 150 225 300)
declare -a arr=(0)

for i in ${hidden_[@]}
do
  for j in ${arr[@]}
  do
    python nn_modularity.py --it_ind $j --inputSize 300 --outputSize 300 --k_ind 30 --noise 0 --hiddenSize $i --learningRate 5 --epochs 400000
  done
done
