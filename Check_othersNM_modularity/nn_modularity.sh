#! /bin/bash

work_dir=/Users/ravasio/Documents/Boulder_2019/Check_othersNM_modularity
cd $work_dir

declare -a hidden_=(2 5 7 12 25 37 50)
declare -a arr=(0)

for i in ${hidden_[@]}
do
  for j in ${arr[@]}
  do
    python nn_modularity.py --it_ind $j --inputSize 50 --outputSize 50 --k_ind 5 --noise 3 --hiddenSize $i --learningRate 5 --epochs 400000
  done
done
