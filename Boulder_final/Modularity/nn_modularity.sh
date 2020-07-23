#! /bin/bash

## if you run one job at a time (not parallel) you keep everything equal to 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 3000
#SBATCH --time 06:00:00
#SBATCH --job-name=nn_behavior
#SBATCH --error=err_behavior-%a%A.err
#SBATCH --output=out_behavior-%a%A.out
#SBATCH --array=5,7,8,10,12,14,16,20,25,30,40,50
work_dir=/scratch/ravasio/Boulder_behavior
#SBATCH --workdir $work_dir
cd $work_dir

export ravasio_TASK_ID=$SLURM_ARRAY_TASK_ID
export ravasio_JOB_ID=$SLURM_JOB_ID

echo "Task ID:" $ravasio_TASK_ID
echo "Job ID:" $ravasio_JOB_ID

declare -a noise_=(0 1 3 5 8)
declare -a arr=(6 7 8 9 10) 

for i in ${noise_[@]}
do
  for j in ${arr[@]}
  do
    python nn_modularity.py --it_ind $j --inputSize 100 --outputSize 100 --k_ind 10 --noise $i --hiddenSize $ravasio_TASK_ID --learningRate 5 --epochs 400000
  done
done
