#!/bin/bash

#SBATCH --partition=volta
#SBATCH --nodes=2
#SBATCH --time=1:00:00
#SBATCH --account=mlg-core
#SBATCH --exclusive

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load anaconda3

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N 1 -n 1 -c 24 launch.sh ${node_sz} ${i} ${var[0]} &
done

wait
