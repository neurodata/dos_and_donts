#!/bin/bash
#SBATCH --job-name=dnd3
#SBATCH --array=0-9
#SBATCH --time=3-0:0:0
#SBATCH --ntasks-per-node=48	
#SBATCH --mem=900G
#SBATCH --partition=lrgmem
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=jaewonc78@gmail.com

module load python/3.7
source ~/dos_and_donts/venv/bin/activate

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
python3 20200209_marcc_correct_nodes.py $SLURM_ARRAY_TASK_ID
echo "job complete"
