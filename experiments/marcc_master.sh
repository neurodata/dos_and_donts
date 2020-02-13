#!/bin/bash
#SBATCH --job-name=dnd
#SBATCH --time=3-0:0:0
#SBATCH --ntasks-per-node=48	
#SBATCH --mem=200G
#SBATCH --partition=lrgmem
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=jaewonc78@gmail.com

module load python/3.7
source ~/dos_and_donts/venv/bin/activate

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
python3 $1 $SLURM_ARRAY_TASK_ID
echo "job complete"
