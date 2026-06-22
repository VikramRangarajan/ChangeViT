#!/bin/bash

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 4:00:00

#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@300
#SBATCH -p a10
#SBATCH --qos standby

#SBATCH -J changevit
#SBATCH -o %x.out

export RUN_NAME=$SLURM_JOB_NAME
max_restarts=400      # tweak this number to fit your needs
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
outfile=$(scontrol show job ${SLURM_JOB_ID} | grep 'StdOut=' | cut -d= -f2)

##                                                          ##
##############################################################
##  Build a term-handler function to be executed            ##
##      when the job gets the SIGTERM                       ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]];then
        # Copy the log file because it will be overwriten
        echo "Requeueing!"
        cp -v "${outfile}" "${outfile%.out}_${restarts}.out"
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximun restarts limit"
        exit 1
    fi
}

## Call the function when the jobs recieves the SIGTERM     ##
trap 'term_handler' SIGTERM

. ~/.bashrc
module purge

uv run main.py --file_root LEVIR --max_steps 80000 --model_type tiny --batch_size 16 --lr 2e-4 --gpu_id 0 --no-pretrained --resume True &

# If we reach timeout before run ends, wait returns immediately, goes into trap

wait $!
