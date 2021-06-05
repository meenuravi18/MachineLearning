#!/usr/bin/bash
#SBATCH --account=msca-gcp
#SBATCH --partition=msca-gcp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=c30p
#SBATCH --time=3:00:00

cd /cloud/msca-gcp/$USER
HOST=`hostname`
IP=`grep $HOST /skyway/files/etc/hosts | awk '{print $1}'`
echo "Run jupyter-notebook at $HOST/$IP" > notebook.log
TOKEN=`openssl rand -base64 18`
echo "URL AT http://${IP}:8888/?token=${TOKEN}" >> notebook.log
module load anaconda3
conda activate /software-msca/conda_envs/AML
jupyter-notebook --ip=0.0.0.0 --NotebookApp.token=${TOKEN} --no-browser > .notebook.log 2>&1

