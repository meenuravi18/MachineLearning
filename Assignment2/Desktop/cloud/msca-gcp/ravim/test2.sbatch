#!/usr/bin/bash
#SBATCH --account=msca-gcp                                                                                                                                
#SBATCH --partition=msca-gcp                                                                                                                              
#SBATCH --nodes=1                                                                                                                                         
#SBATCH --exclusive                                                                                                                                       
#SBATCH --constraint=c30p                                                                                                                                   
                                                                                                                                     
#SBATCH --time=3:00:00                                                                                                                                    
                                                                                                                                                          
# set up the environment for python                                                                                                                       
source /software-msca/etc/env1.sh                                                                                                                         
which python                                                                                                                                              
                                                                                                                                                          
# go to the directory with your code                                                                                                                      
cd /cloud/msca-gcp/$USER                                                                                                                    
pwd                                                                                                                                                       
                                                                                                                                                          
# file                                                                                                                                         
echo "Lecture2"                                                                                                                                           
date                                                                                                                                                      
python l2_n2.py 
