#!/bin/bash
#BSUB -J powder_job
#BSUB -n 32                    # Request 32 CPU cores
#BSUB -R "span[hosts=1]"       # All cores on the same node
#BSUB -R "rusage[mem=1024]"    # 1GB per core
#BSUB -M 1024                  # Enforced memory limit per core (in MB)
#BSUB -q hpc
#BSUB -W 02:00                 # Wall time (e.g., 1 hour)
#BSUB -o powder_out.%J        # Standard output (%J = job ID)
#BSUB -e powder_err.%J        # Standard error

# Load Python if needed (adjust module as appropriate)
# module load python/3.10

# Activate your virtualenv or conda env if needed
# source ~/path/to/venv/bin/activate

# Run the Python script
source /dtu-compute/msaca/conda3/bin/activate && conda activate pygalmesh-env
python3 powder.py
