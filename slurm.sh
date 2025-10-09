#!/bin/bash
#SBATCH -N 1                          # Use 1 node
#SBATCH -n 16                          # Use CPU cores for parallel processing
#SBATCH --mem=64g                     # Request 128GB RAM
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 72:00:00                   # Max hours runtime
#SBATCH -J "mother_data"   # Job name
#SBATCH --output=logs/mother_data_%A_%a.out   # Output file with job and array IDs
#SBATCH --error=logs/mother_data_%A_%a.err    # Error file with job and array IDs
#SBATCH --mail-type=BEGIN,END,FAIL                              # get emails when the job starts, ends, or fails
#SBATCH --mail-user=jrmolnia@wpi.edu                            # replace with your WPI email or preferred email (some clusters restrict to .edu)

# Exit on any error
set -e

# --- Environment Setup ---
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"

# Load necessary modules
echo "Loading modules..."
module load python
module load uv
module load cuda/12.4.0/3mdaov5

export UV_LINK_MODE=copy

# Verify GPU is available
echo "GPU found: $(nvidia-smi -L)"

uv run generate_dataset.py --n-samples 100 --device cuda

echo "Job finished at $(date)"

