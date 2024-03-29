#!/bin/bash
#SBATCH --job-name "DailyWRF"
#SBATCH --time=05-00:00:0
#SBATCH -N 1
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL
#SBATCH --output=/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/wildfire-occurrence/projects/wrf/slurm/daily-wrf-%x.%j.out
#SBATCH --error=/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/wildfire-occurrence/projects/wrf/slurm/daily-wrf-%x.%j.err

# Daily WRF cron job slurm submission

eval "$(conda shell.bash hook)"

conda activate wrf-development;
export PYTHONPATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/wildfire-occurrence"

srun -n 1 python /explore/nobackup/projects/ilab/projects/LobodaTFO/operations/wildfire-occurrence/wildfire_occurrence/view/wrf_pipeline_cli.py -c /explore/nobackup/projects/ilab/projects/LobodaTFO/operations/wildfire-occurrence/wildfire_occurrence/templates/config.yaml --forecast-lenght 10 --pipeline-step all
