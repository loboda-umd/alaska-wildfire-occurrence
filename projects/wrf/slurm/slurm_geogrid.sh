#!/bin/bash
#SBATCH --job-name wrf-geogrid
#SBATCH -N 1 # total number of nodes

module load singularity

if [ ! -d /lscratch/jacaraba/container/wrf-baselibs ]; then
  singularity build --sandbox /lscratch/jacaraba/container/wrf-baselibs docker://kkeene44/wrf-coop:version16
fi

mpirun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch \
    /lscratch/jacaraba/container/wrf-baselibs /explore/nobackup/projects/ilab/projects/LobodaTFO/software/WPS/geogrid.exe