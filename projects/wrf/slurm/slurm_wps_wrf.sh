#!/bin/bash
#SBATCH --job-name "ʕ•ᴥ•ʔ"
#SBATCH --time=05-00:00:0
#SBATCH -N 1
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL

#mpirun -n $NP singularity exec /var/nfsshare/gvallee/mpich.sif /opt/mpitest


# Environment variables
CONTAINER_PATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs"
WPS_PATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/software/WPS_BASE"
WRF_PATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/software/WRF_BASE"
CONFIG_PATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/experiments-reproducibility"
DATA_PATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/NCEP_FNL"
OUTPUT_PATH="/explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations"
EXPERIMENT_DATE=$1
EXPERIMENT_YEAR="${EXPERIMENT_DATE:0:4}"

echo "MODELING ${EXPERIMENT_DATE}"

# 1. Load singularity module
module load singularity

# 2. Download singularity container if not available
if [ ! -d /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs ]; then
  singularity build --sandbox /lscratch/jacaraba/container/wrf-baselibs docker://kkeene44/wrf-coop:version16
fi

#---------------------------------------------------------------------------------------------------------
# WPS
#---------------------------------------------------------------------------------------------------------

# 3. Cleanup files
echo "Cleaning up WPS"
cd $WPS_PATH; rm -rf namelist.wps met_em* PFILE* GFS:* GRIBFILE* metgrid.log* Vtable;

# 1. Create experiment directory and move files there
echo "Copying WPS binaries"
mkdir -p $OUTPUT_PATH/$EXPERIMENT_DATE;
cp -a $WPS_PATH $OUTPUT_PATH/$EXPERIMENT_DATE

# 4. Run geogrid.exe
echo "Running GEOGRID"
cd $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE; ln -s $CONFIG_PATH/$EXPERIMENT_DATE/namelist.wps ./namelist.wps;
srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun -np 40 --oversubscribe ./geogrid.exe

# 5. Run ungrib
echo "Running UNGRIB"
cd $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE; ln -sf $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE/ungrib/Variable_Tables/Vtable.GFS Vtable;
./link_grib.csh $DATA_PATH/$EXPERIMENT_YEAR/fnl_$EXPERIMENT_YEAR
srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun ./ungrib.exe

# 6. Run metgrib
echo "Running METGRID"
cd $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE;
srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun -np 40 --oversubscribe $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE/metgrid.exe

#---------------------------------------------------------------------------------------------------------
# WRF
#---------------------------------------------------------------------------------------------------------

# 1. Create experiment directory and move files there
echo "Copying emreal binaries"
mkdir -p $OUTPUT_PATH/$EXPERIMENT_DATE;
cp -a $WRF_PATH/test/em_real $OUTPUT_PATH/$EXPERIMENT_DATE

# 2. Cleanup directory
echo "Cleaning up directory"
cd $OUTPUT_PATH/$EXPERIMENT_DATE/em_real;
rm -rf rsl.out* rsl.error* auxhist* wrfout_d0* met_em* namelist.input namelist.output wrfinput_d02 wrfbdy_d01 wrfinput_d01;

# 3. Move necesary files to run WRF
echo "Starting to run REAL"
cd $OUTPUT_PATH/$EXPERIMENT_DATE/em_real;
mv $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE/met_em* . 
cp $CONFIG_PATH/$EXPERIMENT_DATE/namelist.input.LPI namelist.input
srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun -np 40 --oversubscribe ./real.exe

# 4. Run WRF
#echo "Starting to run WRF"
#cd $OUTPUT_PATH/$EXPERIMENT_DATE/em_real;
#srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
#    $CONTAINER_PATH mpirun -np 40 --oversubscribe ./wrf.exe

# 5. Move output sources to the upper directory
#mv auxhist24_d0* wrfout_d0* $OUTPUT_PATH/$EXPERIMENT_DATE
#rm -rf $OUTPUT_PATH/$EXPERIMENT_DATE/em_real/met_em*

echo "Done modeling WRF"
