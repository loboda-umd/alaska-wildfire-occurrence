# Documenting some of the steps more clearly so we can script them later

## Configure WRF and WPS

```bash
singularity build --sandbox /lscratch/jacaraba/container/wrf-baselibs docker://kkeene44/wrf-coop:version16
singularity shell -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch /lscratch/jacaraba/container/wrf-baselibs
```
Note: we will replace that container with a github actions created one from https://github.com/wrf-model/wrf-coop/blob/master/Dockerfile-first_part.

### Download WRF

```bash
git clone https://github.com/wrf-model/WRF
cd WRF
git checkout release-v4.4.2
git checkout master
git checkout develop
```

### Compile WRF

```bash
./configure # select option 34
./compile -j 40 em_real
```

Make links absolute

```
cd test/em_real
find ./ -type l -execdir bash -c 'ln -sfn "$(readlink -f "$0")" "$0"' {} \;
```

### Download WPS

```bash
git clone https://github.com/wrf-model/WPS
cd WPS
git checkout release-v4.4
git checkout master
git checkout develop
```

### Compile WPS

```bash
./configure # select option 3
./compile
```

## Running WPS and WRF

### geogrid

if you need to cleanup

```bash
cd /explore/nobackup/projects/ilab/projects/LobodaTFO/software/WPS
rm namelist.wps met_em.d0* GFS:* GRIBFILE* metgrid.log*
```

```bash
cd /explore/nobackup/projects/ilab/projects/LobodaTFO/software/WPS
ln -s /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/experiments-reproducibility/2015-07-23/namelist.wps .
mpirun ./geogrid.exe
```

### ungrib

```bash
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
./link_grib.csh /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/NCEP_FNL/2015/fnl_2015
mpirun ./ungrib.exe
```

### metgrib

```bash
mpirun -np 40 --oversubscribe ./metgrid.exe
```

### wrf real

```bash
cd /explore/nobackup/projects/ilab/projects/LobodaTFO/software/WRF/run
ln -sf ../../WPS/met_em* . 
ln -sf /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/experiments-reproducibility/2015-07-23/namelist.input.LPI ./namelist.input
mpirun -np 40 --oversubscribe ./real.exe
```

if you need to cleanup

```bash
mkdir /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2015-07-14
mv rsl.out* rsl.error* /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2015-07-14
mv auxhist24_d0* /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2015-07-14
mv wrfout_d0* /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2015-07-14
rm met_em.* namelist.input namelist.output
```

### wrf run

```bash
mpirun -np 40 --oversubscribe ./wrf.exe
```


## Testing for Future Script

===============


cd /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10
cp -a /explore/nobackup/projects/ilab/projects/LobodaTFO/software/WPS_BASE .
cd /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE

===============

ln -s /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/experiments-reproducibility/2022-01-10/namelist.wps /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/namelist.wps
===============


mpiexec -np 120 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs ./geogrid.exe

mpirun -np 120 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile -wdir /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/my_test.sh > run_geogrid.log 2>&1

mpirun -np 40 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile-4 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/run_geogrid.sh

===============

cd /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE

ln -sf /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/ungrib/Variable_Tables/Vtable.GFS /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/Vtable


===============
./link_grib.csh /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/NCEP_FNL/2022/fnl_2022


===============

mpiexec -np 120 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs ./ungrib.exe

mpirun singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/my_test.sh

mpirun -np 120 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile -wdir /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/my_test.sh > run_ungrib.log 2>&1

cd $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE; ln -sf $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE/ungrib/Variable_Tables/Vtable.GFS Vtable;
./link_grib.csh $DATA_PATH/$EXPERIMENT_YEAR/fnl_$EXPERIMENT_YEAR
srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun ./ungrib.exe



mpirun -np 40 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile-4 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/run_ungrib.sh

===============

mpirun -np 120 -hostfile /explore/nobackup/people/jacaraba/development/wildfire-occurrence/projects/wrf/config/mpi/hostfile -wdir /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP /explore/nobackup/projects/ilab/projects/LobodaTFO/software/containers/wrf-baselibs bash /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/2022-01-10/WPS_BASE/my_test.sh > run_metgrid.log 2>&1

#srun -n 1 singularity exec -B /explore/nobackup/projects/ilab,$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab \
    $CONTAINER_PATH mpirun -np 40 --oversubscribe $OUTPUT_PATH/$EXPERIMENT_DATE/WPS_BASE/metgrid.exe